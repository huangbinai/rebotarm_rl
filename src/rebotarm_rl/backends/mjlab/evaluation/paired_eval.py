"""对同一Reach权重进行离线CPU MuJoCo与GPU Warp配对评估。

CPU使用mjlab编译模型，包括场景挂接与求解器设置。
不导入ROS或硬件驱动；结果仅反映当前任务，不代表实机部署验收。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from rebotarm_rl.evaluation.metrics import errors, success_summary
from .rollout import CpuRollout, pose
from rebotarm_rl.backends.mjlab.runner import RecordedRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from rebotarm_rl.backends.mjlab import registration  # 注册任务，不导入硬件。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--task', choices=['RebotArm-Reach-Mjlab', 'RebotArm-Reach-OfficialAligned-Mjlab', 'RebotArm-Reach-OfficialAligned-Orientation-Mjlab', 'RebotArm-Reach-GravityComp-Mjlab', 'RebotArm-Reach-GravityComp-FixedPenalties-Mjlab'], default='RebotArm-Reach-Mjlab')
    parser.add_argument('--backend', choices=['paired', 'gpu'], default='paired')
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--save-trajectories', action='store_true',
                        help='保留逐步误差轨迹，正式评估或诊断时使用')
    parser.add_argument('--selection-reason', help='选定此权重的依据；省略表示仅评估')
    parser.add_argument('--hold-steps', type=int, default=25,
                        help='连续成功控制步数；25步对应0.5秒')
    parser.add_argument('--exclude-initial-success', action='store_true',
                        help='重采样目标直到初始状态不满足成功阈值')
    parser.add_argument('--target-min-radius', type=float, default=0.0)
    parser.add_argument('--target-max-radius', type=float, default=0.06)
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1 or args.hold_steps < 1:
        parser.error('episodes, steps, and hold-steps must be positive')
    if not 0.0 <= args.target_min_radius <= args.target_max_radius:
        parser.error('target radius range is invalid')
    if args.hold_steps > args.steps + 1:
        parser.error('hold-steps must not exceed the recorded trajectory length')
    checkpoint = args.checkpoint.resolve(strict=True)
    task = args.task
    from rebotarm_rl.contracts.policy import contract_for_task
    contract = contract_for_task(task)
    cfg = load_env_cfg(task, play=False)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    # 固定长度比较：关闭成功及超时自动重置，独立统计首次和最终成功。
    cfg.terminations = {}
    cfg.commands['reach'].position_radius = 0.0
    cfg.commands['reach'].resampling_time_range = (1e9, 1e9)
    rl_cfg = load_rl_cfg(task)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device='cuda:0'),
                            clip_actions=rl_cfg.clip_actions)
    try:
        runner = RecordedRunner(env, asdict(rl_cfg), device='cuda:0')
        runner.load(str(checkpoint), load_cfg={'actor': True}, strict=True,
                    map_location='cuda:0')
        policy = runner.get_inference_policy(device='cuda:0')
        base = env.unwrapped
        robot = base.scene['robot']
        command = base.command_manager.get_term('reach')
        # CPU通过mj_step运行同一份编译后的训练模型。
        model = base.sim.mj_model
        site = int(robot.indexing.site_ids[command.site_id])
        qadr = robot.indexing.joint_q_adr.cpu().numpy()
        vadr = robot.indexing.joint_v_adr.cpu().numpy()
        default_q = robot.data.default_joint_pos[0].cpu().numpy()
        cpu = CpuRollout(model, qadr, vadr, site, default_q, contract, cfg.decimation, rl_cfg.clip_actions)
        data = cpu.data
        rng = np.random.default_rng(args.seed)
        rows = []
        max_initial_obs_delta = 0.0
        max_observation_delta = 0.0
        for episode in range(args.episodes):
            env.reset()
            # Clear CPU solver/control history; then copy the complete GPU joint state.
            cpu.reset(robot.data.joint_pos[0].cpu().numpy(), robot.data.joint_vel[0].cpu().numpy())
            initial_qpos = data.qpos[qadr].copy()
            initial_qvel = data.qvel[vadr].copy()
            home_pos, home_quat = pose(model, data, site)
            for _ in range(1000):
                direction = rng.normal(size=3)
                target = home_pos + direction / np.linalg.norm(direction) * rng.uniform(
                    args.target_min_radius, args.target_max_radius
                )
                initial_position_error, initial_orientation_error = errors(
                    home_pos, home_quat, target, home_quat
                )
                initial_success = (
                    initial_position_error < contract.success_position_m
                    and initial_orientation_error < contract.success_orientation_rad
                )
                if not args.exclude_initial_success or not initial_success:
                    break
            else:
                raise RuntimeError('could not sample a non-success initial target')
            target_quat = home_quat.copy()
            command.target_pos[:] = torch.as_tensor(target, device='cuda:0', dtype=torch.float32)
            command.target_quat[:] = torch.as_tensor(target_quat, device='cuda:0', dtype=torch.float32)
            results = {'cpu': [], 'gpu': []} if args.backend == 'paired' else {'gpu': []}
            q_deltas = []
            movements = []
            for step in range(args.steps + 1):
                if args.backend == 'gpu':
                    gpose = robot.data.site_pose_w[0, command.site_id].cpu().numpy()
                    pe, oe = errors(gpose[:3], gpose[3:], target, target_quat)
                    if not np.isfinite([pe, oe]).all():
                        raise ValueError('Non-finite GPU trajectory')
                    results['gpu'].append([pe, oe, bool(pe < contract.success_position_m and oe < contract.success_orientation_rad)])
                    movements.append(float(np.max(np.abs(robot.data.joint_pos[0].cpu().numpy()[:6] - initial_qpos[:6]))))
                    if step == args.steps:
                        break
                    with torch.inference_mode():
                        ga = policy(TensorDict({'actor': base.observation_manager.compute_group('actor')}, batch_size=[1]))
                    env.step(ga)
                    continue
                cpos, cquat = pose(model, data, site)
                gpose = robot.data.site_pose_w[0, command.site_id].cpu().numpy()
                for backend, p, q in [('cpu', cpos, cquat), ('gpu', gpose[:3], gpose[3:])]:
                    pe, oe = errors(p, q, target, target_quat)
                    results[backend].append([pe, oe, bool(pe < contract.success_position_m and oe < contract.success_orientation_rad)])
                q_deltas.append(float(np.max(np.abs(data.qpos[qadr] - robot.data.joint_pos[0].cpu().numpy()))))
                movements.append(float(np.max(np.abs(robot.data.joint_pos[0].cpu().numpy()[:6] - initial_qpos[:6]))))
                if step == args.steps:
                    break
                cpu_obs = cpu.observe(target, target_quat, (cpos, cquat))
                gpu_obs = base.observation_manager.compute_group('actor')
                delta = float(np.max(np.abs(cpu_obs - gpu_obs[0].cpu().numpy())))
                max_observation_delta = max(max_observation_delta, delta)
                if step == 0:
                    max_initial_obs_delta = max(max_initial_obs_delta, delta)
                    if delta > 1e-4:
                        raise ValueError(f'CPU/GPU initial observation mismatch: {delta}')
                with torch.inference_mode():
                    ca = policy(TensorDict({'actor': torch.tensor(cpu_obs[None], dtype=torch.float32,
                                                                 device='cuda:0')}, batch_size=[1]))
                    ga = policy(TensorDict({'actor': gpu_obs}, batch_size=[1]))
                cpu.step(ca)
                env.step(ga)
                if not np.isfinite(data.qpos).all() or not torch.isfinite(robot.data.joint_pos).all():
                    raise ValueError('Non-finite trajectory')
            row = {'episode': episode, 'target_position_m': target.tolist(),
                   'target_quaternion_wxyz': target_quat.tolist(),
                   'initial_joint_position_rad': initial_qpos.tolist(),
                   'initial_joint_velocity_rad_s': initial_qvel.tolist(),
                   'max_joint_trajectory_delta_rad': max(q_deltas) if q_deltas else None,
                   'max_joint_displacement_from_initial_rad': max(movements)}
            for backend in results:
                trajectory = results[backend]
                row[backend] = {
                    'final_position_error_m': trajectory[-1][0],
                    'final_orientation_error_rad': trajectory[-1][1],
                    **success_summary([v[2] for v in trajectory], args.hold_steps),
                }
                if args.save_trajectories:
                    row[backend]['error_trajectory'] = trajectory
            rows.append(row)
        summary = {}
        for backend in results:
            backend_rows = rows
            if args.exclude_initial_success:
                backend_rows = [r for r in rows if not r[backend]['initial_success']]
            summary[backend] = {
                'episodes_evaluated': len(backend_rows),
                'final_success_count': sum(r[backend]['final_success'] for r in backend_rows),
                'hold_success_count': sum(r[backend]['hold_success'] for r in backend_rows),
                'tail_success_count': sum(r[backend]['tail_success'] for r in backend_rows),
                'mean_final_position_error_m': float(np.mean([r[backend]['final_position_error_m'] for r in backend_rows])),
                'mean_final_orientation_error_rad': float(np.mean([r[backend]['final_orientation_error_rad'] for r in backend_rows])),
            }
        conditions = {
            'backends': ['cpu_mujoco', 'gpu_warp'] if args.backend == 'paired' else ['gpu_warp'], 'device': 'cuda:0',
            'deterministic_policy': True, 'clip_actions': rl_cfg.clip_actions,
            'actuator_limits': 'compiled model actuator limits',
            'automatic_termination': False,
            'target_position': {'center': 'initial TCP world position',
                                'direction': 'normalized NumPy standard normal vector',
                                'radius_uniform_m': [args.target_min_radius, args.target_max_radius]},
            'exclude_initial_success': args.exclude_initial_success,
            'hold_steps': args.hold_steps,
            'target_orientation': 'initial TCP world orientation',
            'initial_state': 'environment reset; CPU copies GPU joint position and velocity',
            'success_position_m': contract.success_position_m,
            'success_orientation_rad': contract.success_orientation_rad,
            'physics_dt_s': model.opt.timestep, 'decimation': cfg.decimation,
            'trajectories_saved': args.save_trajectories,
        }
        report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'task': task, 'contract': contract.to_dict(),
                  'conditions': conditions, 'selection_reason': args.selection_reason,
                  'seed': args.seed, 'episodes': args.episodes, 'steps': args.steps,
                  'control_dt_s': cfg.decimation * model.opt.timestep,
                  'integrator': int(model.opt.integrator), 'max_initial_obs_delta': max_initial_obs_delta if args.backend == 'paired' else None,
                  'max_observation_delta': max_observation_delta if args.backend == 'paired' else None,
                  'scope': f'Deterministic {args.backend} evaluation; alone does not establish convergence, generalization or hardware acceptance',
                  'summary': summary, 'results': rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(summary, indent=2))
        print(f'Report: {args.output}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
