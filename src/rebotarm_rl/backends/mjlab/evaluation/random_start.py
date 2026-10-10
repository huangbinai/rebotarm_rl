"""Fixed-target robustness diagnostic for existing fixed-penalty Reach weights.

This changes evaluation initial conditions, not the checkpoint's task contract.
Targets stay anchored at default FK. CPU rejection sampling checks the compiled
model's joint limits and enabled collision pairs before any GPU rollout.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from rebotarm_rl.backends.mjlab import registration  # noqa: F401
from rebotarm_rl.backends.mjlab.runner import RecordedRunner
from rebotarm_rl.backends.mjlab.rotation import rotation_error_wxyz
from rebotarm_rl.contracts.artifacts import file_hash
from rebotarm_rl.contracts.policy import REACH_GRAVITY_FIXED
from .paired_eval import pose, success_summary


def state_rejection(model, data, qadr, joint_ids, q, margin=0.01):
    """Return a rejection reason; only compiled, enabled contacts are checked.

    Arm limits have a radian margin. Gripper state is unchanged by the sampler.
    This is a geometric reset check, not a path or real hardware safety proof.
    """
    if not np.isfinite(q).all():
        return 'nonfinite'
    limits = model.jnt_range[joint_ids[:6]]
    if not model.jnt_limited[joint_ids[:6]].all():
        raise ValueError('Expected six limited arm joints')
    if np.any(q[:6] < limits[:, 0] + margin) or np.any(q[:6] > limits[:, 1] - margin):
        return 'joint_limit'
    mujoco.mj_resetData(model, data)
    data.qpos[qadr] = q
    mujoco.mj_forward(model, data)
    if any(c.dist <= 0 for c in data.contact):
        return 'self_collision'
    return None


def sample_initial_states(model, qadr, joint_ids, default_q, amplitude, count, seed):
    """Independent uniform offsets, rejection (never clipping), deterministic seed.

    Each episode has its own random stream so rejection does not shift later
    episodes. Across amplitudes, proposals share the same normalized directions.
    """
    data = mujoco.MjData(model)
    rejected = {'joint_limit': 0, 'self_collision': 0, 'nonfinite': 0}
    states, attempts = [], []
    for episode in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, episode]))
        for attempt in range(1, 10001):
            q = default_q.copy()
            q[:6] += rng.uniform(-amplitude, amplitude, 6)
            reason = state_rejection(model, data, qadr, joint_ids, q)
            if reason is None:
                states.append(q)
                attempts.append(attempt)
                break
            rejected[reason] += 1
        else:
            raise RuntimeError(f'No valid initial state for episode {episode}')
    return np.asarray(states), {'rejected': rejected, 'attempts_per_episode': attempts}


def trajectory_metrics(poses, joints, target, target_quat, hold_samples, dt):
    """Metrics include t=0; a 26-sample hold spans 0.50 s at 50 Hz."""
    pe = np.linalg.norm(poses[:, :3] - target, axis=-1)
    quats = poses[:, 3:] / np.linalg.norm(poses[:, 3:], axis=-1, keepdims=True)
    tq = target_quat / np.linalg.norm(target_quat)
    oe = 2 * np.arccos(np.clip(np.abs(quats @ tq), 0, 1))
    flags = (pe < .01) & (oe < np.deg2rad(3.0))
    result = success_summary(flags.tolist(), hold_samples)
    first = result['first_success_step']
    ends = [i for i in range(hold_samples - 1, len(flags))
            if flags[i - hold_samples + 1:i + 1].all()]
    tail = poses[-hold_samples:, :3]
    result.update(
        first_success_time_s=None if first is None else first * dt,
        first_hold_completion_time_s=None if not ends else ends[0] * dt,
        final_position_error_m=float(pe[-1]), final_orientation_error_rad=float(oe[-1]),
        tail_position_jitter_rms_m=float(np.sqrt(np.mean(np.sum((tail - tail.mean(0)) ** 2, axis=1)))),
        tail_tcp_speed_rms_m_s=float(np.sqrt(np.mean(np.sum((np.diff(tail, axis=0) / dt) ** 2, axis=1)))),
        joint_range_rad=np.ptp(joints[:, :6], axis=0).tolist(),
        joint_total_travel_rad=np.abs(np.diff(joints[:, :6], axis=0)).sum(0).tolist(),
        max_joint_displacement_rad=float(np.abs(joints[:, :6] - joints[0, :6]).max()),
        position_error_trajectory_m=pe.tolist(), orientation_error_trajectory_rad=oe.tolist(),
    )
    return result


def summarize(rows):
    def mean(key, selected=rows):
        values = [r[key] for r in selected if r[key] is not None]
        return None if not values else float(np.mean(values))
    noninitial = [r for r in rows if not r['initial_success']]
    return {
        'episodes': len(rows),
        **{f'{key}_count': sum(r[key] for r in rows)
           for key in ('initial_success', 'final_success', 'hold_success', 'tail_success')},
        'noninitial_episodes': len(noninitial),
        'noninitial_tail_success_count': sum(r['tail_success'] for r in noninitial),
        'mean_final_position_error_m': mean('final_position_error_m'),
        'mean_final_orientation_error_rad': mean('final_orientation_error_rad'),
        'mean_first_success_time_s_successes_only': mean('first_success_time_s'),
        'mean_first_hold_completion_time_s_successes_only': mean('first_hold_completion_time_s'),
        'mean_tail_position_jitter_rms_m': mean('tail_position_jitter_rms_m'),
        'mean_tail_tcp_speed_rms_m_s': mean('tail_tcp_speed_rms_m_s'),
        'mean_max_joint_displacement_rad': mean('max_joint_displacement_rad'),
        'mean_joint_range_rad': np.mean([r['joint_range_rad'] for r in rows], axis=0).tolist(),
        'mean_joint_total_travel_rad': np.mean([r['joint_total_travel_rad'] for r in rows], axis=0).tolist(),
    }


def trajectory_validity(model, qadr, joint_ids, joints):
    """Check recorded 50 Hz states, not unrecorded physics substeps."""
    data = mujoco.MjData(model)
    violations = {'joint_limit': 0, 'self_collision': 0, 'nonfinite': 0}
    for q in joints:
        reason = state_rejection(model, data, qadr, joint_ids, q, margin=0.0)
        if reason is not None:
            violations[reason] += 1
    return violations


def cpu_rollout(model, qadr, vadr, site, initial, default, target, quat,
                policy, rl_cfg, steps, decimation, device):
    data = mujoco.MjData(model)
    data.qpos[qadr] = initial
    actuator_ids = [model.actuator(f'robot/joint{i}_torque').id for i in range(1, 7)]
    last = np.zeros(6, dtype=np.float32)
    poses, joints, observations = [], [], []
    for step in range(steps + 1):
        p, q = pose(model, data, site)
        obs = np.concatenate((data.qpos[qadr[:6]] - default[:6], data.qvel[vadr[:6]],
                              p - target, rotation_error_wxyz(torch.from_numpy(q),
                              torch.from_numpy(quat)).numpy(), last))
        poses.append(np.r_[p, q]); joints.append(data.qpos[qadr].copy()); observations.append(obs)
        if step == steps:
            break
        with torch.inference_mode():
            action = policy(TensorDict({'actor': torch.as_tensor(obs[None], dtype=torch.float32,
                                                                 device=device)}, batch_size=[1]))
        if rl_cfg.clip_actions is not None:
            action = action.clamp(-rl_cfg.clip_actions, rl_cfg.clip_actions)
        last = action[0].cpu().numpy()
        data.ctrl[:] = 0
        data.ctrl[actuator_ids] = default[:6] + .5 * last
        for _ in range(decimation):
            mujoco.mj_step(model, data)
    return np.asarray(poses), np.asarray(joints), np.asarray(observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--amplitudes', type=float, nargs='+', default=[0, .025, .05, .1, .2, .4])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--hold-samples', type=int, default=26)
    parser.add_argument('--target-seed', type=int, default=130000)
    parser.add_argument('--initial-seed', type=int, default=130001)
    parser.add_argument('--paired-episodes', type=int, default=0)
    parser.add_argument('--output-name', default='task2_random_start_v1')
    parser.add_argument('--output-dir', type=Path, help='Only for temporary verification; default is each run/eval')
    args = parser.parse_args()
    if (args.episodes < 1 or args.steps < 1 or not 2 <= args.hold_samples <= args.steps + 1
            or not 0 <= args.paired_episodes <= args.episodes
            or any(not np.isfinite(a) or a < 0 for a in args.amplitudes)
            or min(args.target_seed, args.initial_seed) < 0):
        parser.error('Invalid sample count, amplitude or seed')
    if Path(args.output_name).name != args.output_name:
        parser.error('output-name must be a filename stem')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    device = 'cuda:0'
    contract = REACH_GRAVITY_FIXED
    cfg = load_env_cfg(contract.task_id)
    cfg.scene.num_envs = args.episodes
    cfg.seed = args.target_seed
    cfg.terminations = {}
    cfg.commands['reach'].position_radius = 0.0
    cfg.commands['reach'].resampling_time_range = (1e9, 1e9)
    rl_cfg = load_rl_cfg(contract.task_id)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=device), clip_actions=rl_cfg.clip_actions)
    try:
        base = env.unwrapped
        robot = base.scene['robot']
        command = base.command_manager.get_term('reach')
        if torch.count_nonzero(base.scene.env_origins):
            raise ValueError('Diagnostic requires coincident independent world origins')
        model = base.sim.mj_model
        qadr = robot.indexing.joint_q_adr.cpu().numpy()
        vadr = robot.indexing.joint_v_adr.cpu().numpy()
        joint_ids = np.array([model.joint(f'robot/{name}').id for name in robot.joint_names])
        default = robot.data.default_joint_pos[0].cpu().numpy().astype(float)
        site = int(robot.indexing.site_ids[command.site_id])
        data = mujoco.MjData(model)
        if state_rejection(model, data, qadr, joint_ids, default):
            raise ValueError('Default pose is invalid')
        anchor, quat = pose(model, data, site)
        rng = np.random.default_rng(args.target_seed)
        targets = []
        for _ in range(args.episodes):
            direction = rng.normal(size=3)
            targets.append(anchor + direction / np.linalg.norm(direction) * rng.uniform(.02, .06))
        targets = np.asarray(targets)
        states = {a: sample_initial_states(model, qadr, joint_ids, default, a,
                                           args.episodes, args.initial_seed) for a in args.amplitudes}
        runner = RecordedRunner(env, asdict(rl_cfg), device=device)
        for checkpoint in args.checkpoints:
            checkpoint = checkpoint.resolve(strict=True)
            output_dir = args.output_dir or checkpoint.parent / 'eval'
            output = output_dir / f'{args.output_name}_{checkpoint.stem}_seed{args.target_seed}.json'
            if output.exists():
                raise FileExistsError(output)
            runner.load(str(checkpoint), load_cfg={'actor': True}, strict=True, map_location=device)
            policy = runner.get_inference_policy(device=device)
            levels = []
            for amplitude in args.amplitudes:
                initial, sampling = states[amplitude]
                env.reset()
                q = torch.as_tensor(initial, device=device, dtype=torch.float32)
                robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                base.sim.forward(); base.sim.sense()
                command.target_pos[:] = torch.as_tensor(targets, device=device, dtype=torch.float32)
                command.target_quat[:] = torch.as_tensor(quat, device=device, dtype=torch.float32)
                poses, joints, observations = [], [], []
                for step in range(args.steps + 1):
                    obs = base.observation_manager.compute_group('actor')
                    poses.append(robot.data.site_pose_w[:, command.site_id].clone())
                    joints.append(robot.data.joint_pos.clone())
                    observations.append(obs[:args.paired_episodes].clone())
                    if step != args.steps:
                        with torch.inference_mode():
                            action = policy(TensorDict({'actor': obs}, batch_size=[args.episodes]))
                        env.step(action)
                poses = torch.stack(poses).cpu().numpy().astype(float)
                joints = torch.stack(joints).cpu().numpy().astype(float)
                observations = torch.stack(observations).cpu().numpy()
                if not np.isfinite(poses).all() or not np.isfinite(joints).all():
                    raise ValueError('Nonfinite GPU trajectory')
                np.testing.assert_allclose(joints[0], initial, atol=2e-7, rtol=0)
                np.testing.assert_allclose(command.target_pos.cpu().numpy(), targets, atol=1e-7, rtol=0)
                dt = cfg.decimation * model.opt.timestep
                rows, paired = [], []
                for episode in range(args.episodes):
                    metric = trajectory_metrics(poses[:, episode], joints[:, episode], targets[episode],
                                                quat, args.hold_samples, dt)
                    metric.update(episode=episode, initial_joint_position_rad=initial[episode].tolist(),
                                  initial_joint_velocity_rad_s=[0.0] * len(default),
                                  target_position_m=targets[episode].tolist())
                    metric['trajectory_violation_samples'] = trajectory_validity(
                        model, qadr, joint_ids, joints[:, episode])
                    rows.append(metric)
                    if episode < args.paired_episodes:
                        cp, cq, co = cpu_rollout(model, qadr, vadr, site, initial[episode], default,
                                                 targets[episode], quat, policy, rl_cfg,
                                                 args.steps, cfg.decimation, device)
                        cm = trajectory_metrics(cp, cq, targets[episode], quat, args.hold_samples, dt)
                        delta = float(np.abs(co[0] - observations[0, episode]).max())
                        if delta > 1e-4:
                            raise ValueError(f'CPU/GPU initial observation mismatch: {delta}')
                        paired.append({'episode': episode, 'cpu': cm,
                                       'max_initial_observation_delta': delta,
                                       'max_observation_delta': float(np.abs(co - observations[:, episode]).max()),
                                       'max_joint_delta_rad': float(np.abs(cq - joints[:, episode]).max()),
                                       'max_tcp_position_delta_m': float(np.linalg.norm(cp[:, :3] - poses[:, episode, :3], axis=1).max())})
                level = {'amplitude_rad': amplitude, 'sampling': sampling,
                         'summary': summarize(rows), 'results': rows, 'paired': paired}
                level['summary']['trajectory_violation_episode_count'] = sum(
                    any(r['trajectory_violation_samples'].values()) for r in rows)
                level['summary']['tail_success_without_sampled_violation_count'] = sum(
                    r['tail_success'] and not any(r['trajectory_violation_samples'].values()) for r in rows)
                levels.append(level)
                print(json.dumps({'checkpoint': str(checkpoint), 'amplitude': amplitude,
                                  'summary': level['summary'], 'rejected': sampling['rejected']}), flush=True)
            repo = Path(__file__).resolve().parents[5]
            def git(*parts):
                return subprocess.check_output(['git', '-C', str(repo), *parts], text=True).strip()
            report = {
                'protocol': 'task2-random-start-diagnostic-v1',
                'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint),
                'policy_task': contract.task_id, 'policy_contract': contract.to_dict(),
                'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
                'evaluator_sha256': file_hash(Path(__file__)), 'command': sys.orig_argv,
                'conditions': {'target_seed': args.target_seed, 'initial_seed': args.initial_seed,
                    'episodes': args.episodes, 'steps': args.steps, 'control_dt_s': dt,
                    'hold_samples': args.hold_samples, 'hold_span_s': (args.hold_samples - 1) * dt,
                    'success_position_m': .01, 'success_orientation_rad': float(np.deg2rad(3)),
                    'target_radius_uniform_m': [.02, .06], 'target_center_m': anchor.tolist(),
                    'target_quaternion_wxyz': quat.tolist(), 'default_joint_position_rad': default.tolist(),
                    'reset': 'independent uniform six-joint offsets; zero velocity and previous action',
                    'joint_limit_margin_rad': .01, 'joint_limits_rad': model.jnt_range[joint_ids[:6]].tolist(),
                    'collision_check': 'CPU mj_forward, reject enabled contact dist<=0; compiled exclusions retained',
                    'collision_exclusions': model.exclude_signature.tolist(),
                    'trajectory_collision_check': 'CPU replay of all 50 Hz joint samples; physics substeps not checked',
                    'automatic_termination': False, 'target_refresh': False,
                    'deterministic_policy': True, 'clip_actions': rl_cfg.clip_actions,
                    'backend': 'gpu_warp', 'paired_episodes_per_level': args.paired_episodes},
                'scope': 'Diagnostic validation; not a new trained task, independent holdout, or hardware acceptance',
                'levels': levels,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(f'Report: {output}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
