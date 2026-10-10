"""Task 3 diagnostic: expand IK-checked target range at a fixed random-start amplitude."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from rebotarm_rl.backends.mjlab import registration  # noqa: F401
from rebotarm_rl.backends.mjlab.runner import RecordedRunner
from rebotarm_rl.contracts.artifacts import EvaluationReport, evaluation_provenance, file_hash
from rebotarm_rl.contracts.evaluation import RANDOM_START, TARGET_RANGE
from rebotarm_rl.contracts.policy import REACH_GRAVITY_FIXED
from .fixed_target import evaluate_batch
from .reachability import sample_reachable_targets
from .rollout import pose
from .state_checks import sample_initial_states, state_rejection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--paired-episodes', type=int, default=3)
    parser.add_argument('--target-seed', type=int, default=TARGET_RANGE.target_seed)
    parser.add_argument('--initial-seed', type=int, default=TARGET_RANGE.initial_seed)
    args = parser.parse_args()
    if (args.episodes < 1 or not 0 <= args.paired_episodes <= args.episodes
            or min(args.target_seed, args.initial_seed) < 0):
        parser.error('Invalid episode count or seed')
    checkpoint = args.checkpoint.resolve(strict=True)
    provenance = evaluation_provenance(Path(__file__).resolve().parents[3], Path(__file__))
    record = {**provenance, 'protocol': TARGET_RANGE.name, 'role': 'diagnostic_validation',
              'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint),
              'policy_task': REACH_GRAVITY_FIXED.task_id, 'policy_contract': REACH_GRAVITY_FIXED.to_dict(),
              'requested_settings': vars(args) | {'checkpoint': str(checkpoint), 'output': str(args.output)},
              'decision_before_validation': {'minimum_tail_success_rate': TARGET_RANGE.minimum_tail_success_rate,
                                            'maximum_sampled_violation_episodes': 0,
                                            'use': 'Choose range or decide whether training is needed; not final acceptance'},
              'reserved_test_seeds': {'target_seed': TARGET_RANGE.reserved_test_target_seed, 'initial_seed': TARGET_RANGE.reserved_test_initial_seed}, 'levels': []}
    if args.target_seed == TARGET_RANGE.reserved_test_target_seed or args.initial_seed == TARGET_RANGE.reserved_test_initial_seed:
        parser.error('Reserved final-test seeds cannot be used for diagnostic selection')
    with EvaluationReport(args.output, record) as report:
        run_evaluation(args, report)


def run_evaluation(args: argparse.Namespace, report: EvaluationReport) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    cfg = load_env_cfg(REACH_GRAVITY_FIXED.task_id)
    cfg.scene.num_envs = args.episodes
    cfg.seed = args.target_seed
    cfg.terminations = {}
    cfg.commands['reach'].position_radius = 0.
    cfg.commands['reach'].resampling_time_range = (1e9, 1e9)
    rl_cfg = load_rl_cfg(REACH_GRAVITY_FIXED.task_id)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device='cuda:0'), clip_actions=rl_cfg.clip_actions)
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
        initial, sampling = sample_initial_states(model, qadr, joint_ids, default, TARGET_RANGE.initial_amplitude_rad, args.episodes, args.initial_seed)
        report.record['conditions'] = {
            'target_seed': args.target_seed, 'initial_seed': args.initial_seed,
            'initial_amplitude_rad': TARGET_RANGE.initial_amplitude_rad, 'initial_sampling': sampling,
            'episodes_per_band': args.episodes, 'steps': RANDOM_START.steps,
            'hold_samples': RANDOM_START.hold_samples, 'hold_span_s': (RANDOM_START.hold_samples - 1) * cfg.decimation * model.opt.timestep,
            'control_dt_s': cfg.decimation * model.opt.timestep,
            'success_position_m': RANDOM_START.success_position_m,
            'success_orientation_rad': RANDOM_START.success_orientation_rad,
            'target_center_m': anchor.tolist(), 'target_quaternion_wxyz': quat.tolist(),
            'target_radius_bands_m': TARGET_RANGE.bands_m, 'target_distribution': 'radially uniform proposals conditioned on a valid static IK witness',
            'default_joint_position_rad': default.tolist(), 'joint_limits_rad': model.jnt_range[joint_ids[:6]].tolist(),
            'joint_limit_margin_rad': RANDOM_START.joint_limit_margin_rad,
            'collision_exclusions': model.exclude_signature.tolist(),
            'trajectory_collision_check': 'CPU replay of 50 Hz joint samples; physics substeps not checked',
            'automatic_termination': False, 'target_refresh': False,
            'deterministic_policy': True, 'clip_actions': rl_cfg.clip_actions,
            'backend': 'gpu_warp', 'paired_episodes_per_band': args.paired_episodes,
        }
        report.save_progress()
        runner = RecordedRunner(env, asdict(rl_cfg), device='cuda:0')
        runner.load(str(args.checkpoint.resolve()), load_cfg={'actor': True}, strict=True, map_location='cuda:0')
        policy = runner.get_inference_policy(device='cuda:0')
        for radius in TARGET_RANGE.bands_m:
            targets, target_sampling = sample_reachable_targets(
                model, qadr, vadr, joint_ids, site, default, anchor, quat, radius, args.episodes, args.target_seed)
            level = {'target_radius_uniform_m': radius, 'target_sampling': target_sampling,
                     **evaluate_batch(env, policy, initial, targets, quat, rl_cfg, steps=RANDOM_START.steps,
                                      hold_samples=RANDOM_START.hold_samples, paired_episodes=args.paired_episodes)}
            report.record['levels'].append(level)
            report.save_progress()
            print(json.dumps({'radius_m': radius, 'summary': level['summary'],
                              'target_rejected': target_sampling['rejected']}), flush=True)
        report.complete()
        print(f'Report: {report.output}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
