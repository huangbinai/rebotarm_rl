"""Fixed-target robustness diagnostic for existing fixed-penalty Reach weights.

This changes evaluation initial conditions, not the checkpoint's task contract.
Targets stay anchored at default FK. CPU rejection sampling checks the compiled
model's joint limits and enabled collision pairs before any GPU rollout.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
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
from rebotarm_rl.contracts.artifacts import file_hash, evaluation_provenance, EvaluationReport
from rebotarm_rl.contracts.evaluation import RANDOM_START, validate_frozen_selection, validate_settings
from rebotarm_rl.contracts.policy import REACH_GRAVITY_FIXED
from .rollout import pose
from .fixed_target import evaluate_batch
from .state_checks import state_rejection, sample_initial_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--amplitudes', type=float, nargs='+', default=[0, .025, .05, .1, .2, .4])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=RANDOM_START.steps)
    parser.add_argument('--hold-samples', type=int, default=RANDOM_START.hold_samples)
    parser.add_argument('--target-seed', type=int, default=RANDOM_START.target_seed)
    parser.add_argument('--initial-seed', type=int, default=RANDOM_START.initial_seed)
    parser.add_argument('--paired-episodes', type=int, default=0)
    parser.add_argument('--output-name', default='task2_random_start_v1')
    parser.add_argument('--output-dir', type=Path, help='Only for temporary verification; default is each run/eval')
    parser.add_argument('--selection-file', type=Path,
                        help='Frozen candidate and exact fresh test settings; never reselect from test results')
    args = parser.parse_args()
    settings = {name: getattr(args, name) for name in
                ('episodes', 'steps', 'hold_samples', 'target_seed', 'initial_seed')}
    settings['amplitudes_rad'] = args.amplitudes
    try:
        validate_settings(settings)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.paired_episodes <= args.episodes:
        parser.error('Invalid paired sample count')
    if Path(args.output_name).name != args.output_name:
        parser.error('output-name must be a filename stem')
    selection = (validate_frozen_selection(args.selection_file, args.checkpoints, settings)
                 if args.selection_file is not None else None)
    provenance = evaluation_provenance(Path(__file__).resolve().parents[3], Path(__file__))
    with ExitStack() as stack:
        reports = []
        for checkpoint in args.checkpoints:
            checkpoint = checkpoint.resolve(strict=True)
            output_dir = args.output_dir or checkpoint.parent / 'eval'
            output = output_dir / f'{args.output_name}_{checkpoint.stem}_seed{args.target_seed}.json'
            record = {
                **provenance, 'protocol': RANDOM_START.name,
                'role': 'frozen_candidate_test' if selection else 'diagnostic_validation',
                'frozen_selection': selection, 'requested_settings': settings,
                'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint),
                'policy_task': REACH_GRAVITY_FIXED.task_id, 'policy_contract': REACH_GRAVITY_FIXED.to_dict(),
                'levels': [],
            }
            reports.append(stack.enter_context(EvaluationReport(output, record)))
        run_evaluation(args, reports)


def run_evaluation(args: argparse.Namespace, reports: list[EvaluationReport]) -> None:
    """Own the simulator lifecycle; reports and provenance exist before startup."""
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
            targets.append(anchor + direction / np.linalg.norm(direction) * rng.uniform(*RANDOM_START.target_radius_m))
        targets = np.asarray(targets)
        states = {a: sample_initial_states(model, qadr, joint_ids, default, a,
                                           args.episodes, args.initial_seed) for a in args.amplitudes}
        dt = cfg.decimation * model.opt.timestep
        runner = RecordedRunner(env, asdict(rl_cfg), device=device)
        for artifact in reports:
            checkpoint = Path(artifact.record['checkpoint'])
            runner.load(str(checkpoint), load_cfg={'actor': True}, strict=True, map_location=device)
            policy = runner.get_inference_policy(device=device)
            levels = artifact.record['levels']
            for amplitude in args.amplitudes:
                initial, sampling = states[amplitude]
                level = {'amplitude_rad': amplitude, 'sampling': sampling,
                         **evaluate_batch(env, policy, initial, targets, quat, rl_cfg,
                                          steps=args.steps, hold_samples=args.hold_samples,
                                          paired_episodes=args.paired_episodes)}
                levels.append(level)
                print(json.dumps({'checkpoint': str(checkpoint), 'amplitude': amplitude,
                                  'summary': level['summary'], 'rejected': sampling['rejected']}), flush=True)
                artifact.save_progress()
            artifact.record.update({
                'conditions': {'target_seed': args.target_seed, 'initial_seed': args.initial_seed,
                    'episodes': args.episodes, 'steps': args.steps, 'control_dt_s': dt,
                    'hold_samples': args.hold_samples, 'hold_span_s': (args.hold_samples - 1) * dt,
                    'success_position_m': RANDOM_START.success_position_m, 'success_orientation_rad': RANDOM_START.success_orientation_rad,
                    'target_radius_uniform_m': list(RANDOM_START.target_radius_m), 'target_center_m': anchor.tolist(),
                    'target_quaternion_wxyz': quat.tolist(), 'default_joint_position_rad': default.tolist(),
                    'reset': 'independent uniform six-joint offsets; zero velocity and previous action',
                    'joint_limit_margin_rad': RANDOM_START.joint_limit_margin_rad, 'joint_limits_rad': model.jnt_range[joint_ids[:6]].tolist(),
                    'collision_check': 'CPU mj_forward, reject enabled contact dist<=0; compiled exclusions retained',
                    'collision_exclusions': model.exclude_signature.tolist(),
                    'trajectory_collision_check': 'CPU replay of all 50 Hz joint samples; physics substeps not checked',
                    'automatic_termination': False, 'target_refresh': False,
                    'deterministic_policy': True, 'clip_actions': rl_cfg.clip_actions,
                    'backend': 'gpu_warp', 'paired_episodes_per_level': args.paired_episodes},
                'scope': 'Offline robustness evaluation; not a new trained task or hardware acceptance. Test role requires frozen_selection.',
            })
            artifact.complete()
            print(f'Report: {artifact.output}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
