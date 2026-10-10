"""Shared fixed-target batch evaluation, including CPU replay and paired diagnostics."""
from __future__ import annotations

import numpy as np

from rebotarm_rl.evaluation.metrics import trajectory_metrics, summarize
from .rollout import cpu_rollout, gpu_rollout
from .state_checks import trajectory_validity


def evaluate_batch(env, policy, initial: np.ndarray, targets: np.ndarray, quat: np.ndarray,
                   rl_cfg, *, steps: int, hold_samples: int, paired_episodes: int) -> dict:
    """Evaluate one fixed-goal batch; thresholds and t=0 sampling match Task 2."""
    base = env.unwrapped
    device = base.device
    robot = base.scene['robot']
    command = base.command_manager.get_term('reach')
    model = base.sim.mj_model
    qadr = robot.indexing.joint_q_adr.cpu().numpy()
    vadr = robot.indexing.joint_v_adr.cpu().numpy()
    joint_ids = np.array([model.joint(f'robot/{name}').id for name in robot.joint_names])
    default = robot.data.default_joint_pos[0].cpu().numpy().astype(float)
    site = int(robot.indexing.site_ids[command.site_id])
    poses, joints, observations = gpu_rollout(
        env, policy, initial, targets, quat, steps, paired_episodes, device)
    dt = base.cfg.decimation * model.opt.timestep
    rows, paired = [], []
    for episode in range(len(initial)):
        metric = trajectory_metrics(poses[:, episode], joints[:, episode], targets[episode],
                                    quat, hold_samples, dt)
        metric.update(episode=episode, initial_joint_position_rad=initial[episode].tolist(),
                      initial_joint_velocity_rad_s=[0.0] * len(default),
                      target_position_m=targets[episode].tolist())
        metric['trajectory_violation_samples'] = trajectory_validity(
            model, qadr, joint_ids, joints[:, episode])
        rows.append(metric)
        if episode < paired_episodes:
            cp, cq, co = cpu_rollout(model, qadr, vadr, site, initial[episode], default,
                                     targets[episode], quat, policy, rl_cfg,
                                     steps, base.cfg.decimation, device)
            cm = trajectory_metrics(cp, cq, targets[episode], quat, hold_samples, dt)
            delta = float(np.abs(co[0] - observations[0, episode]).max())
            if delta > 1e-4:
                raise ValueError(f'CPU/GPU initial observation mismatch: {delta}')
            paired.append({'episode': episode, 'cpu': cm,
                           'max_initial_observation_delta': delta,
                           'max_observation_delta': float(np.abs(co - observations[:, episode]).max()),
                           'max_joint_delta_rad': float(np.abs(cq - joints[:, episode]).max()),
                           'max_tcp_position_delta_m': float(np.linalg.norm(cp[:, :3] - poses[:, episode, :3], axis=1).max())})
    summary = summarize(rows)
    summary['trajectory_violation_episode_count'] = sum(
        any(r['trajectory_violation_samples'].values()) for r in rows)
    summary['tail_success_without_sampled_violation_count'] = sum(
        r['tail_success'] and not any(r['trajectory_violation_samples'].values()) for r in rows)
    return {'summary': summary, 'results': rows, 'paired': paired}
