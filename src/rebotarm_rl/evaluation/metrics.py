"""Engine-independent Reach metrics; keep legacy and Task 2 thresholds distinct."""
from __future__ import annotations

import numpy as np
from rebotarm_rl.contracts.evaluation import RANDOM_START


def errors(pos: np.ndarray, quat: np.ndarray, target: np.ndarray, target_quat: np.ndarray) -> tuple[float, float]:
    """Legacy paired-evaluator pose errors; preserve its quaternion arithmetic."""
    cosine = np.clip(abs(np.dot(quat, target_quat)), 0.0, 1.0)
    return float(np.linalg.norm(pos - target)), float(2 * np.arccos(cosine))


def success_summary(flags: list[bool], hold_steps: int) -> dict[str, object]:
    """Summarize first-hit, continuous-hold, and end-of-rollout success."""
    if not flags:
        raise ValueError("success trajectory must not be empty")
    longest = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    tail_success = len(flags) >= hold_steps and all(flags[-hold_steps:])
    hold_success = longest >= hold_steps
    return {
        "initial_success": bool(flags[0]),
        "final_success": bool(flags[-1]),
        "first_success_step": next((i for i, flag in enumerate(flags) if flag), None),
        "longest_success_hold_steps": longest,
        "hold_success": hold_success,
        "tail_success": tail_success,
    }


def trajectory_metrics(poses: np.ndarray, joints: np.ndarray, target: np.ndarray,
                       target_quat: np.ndarray, hold_samples: int, dt: float) -> dict:
    """Metrics include t=0; a 26-sample hold spans 0.50 s at 50 Hz."""
    pe = np.linalg.norm(poses[:, :3] - target, axis=-1)
    quats = poses[:, 3:] / np.linalg.norm(poses[:, 3:], axis=-1, keepdims=True)
    tq = target_quat / np.linalg.norm(target_quat)
    oe = 2 * np.arccos(np.clip(np.abs(quats @ tq), 0, 1))
    flags = (pe < RANDOM_START.success_position_m) & (oe < RANDOM_START.success_orientation_rad)
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


def summarize(rows: list[dict]) -> dict:
    """Aggregate fixed-target episodes, retaining successes-only time denominators."""
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
