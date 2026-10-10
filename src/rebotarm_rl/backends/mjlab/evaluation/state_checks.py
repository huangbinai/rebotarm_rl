"""MuJoCo geometric checks and deterministic six-joint reset sampling."""
from __future__ import annotations

import mujoco
import numpy as np
from rebotarm_rl.contracts.evaluation import RANDOM_START


def state_rejection(model, data, qadr, joint_ids, q: np.ndarray, margin: float = RANDOM_START.joint_limit_margin_rad) -> str | None:
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


def trajectory_validity(model, qadr, joint_ids, joints: np.ndarray) -> dict[str, int]:
    """Count independent violations at recorded control samples, not substeps.

    Unlike sampling's first-rejection reason, simultaneous limit/contact failures
    contribute to both counters. Nonfinite states cannot be forwarded safely.
    """
    data = mujoco.MjData(model)
    violations = {'joint_limit': 0, 'self_collision': 0, 'nonfinite': 0}
    limits = model.jnt_range[joint_ids[:6]]
    if not model.jnt_limited[joint_ids[:6]].all():
        raise ValueError('Expected six limited arm joints')
    for q in joints:
        if not np.isfinite(q).all():
            violations['nonfinite'] += 1
            continue
        violations['joint_limit'] += int(np.any(q[:6] < limits[:, 0]) or np.any(q[:6] > limits[:, 1]))
        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        violations['self_collision'] += int(any(c.dist <= 0 for c in data.contact))
    return violations
