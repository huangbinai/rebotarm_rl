"""Static pose feasibility for target-range diagnostics; not a motion planner."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .state_checks import state_rejection


@dataclass(frozen=True)
class IkSettings:
    position_tolerance_m: float = 1e-4
    orientation_tolerance_rad: float = np.deg2rad(.1)
    joint_margin_rad: float = .01
    orientation_weight_m: float = .2
    damping: float = 1e-6
    max_joint_step_rad: float = .15
    iterations: int = 150
    restarts: int = 3


IK = IkSettings()


def solve_pose(model, qadr, vadr, joint_ids, site: int, default: np.ndarray,
               target: np.ndarray, target_quat: np.ndarray, rng: np.random.Generator,
               settings: IkSettings = IK) -> tuple[dict | None, str | None]:
    """Damped Jacobian IK with bounded steps and independently checked endpoints.

    A returned witness proves a feasible static pose to the stated tolerances.
    Failure means this solver did not establish feasibility, not unreachability.
    """
    data = mujoco.MjData(model)
    limits = model.jnt_range[joint_ids[:6]] + [settings.joint_margin_rad, -settings.joint_margin_rad]
    jp = np.zeros((3, model.nv))
    jr = np.zeros_like(jp)
    quat, inverse, relative, rotation = np.empty(4), np.empty(4), np.empty(4), np.empty(3)
    rejection = 'ik_unresolved'
    for restart in range(settings.restarts):
        q = default.copy()
        if restart:
            q[:6] = np.clip(q[:6] + rng.uniform(-.6 * restart, .6 * restart, 6), limits[:, 0], limits[:, 1])
        for iteration in range(settings.iterations):
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            mujoco.mju_mat2Quat(quat, data.site_xmat[site])
            mujoco.mju_negQuat(inverse, quat)
            mujoco.mju_mulQuat(relative, target_quat, inverse)
            if relative[0] < 0:
                relative *= -1
            mujoco.mju_quat2Vel(rotation, relative, 1.)
            position = target - data.site_xpos[site]
            pe, oe = float(np.linalg.norm(position)), float(np.linalg.norm(rotation))
            if pe < settings.position_tolerance_m and oe < settings.orientation_tolerance_rad:
                reason = state_rejection(model, data, qadr, joint_ids, q, settings.joint_margin_rad)
                if reason is None:
                    return {'joint_position_rad': q.tolist(), 'position_error_m': pe,
                            'orientation_error_rad': oe, 'restart': restart, 'iterations': iteration + 1}, None
                rejection = reason
                break
            mujoco.mj_jacSite(model, data, jp, jr, site)
            jacobian = np.concatenate((jp[:, vadr[:6]], settings.orientation_weight_m * jr[:, vadr[:6]]))
            error = np.r_[position, settings.orientation_weight_m * rotation]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + settings.damping * np.eye(6), error)
            if not np.isfinite(delta).all():
                raise ValueError('Nonfinite IK update')
            delta *= min(1., settings.max_joint_step_rad / max(np.max(np.abs(delta)), 1e-12))
            q[:6] = np.clip(q[:6] + delta, limits[:, 0], limits[:, 1])
    return None, rejection


def sample_reachable_targets(model, qadr, vadr, joint_ids, site: int, default: np.ndarray,
                             anchor: np.ndarray, quat: np.ndarray, radius: tuple[float, float],
                             count: int, seed: int) -> tuple[np.ndarray, dict]:
    """Radially uniform proposals conditioned on a valid static IK witness.

    Per-episode target streams and separate solver streams keep solver retries
    from shifting later goals; each rejected proposal remains visible in counts.
    """
    targets, witnesses, attempts = [], [], []
    rejected = {'ik_unresolved': 0, 'joint_limit': 0, 'self_collision': 0, 'nonfinite': 0}
    for episode in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, episode]))
        for attempt in range(1, 1001):
            direction = rng.normal(size=3)
            target = anchor + direction / np.linalg.norm(direction) * rng.uniform(*radius)
            solver_rng = np.random.default_rng(np.random.SeedSequence([seed, episode, attempt, 937]))
            witness, reason = solve_pose(model, qadr, vadr, joint_ids, site, default, target, quat, solver_rng)
            if witness is not None:
                targets.append(target)
                witnesses.append(witness)
                attempts.append(attempt)
                break
            rejected[reason] += 1
        else:
            raise RuntimeError(f'No IK-certified target for episode {episode}; rejected={rejected}')
    return np.asarray(targets), {'ik_settings': asdict(IK), 'rejected': rejected,
                                'attempts_per_episode': attempts, 'witnesses': witnesses,
                                'scope': 'Static pose feasibility only; no collision-free path guarantee'}
