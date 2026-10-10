"""CPU control execution and fixed-target batched GPU rollouts for Reach evaluation."""
from __future__ import annotations

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from rebotarm_rl.backends.mjlab.rotation import rotation_error_wxyz
from rebotarm_rl.contracts.policy import REACH_GRAVITY_FIXED, PolicyContract


def pose(model, data, site: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward CPU state and return world TCP position and wxyz quaternion."""
    mujoco.mj_forward(model, data)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site])
    return data.site_xpos[site].copy(), quat


class CpuRollout:
    """Own CPU solver history and previous action; share the compiled GPU model.

    Preserve the caller's default-position dtype: legacy evaluation uses float32,
    while the random-start diagnostic uses float64 for its CPU reference.
    """

    def __init__(self, model, qadr, vadr, site: int, default: np.ndarray,
                 contract: PolicyContract, decimation: int, clip_actions: float | None):
        self.model, self.qadr, self.vadr, self.site = model, qadr, vadr, site
        self.default, self.contract = default, contract
        self.decimation, self.clip_actions = decimation, clip_actions
        self.data = mujoco.MjData(model)
        self.actuator_ids = [model.actuator(f'robot/joint{i}_torque').id for i in range(1, 7)]
        self.last_action = np.zeros(6, dtype=np.float32)

    def reset(self, initial: np.ndarray, velocity: np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = initial
        if velocity is not None:
            self.data.qvel[self.vadr] = velocity
        self.last_action.fill(0)

    def observe(self, target: np.ndarray, quat: np.ndarray,
                tcp: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """24-D actor observation with current-target position error."""
        p, q = pose(self.model, self.data, self.site) if tcp is None else tcp
        return np.concatenate((self.data.qpos[self.qadr[:6]] - self.default[:6],
                               self.data.qvel[self.vadr[:6]], p - target,
                               rotation_error_wxyz(torch.from_numpy(q), torch.from_numpy(quat)).numpy(),
                               self.last_action))

    def step(self, action: torch.Tensor) -> None:
        """Clip, sample the position target once, and hold through all substeps."""
        if self.clip_actions is not None:
            action = action.clamp(-self.clip_actions, self.clip_actions)
        processed = action[0] * self.contract.action_scale
        if self.contract.action_type == 'joint_position_delta':
            processed = processed.clamp(-self.contract.action_scale, self.contract.action_scale)
            reference = self.data.qpos[self.qadr[:6]]
        else:
            reference = self.default[:6]
        self.data.ctrl[:] = 0
        self.data.ctrl[self.actuator_ids] = (torch.as_tensor(reference, device=action.device) + processed).cpu().numpy()
        self.last_action = action[0].cpu().numpy()
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)


def cpu_rollout(model, qadr, vadr, site, initial, default, target, quat,
                policy, rl_cfg, steps: int, decimation: int, device: str):
    """Task 2 trajectory at t=0 and after every control step."""
    cpu = CpuRollout(model, qadr, vadr, site, default, REACH_GRAVITY_FIXED, decimation, rl_cfg.clip_actions)
    cpu.reset(initial)
    poses, joints, observations = [], [], []
    for step in range(steps + 1):
        p, q = pose(model, cpu.data, site)
        obs = cpu.observe(target, quat, (p, q))
        poses.append(np.r_[p, q])
        joints.append(cpu.data.qpos[qadr].copy())
        observations.append(obs)
        if step == steps:
            break
        with torch.inference_mode():
            action = policy(TensorDict({'actor': torch.as_tensor(obs[None], dtype=torch.float32,
                                                                 device=device)}, batch_size=[1]))
        cpu.step(action)
    return np.asarray(poses), np.asarray(joints), np.asarray(observations)


def gpu_rollout(env, policy, initial: np.ndarray, targets: np.ndarray, quat: np.ndarray,
                steps: int, paired_episodes: int, device: str):
    """Reset all batch members, inject valid starts, and keep one goal per episode."""
    base = env.unwrapped
    robot = base.scene['robot']
    command = base.command_manager.get_term('reach')
    env.reset()
    q = torch.as_tensor(initial, device=device, dtype=torch.float32)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    base.sim.forward()
    base.sim.sense()
    command.target_pos[:] = torch.as_tensor(targets, device=device, dtype=torch.float32)
    command.target_quat[:] = torch.as_tensor(quat, device=device, dtype=torch.float32)
    expected_target = command.target_pos.clone()
    expected_quat = command.target_quat.clone()
    if torch.count_nonzero(robot.data.joint_vel) or torch.count_nonzero(base.action_manager.action):
        raise ValueError('Reset must clear joint velocity and previous action')
    poses, joints, observations = [], [], []
    for step in range(steps + 1):
        if not torch.equal(command.target_pos, expected_target) or not torch.equal(command.target_quat, expected_quat):
            raise ValueError('Fixed evaluation target changed during rollout')
        obs = base.observation_manager.compute_group('actor')
        poses.append(robot.data.site_pose_w[:, command.site_id].clone())
        joints.append(robot.data.joint_pos.clone())
        observations.append(obs[:paired_episodes].clone())
        if step != steps:
            with torch.inference_mode():
                action = policy(TensorDict({'actor': obs}, batch_size=[len(initial)]))
            env.step(action)
    poses = torch.stack(poses).cpu().numpy().astype(float)
    joints = torch.stack(joints).cpu().numpy().astype(float)
    observations = torch.stack(observations).cpu().numpy()
    if not np.isfinite(poses).all() or not np.isfinite(joints).all():
        raise ValueError('Nonfinite GPU trajectory')
    np.testing.assert_allclose(joints[0], initial, atol=2e-7, rtol=0)
    return poses, joints, observations
