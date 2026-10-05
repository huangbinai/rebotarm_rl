"""位置伺服 Reach V1 奖励；距离单位为m、角度为rad。"""
from __future__ import annotations
import torch
from mjlab.envs import ManagerBasedRlEnv
from .commands import ReachCommand
from mjlab.utils.lab_api.math import quat_error_magnitude
def _errors(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    command = env.command_manager.get_term("reach")
    assert isinstance(command, ReachCommand)
    current = env.scene["robot"].data.site_pose_w[:, command.site_id]
    pos = torch.linalg.vector_norm(current[:, :3] - command.target_pos, dim=-1)
    ori = quat_error_magnitude(current[:, 3:7], command.target_quat)
    return pos, ori


def position_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _errors(env)[0]


def position_fine(env: ManagerBasedRlEnv) -> torch.Tensor:
    pos, _ = _errors(env)
    return torch.tanh(pos / 0.10)


def orientation_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _errors(env)[1]


def action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.square(env.action_manager.action - env.action_manager.prev_action).sum(dim=-1)


def joint_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.square(robot.data.joint_vel[:, :6]).sum(dim=-1)
