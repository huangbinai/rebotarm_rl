"""距离单位为m、角度为rad；奖励鼓励接近目标并惩罚过大力矩。"""
from __future__ import annotations
import torch
from mjlab.envs import ManagerBasedRlEnv
from .commands import ReachCommand
from mjlab.utils.lab_api.math import quat_error_magnitude
def reach_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    command = env.command_manager.get_term("reach")
    assert isinstance(command, ReachCommand)
    current = env.scene["robot"].data.site_pose_w[:, command.site_id]
    pos = torch.linalg.vector_norm(current[:, :3] - command.target_pos, dim=-1)
    ori = quat_error_magnitude(current[:, 3:7], command.target_quat)
    effort = torch.square(env.action_manager.action).sum(dim=-1)
    return torch.exp(-pos / 0.03) + 0.5 * torch.exp(-ori / 0.20) - 0.001 * effort
