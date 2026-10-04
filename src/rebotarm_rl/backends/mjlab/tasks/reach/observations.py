"""策略观测：六关节相对状态及世界系末端误差。"""
from __future__ import annotations
import torch
from mjlab.envs import ManagerBasedRlEnv
from .commands import ReachCommand
def position_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    command = env.command_manager.get_term("reach")
    assert isinstance(command, ReachCommand)
    ee = env.scene["robot"].data.site_pos_w[:, command.site_id]
    return ee - command.target_pos


def orientation_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    """读取末端世界系姿态，输出current相对target的最短旋转向量。"""
    from rebotarm_rl.backends.mjlab.rotation import rotation_error_wxyz
    command = env.command_manager.get_term("reach")
    assert isinstance(command, ReachCommand)
    current = env.scene["robot"].data.site_quat_w[:, command.site_id]
    return rotation_error_wxyz(current, command.target_quat)
