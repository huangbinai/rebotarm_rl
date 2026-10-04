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
    command = env.command_manager.get_term("reach")
    assert isinstance(command, ReachCommand)
    ee = env.scene["robot"].data.site_quat_w[:, command.site_id]
    # 保留V1历史编码以兼容旧权重；这不是标准wxyz轴角转换。
    # V2修正单独注册，详见docs/policy_contract.md。
    current = ee / ee.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    target = command.target_quat / command.target_quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    xyz = current[:, :3] * target[:, 3:4] - target[:, :3] * current[:, 3:4]
    return 2.0 * xyz


def orientation_error_v2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """读取末端世界系姿态，输出current相对target的最短旋转向量。"""
    from rebotarm_rl.backends.mjlab.rotation import rotation_error_wxyz
    command = env.command_manager.get_term("reach")
    assert isinstance(command, ReachCommand)
    current = env.scene["robot"].data.site_quat_w[:, command.site_id]
    return rotation_error_wxyz(current, command.target_quat)
