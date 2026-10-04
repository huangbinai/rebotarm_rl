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
    # Legacy v1 feature encoding; not a standard wxyz axis-angle conversion.
    # Preserve checkpoint semantics; see docs/policy_contract.md.
    current = ee / ee.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    target = command.target_quat / command.target_quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    xyz = current[:, :3] * target[:, 3:4] - target[:, :3] * current[:, 3:4]
    return 2.0 * xyz
