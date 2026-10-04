"""复用命令指标判断成功，避免奖励与终止分别维护阈值。"""
from __future__ import annotations
import torch
from mjlab.envs import ManagerBasedRlEnv
from .commands import ReachCommand
def success(env: ManagerBasedRlEnv) -> torch.Tensor:
    command = env.command_manager.get_term("reach")
    assert isinstance(command, ReachCommand)
    return command.metrics["success"] > 0.5
