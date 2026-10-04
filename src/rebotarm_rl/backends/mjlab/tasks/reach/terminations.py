from __future__ import annotations
import torch
from mjlab.envs import ManagerBasedRlEnv
from .commands import ReachCommand
def success(env: ManagerBasedRlEnv) -> torch.Tensor:
    command = env.command_manager.get_term("reach")
    assert isinstance(command, ReachCommand)
    return command.metrics["success"] > 0.5
