from __future__ import annotations
from rebotarm_rl.contracts.policy import REACH_V1
import torch
from mjlab.envs import ManagerBasedRlEnv
from dataclasses import dataclass
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_error_magnitude
@dataclass(kw_only=True)
class ReachCommandCfg(CommandTermCfg):
    """在初始TCP位姿附近生成固定或随机位置目标，姿态保持初始值。"""

    position_radius: float = 0.06
    orientation_radius: float = 0.0
    asset_name: str = "robot"
    site_name: str = "ee_site"

    def build(self, env: ManagerBasedRlEnv) -> ReachCommand:
        return ReachCommand(self, env)


class ReachCommand(CommandTerm):
    cfg: ReachCommandCfg

    def __init__(self, cfg: ReachCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.asset_name]
        site_ids, _ = self.robot.find_sites((cfg.site_name,), preserve_order=True)
        if len(site_ids) != 1:
            raise ValueError(f"Expected one site named {cfg.site_name!r}, got {site_ids}")
        self.site_id = site_ids[0]
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.target_pos, self.target_quat), dim=-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        home_pose = self.robot.data.site_pose_w[:, self.site_id]
        self.target_pos[env_ids] = home_pose[env_ids, :3]
        if self.cfg.position_radius > 0.0:
            direction = torch.randn(len(env_ids), 3, device=self.device)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            radius = torch.rand(len(env_ids), 1, device=self.device) * self.cfg.position_radius
            self.target_pos[env_ids] += direction * radius
        self.target_quat[env_ids] = home_pose[env_ids, 3:7]

    def _update_metrics(self) -> None:
        current = self.robot.data.site_pose_w[:, self.site_id]
        position_error = torch.linalg.vector_norm(current[:, :3] - self.target_pos, dim=-1)
        orientation_error = quat_error_magnitude(current[:, 3:7], self.target_quat)
        success = (position_error < REACH_V1.success_position_m) & (orientation_error < REACH_V1.success_orientation_rad)
        self.metrics["position_error"] = position_error
        self.metrics["orientation_error"] = orientation_error
        self.metrics["success"] = success.float()

    def success_mask(self) -> torch.Tensor:
        """Compute success from the current pose, without relying on cached metrics."""
        current = self.robot.data.site_pose_w[:, self.site_id]
        position_error = torch.linalg.vector_norm(current[:, :3] - self.target_pos, dim=-1)
        orientation_error = quat_error_magnitude(current[:, 3:7], self.target_quat)
        return (position_error < REACH_V1.success_position_m) & (
            orientation_error < REACH_V1.success_orientation_rad
        )

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids
