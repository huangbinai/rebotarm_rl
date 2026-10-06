"""Franka Reach recipe adapted to reBotArm; not a PhysX/Franka reproduction."""
import mujoco
import torch
from dataclasses import dataclass
from mjlab.envs.mdp.actions.actions import JointPositionActionCfg
from mjlab.envs.mdp.curriculums import reward_curriculum
from mjlab.managers import CurriculumTermCfg
from rebotarm_rl.contracts.policy import REACH_ALIGNED
from .commands import ReachCommand, ReachCommandCfg
from .config import make_env_cfg


@dataclass(kw_only=True)
class AnchoredReachCommandCfg(ReachCommandCfg):
    def build(self, env):
        return AnchoredReachCommand(self, env)


class AnchoredReachCommand(ReachCommand):
    """Sample around fixed default FK, not stale reset caches or a moving TCP."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        ids = self.robot.indexing.joint_q_adr.cpu().numpy()
        data.qpos[ids] = self.robot.data.default_joint_pos[0].cpu().numpy()
        mujoco.mj_forward(model, data)
        site = int(self.robot.indexing.site_ids[self.site_id])
        quat = torch.empty(4, dtype=torch.float64).numpy()
        mujoco.mju_mat2Quat(quat, data.site_xmat[site])
        self.anchor_pos = torch.as_tensor(data.site_xpos[site].copy(), device=self.device,
                                          dtype=torch.float32) + env.scene.env_origins
        self.anchor_quat = torch.as_tensor(quat, device=self.device, dtype=torch.float32)

    def _resample_command(self, env_ids):
        direction = torch.randn(len(env_ids), 3, device=self.device)
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        radius = torch.rand(len(env_ids), 1, device=self.device) * self.cfg.position_radius
        self.target_pos[env_ids] = self.anchor_pos[env_ids] + direction * radius
        self.target_quat[env_ids] = self.anchor_quat


def make_aligned_env_cfg(*, play=False, num_envs=128):
    cfg = make_env_cfg(play=play, num_envs=num_envs)
    cfg.actions['joint_position'] = JointPositionActionCfg(
        entity_name='robot', actuator_names=('joint[1-6]',),
        scale=REACH_ALIGNED.action_scale, use_default_offset=True,
    )
    cfg.commands['reach'] = AnchoredReachCommandCfg(resampling_time_range=(4., 4.))
    cfg.terminations.pop('success')
    cfg.episode_length_s = 12.
    cfg.curriculum = {
        name: CurriculumTermCfg(func=reward_curriculum, params={
            'reward_name': name, 'stages': [{'step': 4501, 'weight': weight}]})
        for name, weight in [('action_rate', -.005), ('joint_velocity', -.001)]
    }
    return cfg
