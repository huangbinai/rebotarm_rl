"""Reach环境配置：标准世界系姿态误差与关节力矩控制。"""
from rebotarm_rl.contracts.policy import REACH_V1
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, time_out
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from rebotarm_rl.backends.mjlab.robots.rebotarm import ROBOT_CFG, ROBOT
from .commands import ReachCommandCfg
from .observations import position_error, orientation_error
from .rewards import reach_reward
from .terminations import success

def make_env_cfg(*, play: bool = False, num_envs: int = 256) -> ManagerBasedRlEnvCfg:
    observations = {
        "actor": ObservationGroupCfg(
            terms={
                "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": ROBOT}),
                "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": ROBOT}),
                "position_error": ObservationTermCfg(func=position_error),
                "orientation_error": ObservationTermCfg(func=orientation_error),
            },
            enable_corruption=False,
        ),
    }
    actions: dict[str, ActionTermCfg] = {
        "joint_effort": JointEffortActionCfg(
            entity_name="robot",
            actuator_names=("joint[1-6]",),
            scale=1.0,
        ),
    }
    rewards = {"reach": RewardTermCfg(func=reach_reward, weight=1.0)}
    terminations = {
        "success": TerminationTermCfg(func=success, time_out=False),
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(entities={"robot": ROBOT_CFG}, num_envs=num_envs, env_spacing=1.5),
        observations=observations,
        actions=actions,
        commands={"reach": ReachCommandCfg(resampling_time_range=(1.0e9, 1.0e9))},
        rewards=rewards,
        terminations=terminations,
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=REACH_V1.physics_dt_s)),
        decimation=REACH_V1.decimation,
        episode_length_s=5.0 if not play else 1.0e9,
        seed=7,
    )
