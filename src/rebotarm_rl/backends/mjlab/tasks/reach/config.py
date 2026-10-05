"""Reach V1 环境配置：位置目标与底层位置伺服控制。"""
from rebotarm_rl.contracts.policy import REACH_V1
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action, time_out
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from rebotarm_rl.backends.mjlab.robots.rebotarm import ROBOT_CFG, ROBOT
from .commands import ReachCommandCfg
from .observations import position_error, orientation_error
from .rewards import action_rate, joint_velocity, orientation_error as orientation_reward
from .rewards import position_error as position_reward, position_fine
from .terminations import success

def make_env_cfg(*, play: bool = False, num_envs: int = 256) -> ManagerBasedRlEnvCfg:
    observations = {
        "actor": ObservationGroupCfg(
            terms={
                "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": ROBOT}),
                "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": ROBOT}),
                "position_error": ObservationTermCfg(func=position_error),
                "orientation_error": ObservationTermCfg(func=orientation_error),
                "last_action": ObservationTermCfg(func=last_action),
            },
            enable_corruption=False,
        ),
    }
    actions: dict[str, ActionTermCfg] = {
        "joint_position": RelativeJointPositionActionCfg(
            entity_name="robot",
            actuator_names=("joint[1-6]",),
            scale=REACH_V1.action_scale,
            clip={"joint[1-6]": (-REACH_V1.action_scale, REACH_V1.action_scale)},
        ),
    }
    rewards = {
        "position": RewardTermCfg(func=position_reward, weight=-0.2),
        "position_fine": RewardTermCfg(func=position_fine, weight=0.1),
        "orientation": RewardTermCfg(func=orientation_reward, weight=-0.1),
        "action_rate": RewardTermCfg(func=action_rate, weight=-0.0001),
        "joint_velocity": RewardTermCfg(func=joint_velocity, weight=-0.0001),
    }
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
