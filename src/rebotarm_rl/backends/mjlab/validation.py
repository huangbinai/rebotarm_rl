"""在创建训练器之前校验最终配置，并识别对应的策略契约。"""
from rebotarm_rl.contracts.policy import REACH_V1, REACH_ALIGNED, PolicyContract
from mjlab.envs.mdp.actions.actions import JointPositionActionCfg
from .tasks.reach.observations import orientation_error
from .tasks.reach.actions import HeldRelativePositionActionCfg
from mjlab.envs.mdp import last_action


def validate_config(cfg) -> PolicyContract:
    """同时校验观测语义、动作缩放和时序，禁止只凭输入维度判断兼容。"""
    fn = cfg.observations["actor"].terms["orientation_error"].func
    action = cfg.actions["joint_position"]
    if fn is orientation_error:
        c = REACH_ALIGNED if type(action) is JointPositionActionCfg else REACH_V1
    else:
        raise ValueError("未注册的姿态编码，不能推断策略契约")
    action = cfg.actions["joint_position"]
    if (tuple(cfg.observations["actor"].terms) != c.observation_fields
            or cfg.sim.mujoco.timestep != c.physics_dt_s
            or cfg.decimation != c.decimation
            or action.scale != c.action_scale
            or tuple(action.actuator_names) != ("joint[1-6]",)
            or type(action) is not (JointPositionActionCfg if c == REACH_ALIGNED else HeldRelativePositionActionCfg)
            or action.offset != 0.0
            or action.clip != (None if c == REACH_ALIGNED else {"joint[1-6]": (-c.action_scale, c.action_scale)})
            or (c == REACH_ALIGNED and not action.use_default_offset)
            or cfg.observations['actor'].terms['last_action'].func is not last_action):
        raise ValueError("生效的环境配置违反策略契约")
    return c
