"""在创建训练器之前校验最终配置，并识别对应的策略契约。"""
from rebotarm_rl.contracts.policy import REACH_V1, REACH_V2, PolicyContract
from .tasks.reach.observations import orientation_error, orientation_error_v2


def validate_config(cfg) -> PolicyContract:
    """同时校验观测语义、动作缩放和时序，禁止只凭输入维度判断兼容。"""
    fn = cfg.observations["actor"].terms["orientation_error"].func
    if fn is orientation_error:
        c = REACH_V1
    elif fn is orientation_error_v2:
        c = REACH_V2
    else:
        raise ValueError("未注册的姿态编码，不能推断策略契约")
    action = cfg.actions["joint_effort"]
    if (tuple(cfg.observations["actor"].terms) != c.observation_fields
            or cfg.sim.mujoco.timestep != c.physics_dt_s
            or cfg.decimation != c.decimation
            or action.scale != c.action_scale
            or tuple(action.actuator_names) != ("joint[1-6]",)):
        raise ValueError("生效的环境配置违反策略契约")
    return c
