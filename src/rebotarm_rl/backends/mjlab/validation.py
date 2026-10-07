"""在创建训练器之前校验最终配置，并识别对应的策略契约。"""
from rebotarm_rl.contracts.policy import REACH_V1, REACH_ALIGNED, REACH_ORIENTATION_ALIGNED, PolicyContract
from mjlab.envs.mdp.actions.actions import JointPositionActionCfg
from .tasks.reach.observations import orientation_error
from .tasks.reach.actions import HeldRelativePositionActionCfg
from mjlab.envs.mdp import last_action
from rebotarm_rl.contracts.policy import REACH_GRAVITY
from .robots.rebotarm import gravity_compensated_spec, _get_spec


def validate_config(cfg) -> PolicyContract:
    """同时校验观测语义、动作缩放和时序，禁止只凭输入维度判断兼容。"""
    fn = cfg.observations["actor"].terms["orientation_error"].func
    action = cfg.actions["joint_position"]
    gravity_compensated = cfg.scene.entities['robot'].spec_fn is gravity_compensated_spec
    if fn is orientation_error:
        if type(action) is JointPositionActionCfg:
            reward_weight = cfg.rewards['orientation'].weight
            c = REACH_ORIENTATION_ALIGNED if reward_weight == REACH_ORIENTATION_ALIGNED.orientation_reward_weight else REACH_ALIGNED
        else:
            c = REACH_V1
    else:
        raise ValueError("未注册的姿态编码，不能推断策略契约")
    if gravity_compensated:
        if c != REACH_ALIGNED:
            raise ValueError('重力补偿任务必须沿用对齐版动作及-0.1姿态奖励')
        c = REACH_GRAVITY
    elif cfg.scene.entities['robot'].spec_fn is not _get_spec:
        raise ValueError('未注册的机器人伺服模型')
    aligned = c in (REACH_ALIGNED, REACH_ORIENTATION_ALIGNED, REACH_GRAVITY)
    if (tuple(cfg.observations["actor"].terms) != c.observation_fields
            or cfg.sim.mujoco.timestep != c.physics_dt_s
            or cfg.decimation != c.decimation
            or action.scale != c.action_scale
            or tuple(action.actuator_names) != ("joint[1-6]",)
            or type(action) is not (JointPositionActionCfg if aligned else HeldRelativePositionActionCfg)
            or action.offset != 0.0
            or action.clip != (None if aligned else {"joint[1-6]": (-c.action_scale, c.action_scale)})
            or (aligned and not action.use_default_offset)
            or cfg.rewards['orientation'].weight != c.orientation_reward_weight
            or cfg.observations['actor'].terms['last_action'].func is not last_action):
        raise ValueError("生效的环境配置违反策略契约")
    return c
