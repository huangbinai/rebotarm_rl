"""Validate effective engine config against the existing policy contract."""
from rebotarm_rl.contracts.policy import REACH_V1


def validate_config(cfg) -> None:
    c = REACH_V1
    action = cfg.actions["joint_effort"]
    if (tuple(cfg.observations["actor"].terms) != c.observation_fields
            or cfg.sim.mujoco.timestep != c.physics_dt_s
            or cfg.decimation != c.decimation
            or action.scale != c.action_scale
            or tuple(action.actuator_names) != ("joint[1-6]",)):
        raise ValueError("Effective environment violates reach-effort-v1")
