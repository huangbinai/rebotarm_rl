import mujoco
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from rebotarm_rl.assets.resources import reach_scene_path

def _get_spec():
    spec = mujoco.MjSpec.from_file(str(reach_scene_path()))
    # The source XML contains the real robot's effort limits.  Convert only the
    # six arm actuators in the compiled training spec to bounded position servos;
    # the packaged source model remains unchanged and hash-verifiable.
    gains = {
        "joint1_torque": (100.0, 10.0),
        "joint2_torque": (100.0, 10.0),
        "joint3_torque": (100.0, 10.0),
        "joint4_torque": (40.0, 4.0),
        "joint5_torque": (40.0, 4.0),
        "joint6_torque": (40.0, 4.0),
    }
    for actuator in spec.actuators:
        if actuator.name in gains:
            kp, kv = gains[actuator.name]
            actuator.set_to_position(kp, kv, inheritrange=False)
    return spec

ROBOT_CFG = EntityCfg(
    spec_fn=_get_spec,
    articulation=EntityArticulationInfoCfg(
        # 仅包装XML中六个力矩执行器；Reach任务不控制夹爪。
        actuators=(XmlActuatorCfg(target_names_expr=("joint[1-6]",)),),
    ),
    init_state=EntityCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.8,
            "joint3": -1.0,
            "joint4": 0.3,
            "joint5": 0.0,
            "joint6": 0.0,
            "left_finger_joint": 0.03,
            "right_finger_joint": -0.03,
        },
        joint_vel={".*": 0.0},
    ),
)

ROBOT = SceneEntityCfg("robot", joint_names=("joint[1-6]",))


def gravity_compensated_spec():
    """Model gravity feedforward, routed through bounded arm joint actuation.

    Gravity remains enabled. Include the fixed camera/gripper payload masses.
    Joint actuation limits bound the combined PD and gravity contribution.
    """
    spec = _get_spec()
    for body in spec.bodies:
        if body.name != 'world':
            body.gravcomp = 1.0
    for i, limit in enumerate((27., 27., 27., 7., 7., 7.), start=1):
        joint = spec.joint(f'joint{i}')
        joint.actgravcomp = True
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.actfrcrange = [-limit, limit]
    return spec
