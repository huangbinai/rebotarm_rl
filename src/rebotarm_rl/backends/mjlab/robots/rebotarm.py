import mujoco
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from rebotarm_rl.assets.resources import reach_scene_path

def _get_spec():
    return mujoco.MjSpec.from_file(str(reach_scene_path()))

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
