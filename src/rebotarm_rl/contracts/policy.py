"""带版本的Reach策略契约；不导入仿真引擎。"""
from dataclasses import asdict, dataclass
from .robot import ARM_JOINTS


@dataclass(frozen=True)
class PolicyContract:
    version: str = "reach-position-v1"
    task_id: str = "RebotArm-Reach-Mjlab"
    joints: tuple[str, ...] = ARM_JOINTS
    action_type: str = "joint_position_delta"
    action_scale: float = 0.25
    observation_fields: tuple[str, ...] = (
        "joint_pos", "joint_vel", "position_error", "orientation_error", "last_action"
    )
    # 关节选择器将观测限制为机械臂的六个关节。
    observation_sizes: tuple[int, ...] = (6, 6, 3, 3, 6)
    orientation_encoding: str = "world-current-minus-target-rotvec-wxyz"
    physics_dt_s: float = 0.002
    decimation: int = 10
    success_position_m: float = 0.01
    success_orientation_rad: float = 0.05236

    def to_dict(self) -> dict:
        return asdict(self)

    def validate_shapes(self, observation_size: int, action_size: int) -> None:
        if observation_size != sum(self.observation_sizes) or action_size != len(self.joints):
            raise ValueError("策略观测/动作维度与契约不一致")

    def require_compatible(self, other: dict) -> None:
        import json
        if json.loads(json.dumps(self.to_dict())) != json.loads(json.dumps(other)):
            raise ValueError("策略契约不兼容，禁止直接复用checkpoint")


REACH_V1 = PolicyContract()


def contract_for_task(task_id: str) -> PolicyContract:
    """根据任务名取契约，不用张量维度猜测策略版本。"""
    if REACH_V1.task_id == task_id:
        return REACH_V1
    raise ValueError(f"未知任务: {task_id}")
