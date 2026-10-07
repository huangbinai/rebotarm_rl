"""带版本的Reach策略契约；不导入仿真引擎。"""
from dataclasses import asdict, dataclass, replace
from .robot import ARM_JOINTS


@dataclass(frozen=True)
class PolicyContract:
    version: str = "reach-position-v1.1"
    target_update: str = "sample_current_at_control_step_hold_substeps"
    last_action_encoding: str = "unscaled_action_after_optional_runner_clip"
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
    orientation_reward_weight: float = -0.1

    def to_dict(self) -> dict:
        result = asdict(self)
        # Published V1.1/V2 checkpoints predate the reward-specific experiment.
        # Keep their exact serialized contract; new versions carry the weight.
        if self.version in ("reach-position-v1.1", "reach-position-official-aligned-v2"):
            result.pop("orientation_reward_weight")
        return result

    def validate_shapes(self, observation_size: int, action_size: int) -> None:
        if observation_size != sum(self.observation_sizes) or action_size != len(self.joints):
            raise ValueError("策略观测/动作维度与契约不一致")

    def require_compatible(self, other: dict) -> None:
        import json
        if json.loads(json.dumps(self.to_dict())) != json.loads(json.dumps(other)):
            raise ValueError("策略契约不兼容，禁止直接复用checkpoint")


REACH_V1 = PolicyContract()
REACH_ALIGNED = PolicyContract(
    version="reach-position-official-aligned-v2",
    task_id="RebotArm-Reach-OfficialAligned-Mjlab",
    action_type="joint_position_default_offset",
    target_update="default_joint_position_plus_scaled_action_hold_substeps",
    action_scale=0.5,
)
REACH_ORIENTATION_ALIGNED = PolicyContract(
    version="reach-position-official-aligned-v2-orientation-v1",
    task_id="RebotArm-Reach-OfficialAligned-Orientation-Mjlab",
    action_type="joint_position_default_offset",
    target_update="default_joint_position_plus_scaled_action_hold_substeps",
    action_scale=0.5,
    orientation_reward_weight=-0.2,
)


REACH_GRAVITY = replace(
    REACH_ALIGNED,
    version="reach-position-gravity-compensated-v3",
    task_id="RebotArm-Reach-GravityComp-Mjlab",
)


REACH_GRAVITY_FIXED = replace(
    REACH_GRAVITY,
    version="reach-position-gravity-compensated-v3-fixed-penalties-v1",
    task_id="RebotArm-Reach-GravityComp-FixedPenalties-Mjlab",
)


def contract_for_task(task_id: str) -> PolicyContract:
    """根据任务名取契约，不用张量维度猜测策略版本。"""
    for contract in (REACH_V1, REACH_ALIGNED, REACH_ORIENTATION_ALIGNED, REACH_GRAVITY, REACH_GRAVITY_FIXED):
        if contract.task_id == task_id:
            return contract
    raise ValueError(f"未知任务: {task_id}")
