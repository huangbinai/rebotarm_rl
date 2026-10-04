"""Versioned description of the existing Reach policy; no simulator imports."""
from dataclasses import asdict, dataclass
from .robot import ARM_JOINTS


@dataclass(frozen=True)
class PolicyContract:
    version: str = "reach-effort-v1"
    task_id: str = "RebotArm-Reach-Mjlab"
    joints: tuple[str, ...] = ARM_JOINTS
    action_type: str = "joint_effort"
    action_scale: float = 1.0
    observation_fields: tuple[str, ...] = (
        "joint_pos", "joint_vel", "position_error", "orientation_error"
    )
    # Robot selection restricts observations to the six arm joints.
    observation_sizes: tuple[int, ...] = (6, 6, 3, 3)
    orientation_encoding: str = "legacy-reach-v1"
    physics_dt_s: float = 0.002
    decimation: int = 10
    success_position_m: float = 0.01
    success_orientation_rad: float = 0.05236

    def to_dict(self) -> dict:
        return asdict(self)

    def validate_shapes(self, observation_size: int, action_size: int) -> None:
        if observation_size != sum(self.observation_sizes) or action_size != len(self.joints):
            raise ValueError("Policy observation/action dimensions violate reach-effort-v1")

    def require_compatible(self, other: dict) -> None:
        import json
        if json.loads(json.dumps(self.to_dict())) != json.loads(json.dumps(other)):
            raise ValueError("Incompatible policy contract; do not reuse checkpoint silently")


REACH_V1 = PolicyContract()
