"""独立于引擎的关节顺序与单位约定。"""
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
QUATERNION_ORDER = "wxyz"
JOINT_POSITION_UNIT = "rad"
JOINT_TORQUE_UNIT = "N*m"
