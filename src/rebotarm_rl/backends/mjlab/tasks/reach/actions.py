"""控制步开始时采样相对目标，在所有物理子步中保持该目标。"""
from dataclasses import dataclass
from mjlab.envs.mdp.actions.actions import RelativeJointPositionActionCfg, RelativeJointPositionAction


@dataclass(kw_only=True)
class HeldRelativePositionActionCfg(RelativeJointPositionActionCfg):
    def build(self, env):
        return HeldRelativePositionAction(self, env)


class HeldRelativePositionAction(RelativeJointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        self._target = self._entity.data.joint_pos[:, self._target_ids].clone() + self._processed_actions

    def apply_actions(self):
        self._entity.set_joint_position_target(self._target, joint_ids=self._target_ids)
