"""IK witnesses must independently reproduce fixed-orientation feasible targets."""
import pytest


@pytest.fixture
def robot():
    mujoco = pytest.importorskip('mujoco')
    import numpy as np
    from rebotarm_rl.assets.resources import reach_scene_path
    model = mujoco.MjModel.from_xml_path(str(reach_scene_path()))
    names = [f'joint{i}' for i in range(1, 7)] + ['left_finger_joint', 'right_finger_joint']
    ids = np.array([model.joint(name).id for name in names])
    qadr, vadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    default = np.array([0., -.8, -1., .3, 0., 0., .03, -.03])
    data = mujoco.MjData(model)
    data.qpos[qadr] = default
    mujoco.mj_forward(model, data)
    site = model.site('ee_site').id
    anchor = data.site_xpos[site].copy()
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site])
    return model, qadr, vadr, ids, site, default, anchor, quat


def test_reachable_target_witnesses_are_deterministic_and_valid(robot):
    import mujoco
    import numpy as np
    from rebotarm_rl.backends.mjlab.evaluation.reachability import sample_reachable_targets, IK
    from rebotarm_rl.backends.mjlab.evaluation.state_checks import state_rejection
    targets, audit = sample_reachable_targets(*robot, (.10, .12), 8, 330000)
    again, repeated = sample_reachable_targets(*robot, (.10, .12), 8, 330000)
    np.testing.assert_array_equal(targets, again)
    assert audit == repeated
    model, qadr, _, ids, site, _, anchor, quat = robot
    radii = np.linalg.norm(targets - anchor, axis=1)
    assert (radii >= .10).all() and (radii <= .12).all()
    data = mujoco.MjData(model)
    for target, witness in zip(targets, audit['witnesses']):
        q = np.array(witness['joint_position_rad'])
        assert state_rejection(model, data, qadr, ids, q) is None
        assert np.linalg.norm(data.site_xpos[site] - target) < IK.position_tolerance_m
        actual = np.empty(4)
        mujoco.mju_mat2Quat(actual, data.site_xmat[site])
        assert 2 * np.arccos(np.clip(abs(actual @ quat), 0, 1)) < IK.orientation_tolerance_rad


def test_unsolved_target_is_not_certified(robot):
    import numpy as np
    from rebotarm_rl.backends.mjlab.evaluation.reachability import solve_pose, IkSettings
    model, qadr, vadr, ids, site, default, _, quat = robot
    witness, reason = solve_pose(model, qadr, vadr, ids, site, default, np.array([9., 9., 9.]), quat,
                                 np.random.default_rng(1), IkSettings(iterations=5, restarts=1))
    assert witness is None and reason == 'ik_unresolved'


def test_collision_at_solved_pose_cannot_be_a_witness(robot, monkeypatch):
    import numpy as np
    from rebotarm_rl.backends.mjlab.evaluation import reachability
    model, qadr, vadr, ids, site, default, anchor, quat = robot
    # Exercise the final geometric gate even when the pose residual is zero.
    monkeypatch.setattr(reachability, 'state_rejection', lambda *args: 'self_collision')
    witness, reason = reachability.solve_pose(model, qadr, vadr, ids, site, default, anchor, quat,
                                              np.random.default_rng(1), reachability.IkSettings(restarts=1))
    assert witness is None and reason == 'self_collision'
