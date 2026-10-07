"""Gravity feedforward remains bounded and preserves the published baselines."""
import pytest


def test_gravity_contract_and_model():
    pytest.importorskip('mjlab')
    import numpy as np
    from rebotarm_rl.backends.mjlab.tasks.reach.aligned import make_aligned_env_cfg, make_gravity_aligned_env_cfg
    from rebotarm_rl.backends.mjlab.validation import validate_config
    from rebotarm_rl.contracts.policy import REACH_ALIGNED, REACH_GRAVITY
    from rebotarm_rl.backends.mjlab.robots.rebotarm import _get_spec, gravity_compensated_spec
    assert validate_config(make_aligned_env_cfg()) == REACH_ALIGNED
    assert validate_config(make_gravity_aligned_env_cfg()) == REACH_GRAVITY
    with pytest.raises(ValueError):
        REACH_GRAVITY.require_compatible(REACH_ALIGNED.to_dict())
    old, new = _get_spec().compile(), gravity_compensated_spec().compile()
    np.testing.assert_array_equal(old.opt.gravity, new.opt.gravity)
    np.testing.assert_array_equal(old.body_mass, new.body_mass)
    assert not old.jnt_actgravcomp.any()
    assert new.jnt_actgravcomp[:6].all()
    assert new.jnt_actfrclimited[:6].all()
    np.testing.assert_array_equal(new.jnt_actfrcrange[:6,1], [27,27,27,7,7,7])


def test_gpu_gravity_hold_and_total_force_limit():
    torch = pytest.importorskip('torch')
    pytest.importorskip('mjlab')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from rebotarm_rl.backends.mjlab.tasks.reach.aligned import make_gravity_aligned_env_cfg
    env = ManagerBasedRlEnv(make_gravity_aligned_env_cfg(num_envs=1), device='cuda:0')
    try:
        env.reset()
        robot = env.scene['robot']
        initial = robot.data.joint_pos[:,:6].clone()
        for _ in range(250):
            env.step(torch.zeros((1,6),device='cuda:0'))
        assert (robot.data.joint_pos[:,:6]-initial).abs().max() < .001
        # Force saturation must apply after adding gravity, not only to PD.
        env.step(torch.full((1,6),100.,device='cuda:0'))
        force = env.sim.data.qfrc_actuator[:,robot.indexing.joint_v_adr[:6]]
        limits = torch.tensor([27,27,27,7,7,7],device='cuda:0')
        assert torch.all(force.abs() <= limits+1e-4)
    finally:
        env.close()


def test_fixed_penalties_only_remove_curriculum():
    pytest.importorskip('mjlab')
    from rebotarm_rl.backends.mjlab.tasks.reach.aligned import (
        make_gravity_aligned_env_cfg, make_gravity_fixed_env_cfg,
    )
    from rebotarm_rl.backends.mjlab.validation import validate_config
    from rebotarm_rl.contracts.policy import REACH_GRAVITY, REACH_GRAVITY_FIXED
    baseline = make_gravity_aligned_env_cfg()
    fixed = make_gravity_fixed_env_cfg()
    assert baseline.curriculum
    assert not fixed.curriculum
    baseline.curriculum = {}
    assert baseline == fixed
    assert validate_config(fixed) == REACH_GRAVITY_FIXED
    for source, target in [(REACH_GRAVITY, REACH_GRAVITY_FIXED),
                           (REACH_GRAVITY_FIXED, REACH_GRAVITY)]:
        with pytest.raises(ValueError):
            target.require_compatible(source.to_dict())
    fixed.rewards['action_rate'].weight = -0.005
    with pytest.raises(ValueError, match='固定惩罚'):
        validate_config(fixed)
