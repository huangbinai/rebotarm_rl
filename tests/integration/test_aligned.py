import pytest


def test_aligned_recipe_and_contract():
    pytest.importorskip('mjlab')
    from rebotarm_rl.backends.mjlab.tasks.reach.aligned import make_aligned_env_cfg
    from rebotarm_rl.training.rsl_rl.reach_ppo import aligned_runner_cfg
    from rebotarm_rl.backends.mjlab.validation import validate_config
    from rebotarm_rl.contracts.policy import REACH_ALIGNED, REACH_V1
    cfg = make_aligned_env_cfg()
    assert validate_config(cfg) == REACH_ALIGNED
    with pytest.raises(ValueError):
        REACH_ALIGNED.require_compatible(REACH_V1.to_dict())
    assert cfg.commands['reach'].resampling_time_range == (4., 4.)
    assert 'success' not in cfg.terminations
    assert cfg.episode_length_s == 12.
    assert cfg.curriculum['action_rate'].params['stages'][0]['weight'] == -.005
    agent = aligned_runner_cfg()
    assert agent.actor.hidden_dims == (64, 64)
    assert agent.actor.distribution_cfg['init_std'] == 1.
    assert agent.algorithm.entropy_coef == .001
    assert agent.algorithm.num_learning_epochs == 8
    cfg.actions['joint_position'].use_default_offset = False
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_aligned_gpu_anchor_and_action():
    torch = pytest.importorskip('torch')
    pytest.importorskip('mjlab')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from rebotarm_rl.backends.mjlab.tasks.reach.aligned import make_aligned_env_cfg
    env = ManagerBasedRlEnv(make_aligned_env_cfg(num_envs=2), device='cuda:0')
    try:
        env.reset()
        robot = env.scene['robot']
        command = env.command_manager.get_term('reach')
        torch.testing.assert_close(command.anchor_pos, robot.data.site_pose_w[:, command.site_id, :3])
        anchor = command.anchor_pos.clone()
        action = torch.full((2, 6), .2, device='cuda:0')
        env.step(action)
        term = env.action_manager.get_term('joint_position')
        torch.testing.assert_close(robot.data.joint_pos_target[:, term.target_ids],
                                   robot.data.default_joint_pos[:, term.target_ids] + .1)
        command._resample_command(torch.arange(2, device='cuda:0'))
        torch.testing.assert_close(anchor, command.anchor_pos)
        assert torch.all((command.target_pos - anchor).norm(dim=-1) <= .060001)
    finally:
        env.close()
