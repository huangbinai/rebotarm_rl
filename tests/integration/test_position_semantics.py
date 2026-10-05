"""奖励方向、控制步目标锁定及动作历史的回归测试。"""
import pytest

def test_position_reward_decreases_with_distance(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('mjlab')
    from rebotarm_rl.backends.mjlab.tasks.reach import rewards
    distance = torch.tensor([0., .001, .01, .06, .2, 1.])
    monkeypatch.setattr(rewards, '_errors', lambda env: (distance, torch.zeros_like(distance)))
    fine = rewards.position_fine(None)
    total = -.2 * distance + .1 * fine
    assert fine[0] == 1
    assert torch.all(torch.diff(fine) < 0)
    assert torch.all(torch.diff(total) < 0)


def test_gpu_target_is_held_and_history_is_unscaled():
    torch = pytest.importorskip('torch')
    pytest.importorskip('mjlab')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from rebotarm_rl.backends.mjlab.tasks.reach.config import make_env_cfg
    cfg = make_env_cfg(num_envs=1)
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        env.reset()
        robot = env.scene['robot']
        term = env.action_manager.get_term('joint_position')
        action = torch.tensor([[2., -.8, .4, -.2, .1, -2.]], device='cuda:0')
        q = robot.data.joint_pos[:, term.target_ids].clone()
        env.action_manager.process_action(action)
        expected = q + torch.tensor([[.25, -.2, .1, -.05, .025, -.25]], device='cuda:0')
        for _ in range(10):
            env.action_manager.apply_action()
            torch.testing.assert_close(robot.data.joint_pos_target[:, term.target_ids], expected)
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=cfg.sim.mujoco.timestep)
        env.sim.forward()
        obs = env.observation_manager.compute_group('actor')
        torch.testing.assert_close(obs[:, -6:], action)
        assert (robot.data.joint_pos[:, term.target_ids] - q).abs().max() > 1e-5
    finally:
        env.close()
