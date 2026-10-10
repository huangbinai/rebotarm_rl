"""Random-reset rejection and hold metrics need independent regression checks."""
import pytest


def test_rejects_limits_and_enabled_self_contact():
    mujoco = pytest.importorskip('mujoco')
    np = pytest.importorskip('numpy')
    from rebotarm_rl.backends.mjlab.evaluation.state_checks import state_rejection, sample_initial_states
    # Six independent slider bodies; the first collides with a fixed obstacle.
    bodies = ''.join(f'<body pos="{i * 3} 0 0"><joint type="slide" axis="1 0 0" range="-1 1"/>'
                     '<geom type="sphere" size="0.1" mass="1"/></body>' for i in range(6))
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1" '
                                         'pos="0.5 0 0"/>' + bodies + '</worldbody></mujoco>')
    data = mujoco.MjData(model)
    ids = np.arange(6)
    home = np.zeros(6)
    assert state_rejection(model, data, ids, ids, home) is None
    collision = home.copy(); collision[0] = .5
    assert state_rejection(model, data, ids, ids, collision) == 'self_collision'
    limit = home.copy(); limit[1] = .995
    assert state_rejection(model, data, ids, ids, limit) == 'joint_limit'
    states, audit = sample_initial_states(model, ids, ids, home, 1.5, 100, 42)
    assert audit['rejected']['self_collision'] > 0
    assert audit['rejected']['joint_limit'] > 0
    assert all(state_rejection(model, data, ids, ids, q) is None for q in states)
    again, _ = sample_initial_states(model, ids, ids, home, 1.5, 100, 42)
    np.testing.assert_array_equal(again, states)


def test_simultaneous_limit_and_collision_count_independently():
    mujoco = pytest.importorskip('mujoco')
    import numpy as np
    from rebotarm_rl.backends.mjlab.evaluation.state_checks import trajectory_validity
    bodies = ''.join(f'<body pos="{i * 3} 0 0"><joint type="slide" axis="1 0 0" range="-1 1"/>'
                     '<geom type="sphere" size="0.1" mass="1"/></body>' for i in range(6))
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1" '
                                         'pos="0.5 0 0"/>' + bodies + '</worldbody></mujoco>')
    joints = np.zeros((2, 6))
    joints[0, :2] = [.5, 1.2]
    joints[1, 0] = np.nan
    assert trajectory_validity(model, np.arange(6), np.arange(6), joints) == {
        'joint_limit': 1, 'self_collision': 1, 'nonfinite': 1}


def test_gpu_batched_reset_clears_history_and_keeps_goals():
    torch = pytest.importorskip('torch')
    pytest.importorskip('mjlab')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    import numpy as np
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from rebotarm_rl.backends.mjlab.tasks.reach.aligned import make_gravity_fixed_env_cfg
    from rebotarm_rl.backends.mjlab.evaluation.rollout import gpu_rollout
    cfg = make_gravity_fixed_env_cfg(num_envs=3)
    cfg.terminations = {}
    cfg.commands['reach'].position_radius = 0.
    cfg.commands['reach'].resampling_time_range = (1e9, 1e9)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device='cuda:0'))
    try:
        base = env.unwrapped
        robot = base.scene['robot']
        command = base.command_manager.get_term('reach')
        env.reset()
        initial = robot.data.default_joint_pos.cpu().numpy().copy()
        initial[:, 0] += np.array([-.05, 0., .05])
        targets = command.target_pos.cpu().numpy().copy()
        targets[:, 0] += np.array([-.02, .03, .04])
        quat = command.target_quat[0].cpu().numpy().copy()
        env.step(torch.full((3, 6), .1, device='cuda:0'))
        seen = []
        def policy(obs):
            seen.append(obs['actor'].clone())
            return torch.full((3, 6), .02, device='cuda:0')
        _, joints, observations = gpu_rollout(env, policy, initial, targets, quat, 3, 3, 'cuda:0')
        np.testing.assert_allclose(joints[0], initial, atol=2e-7, rtol=0)
        np.testing.assert_array_equal(observations[0, :, 6:12], 0.)
        np.testing.assert_array_equal(observations[0, :, -6:], 0.)
        np.testing.assert_allclose(observations[1, :, -6:], .02)
        np.testing.assert_allclose(command.target_pos.cpu().numpy(), targets)
        assert len(seen) == 3
    finally:
        env.close()


@pytest.mark.parametrize('default_dtype', ['float32', 'float64'])
def test_cpu_control_preserves_default_reference_precision_and_delta_semantics(default_dtype):
    mujoco = pytest.importorskip('mujoco')
    torch = pytest.importorskip('torch')
    import numpy as np
    from rebotarm_rl.backends.mjlab.evaluation.rollout import CpuRollout
    from rebotarm_rl.contracts.policy import REACH_GRAVITY_FIXED, REACH_V1
    bodies = ''.join(f'<body pos="{i * 3} 0 0"><joint name="j{i}" type="slide"/>'
                     '<geom type="sphere" size=".1" mass="1"/></body>' for i in range(6))
    actuators = ''.join(f'<position name="robot/joint{i+1}_torque" joint="j{i}"/>' for i in range(6))
    model = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{bodies}</worldbody><actuator>{actuators}</actuator></mujoco>')
    default = np.full(6, .123456789, dtype=default_dtype)
    initial = np.full(6, .32123456789)
    action = torch.tensor([[2., -.8, .4, -.2, .1, -2.]])
    for contract in (REACH_GRAVITY_FIXED, REACH_V1):
        cpu = CpuRollout(model, np.arange(6), np.arange(6), 0, default, contract, 0, None)
        cpu.reset(initial)
        cpu.step(action)
        if contract.action_type == 'joint_position_delta':
            expected = torch.as_tensor(initial) + (action[0] * contract.action_scale).clamp(-contract.action_scale, contract.action_scale)
        else:
            expected = torch.as_tensor(default) + action[0] * contract.action_scale
        np.testing.assert_array_equal(cpu.data.ctrl, expected.numpy())
        np.testing.assert_array_equal(cpu.last_action, action[0].numpy())
