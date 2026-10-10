"""Random-reset rejection and hold metrics need independent regression checks."""
import pytest


def test_rejects_limits_and_enabled_self_contact():
    mujoco = pytest.importorskip('mujoco')
    np = pytest.importorskip('numpy')
    pytest.importorskip('mjlab')
    from rebotarm_rl.backends.mjlab.evaluation.random_start import state_rejection, sample_initial_states
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


def test_hold_timing_jitter_and_joint_travel():
    np = pytest.importorskip('numpy')
    pytest.importorskip('mjlab')
    from rebotarm_rl.backends.mjlab.evaluation.random_start import trajectory_metrics, summarize
    poses = np.zeros((30, 7)); poses[:, 3] = 1
    poses[:4, 0] = .02
    joints = np.zeros((30, 8)); joints[:, 0] = np.arange(30) * .01
    result = trajectory_metrics(poses, joints, np.zeros(3), np.array([1., 0, 0, 0]), 26, .02)
    assert result['tail_success']
    assert result['first_success_time_s'] == pytest.approx(.08)
    assert result['first_hold_completion_time_s'] == pytest.approx(.58)
    assert result['tail_position_jitter_rms_m'] == 0
    assert result['joint_range_rad'][0] == pytest.approx(.29)
    assert result['joint_total_travel_rad'][0] == pytest.approx(.29)
    # A last-point hit is not a hold; unsuccessful arrival stays None.
    poses[-2, 0] = .02
    broken = trajectory_metrics(poses, joints, np.zeros(3), np.array([1., 0, 0, 0]), 26, .02)
    assert not broken['tail_success']
    assert broken['first_hold_completion_time_s'] is None
    poses[:, 0] = .02
    failed = trajectory_metrics(poses, joints, np.zeros(3), np.array([1., 0, 0, 0]), 26, .02)
    assert summarize([failed])['mean_first_success_time_s_successes_only'] is None


def test_frozen_test_rejects_weight_changes_and_reused_seeds(tmp_path):
    import json
    pytest.importorskip('mjlab')
    from rebotarm_rl.contracts.artifacts import file_hash
    from rebotarm_rl.backends.mjlab.evaluation.random_start import validate_frozen_selection
    checkpoint = tmp_path / 'model.pt'; checkpoint.write_bytes(b'frozen test identity')
    path = tmp_path / 'selection.json'
    settings = {'target_seed': 230000, 'initial_seed': 230001, 'episodes': 200}
    record = {'checkpoint_sha256': file_hash(checkpoint), 'test_settings': settings,
              'validation_target_seeds': [130000], 'validation_initial_seeds': [130001]}
    path.write_text(json.dumps(record))
    assert validate_frozen_selection(path, [checkpoint], settings)['record'] == record
    with pytest.raises(ValueError, match='settings'):
        validate_frozen_selection(path, [checkpoint], dict(settings, episodes=100))
    record['validation_initial_seeds'].append(230001)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='fresh'):
        validate_frozen_selection(path, [checkpoint], settings)
    checkpoint.write_bytes(b'overwritten')
    with pytest.raises(ValueError, match='checkpoint'):
        validate_frozen_selection(path, [checkpoint], settings)
