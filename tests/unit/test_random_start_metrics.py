import pytest


def test_hold_timing_jitter_and_joint_travel():
    np = pytest.importorskip('numpy')
    from rebotarm_rl.evaluation.metrics import trajectory_metrics, summarize
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
