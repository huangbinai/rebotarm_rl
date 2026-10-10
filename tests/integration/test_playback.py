"""回放配置隔离、即时指标和仅渲染标记。"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest


def test_playback_keeps_training_contract_and_dynamics():
    pytest.importorskip("mjlab")
    from rebotarm_rl.backends.mjlab.play import playback_config
    from rebotarm_rl.backends.mjlab.tasks.reach.aligned import make_gravity_fixed_env_cfg
    from rebotarm_rl.backends.mjlab.validation import validate_config
    from rebotarm_rl.contracts.policy import REACH_GRAVITY_FIXED
    train = make_gravity_fixed_env_cfg(num_envs=1)
    for mode, interval in (("fixed", (1e9, 1e9)), ("refresh", (4., 4.))):
        play = playback_config(mode, 19)
        assert validate_config(play) == REACH_GRAVITY_FIXED
        for name in ("sim", "actions", "observations", "rewards", "curriculum"):
            assert getattr(play, name) == getattr(train, name)
        assert play.scene.entities["robot"].spec_fn is train.scene.entities["robot"].spec_fn
        assert play.terminations == {}
        command = copy.deepcopy(play.commands["reach"])
        assert command.resampling_time_range == interval
        command.resampling_time_range = train.commands["reach"].resampling_time_range
        assert command == train.commands["reach"]
    assert train.episode_length_s == 12.
    assert "time_out" in train.terminations
    assert train.commands["reach"].resampling_time_range == (4., 4.)


def test_live_errors_and_world_frame_markers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    from rebotarm_rl.backends.mjlab.play import reach_state, draw_reach
    from mjlab.viewer.debug_visualizer import NullDebugVisualizer
    # 目标与TCP相差90度，位置相差2cm；不读取缓存metrics。
    pose = torch.tensor([[[.02, 0., 0., 1., 0., 0., 0.]]])
    target = torch.tensor([[0., 0., 0., np.sqrt(.5), 0., 0., np.sqrt(.5)]])
    command = SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(site_pose_w=pose)),
                              site_id=0, command=target)
    base = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    current, goal, pe, oe, success = reach_state(base)
    assert pe == pytest.approx(.02)
    assert oe == pytest.approx(np.pi/2)
    assert not success

    class Recorder(NullDebugVisualizer):
        def __init__(self):
            super().__init__()
            self.frames = []
        def add_frame(self, position, rotation_matrix, **kwargs):
            self.frames.append((position.copy(), rotation_matrix.copy()))

    vis = Recorder()
    draw_reach(vis, current, goal)
    np.testing.assert_allclose(vis.frames[0][0], goal[:3])
    np.testing.assert_allclose(vis.frames[0][1][:, 0], [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(vis.frames[1][1], np.eye(3))
    command.command = pose[:, 0].clone()
    command.command[:, 3:] *= -1
    assert reach_state(base)[3] == pytest.approx(0.)
    assert reach_state(base)[4]
