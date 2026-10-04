"""用独立旋转矩阵参照和权重文件验证V2语义及版本隔离。"""
import json
import math
import pytest

torch = pytest.importorskip('torch')
from rebotarm_rl.backends.mjlab.rotation import rotation_error_wxyz


def test_known_angles_and_double_cover():
    identity = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    for angle in (0., 1e-10, math.pi / 2, -math.pi / 2, math.pi - 1e-9):
        for axis in range(3):
            q = identity.clone()
            q[0] = math.cos(angle / 2)
            q[axis + 1] = math.sin(angle / 2)
            expected = torch.zeros(3, dtype=q.dtype)
            expected[axis] = angle
            for a, b in ((q, identity), (-q, identity), (q, -identity)):
                torch.testing.assert_close(rotation_error_wxyz(a, b), expected)
    q = torch.tensor([0., -1., 0., 0.], dtype=identity.dtype)
    torch.testing.assert_close(rotation_error_wxyz(q, identity),
                               rotation_error_wxyz(-q, identity))


def test_world_frame_against_independent_matrix_reference():
    # SciPy使用矩阵复合求参照，避免用相同四元数公式自证。
    rotation = pytest.importorskip('scipy.spatial.transform').Rotation
    import numpy as np
    rng = np.random.default_rng(42)
    current = rotation.random(128, random_state=rng)
    target = rotation.random(128, random_state=rng)
    expected = rotation.from_matrix(current.as_matrix() @ target.as_matrix().transpose(0, 2, 1)).as_rotvec()
    a = torch.tensor(current.as_quat()[:, [3, 0, 1, 2]])
    b = torch.tensor(target.as_quat()[:, [3, 0, 1, 2]])
    actual = rotation_error_wxyz(a, b)
    torch.testing.assert_close(actual, torch.tensor(expected))
    torch.testing.assert_close(rotation_error_wxyz(b, a), -actual)
    assert torch.isfinite(actual).all()
    if torch.cuda.is_available():
        torch.testing.assert_close(rotation_error_wxyz(a.cuda(), b.cuda()).cpu(), actual)


def test_v1_v2_config_only_changes_orientation():
    pytest.importorskip('mjlab')
    from rebotarm_rl.backends.mjlab.tasks.reach.config import make_env_cfg
    from rebotarm_rl.backends.mjlab.validation import validate_config
    from rebotarm_rl.contracts.policy import REACH_V1, REACH_V2
    a, b = make_env_cfg(), make_env_cfg(version=2)
    assert validate_config(a) == REACH_V1
    assert validate_config(b) == REACH_V2
    assert a.actions == b.actions
    assert a.rewards == b.rewards
    assert a.terminations == b.terminations
    assert a.decimation == b.decimation
    assert a.sim == b.sim
    assert a.commands == b.commands
    with pytest.raises(ValueError):
        make_env_cfg(version=3)


def test_checkpoint_version_guards(tmp_path):
    pytest.importorskip('mjlab')
    from rebotarm_rl.backends.mjlab.runner import validate_checkpoint_contract
    from rebotarm_rl.contracts.policy import REACH_V1, REACH_V2
    path = tmp_path / 'copied.pt'
    for saved, other in ((REACH_V1, REACH_V2), (REACH_V2, REACH_V1)):
        torch.save({'infos': {'policy_contract': saved.to_dict()}}, path)
        validate_checkpoint_contract(path, saved)
        with pytest.raises(ValueError):
            validate_checkpoint_contract(path, other)
    manifest = tmp_path / 'run_manifest.json'
    manifest.write_text(json.dumps({'contract': REACH_V1.to_dict()}))
    with pytest.raises(ValueError):
        validate_checkpoint_contract(path, REACH_V2)
    manifest.unlink()
    torch.save({'infos': None}, path)
    with pytest.raises(ValueError):
        validate_checkpoint_contract(path, REACH_V2)
    with pytest.warns(UserWarning):
        validate_checkpoint_contract(path, REACH_V1)
