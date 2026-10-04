import json
import pytest
from rebotarm_rl.contracts.policy import REACH_V1
from rebotarm_rl.contracts.artifacts import write_run_record


def test_shape_and_serialized_compatibility():
    REACH_V1.validate_shapes(18, 6)
    REACH_V1.require_compatible(json.loads(json.dumps(REACH_V1.to_dict())))
    with pytest.raises(ValueError):
        REACH_V1.validate_shapes(22, 6)
    incompatible = REACH_V1.to_dict() | {"action_type": "joint_position"}
    with pytest.raises(ValueError):
        REACH_V1.require_compatible(incompatible)


def test_record_without_git_is_explicit(tmp_path):
    write_run_record(tmp_path / "run", repository=tmp_path,
                     model={"commit": "abc"}, runtime={"seed": 7})
    data = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert data["git_commit"] is None
    assert data["git_dirty"] is None
    assert data["runtime"]["seed"] == 7
    assert (tmp_path / "run/dependencies.txt").is_file()
