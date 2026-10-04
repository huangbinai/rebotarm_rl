"""验证契约序列化、维度检查与缺少Git来源时的明确记录。"""
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


def test_record_without_git_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("REBOTARM_RL_RUN_KIND", "smoke")
    write_run_record(tmp_path / "run", repository=tmp_path,
                     runtime={"seed": 7})
    data = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert data["git_commit"] is None
    assert data["git_dirty"] is None
    assert data["runtime"]["seed"] == 7
    assert data["schema_version"] == 2
    assert data["command"]
    assert data["environment"]
    assert "contract" not in data
    assert not (tmp_path / "run/dependencies.txt").exists()
    assert not (tmp_path / "run/model_manifest.json").exists()


def test_formal_requires_clean_commit(tmp_path, monkeypatch):
    import subprocess
    def git(*args):
        subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True)
    monkeypatch.delenv('REBOTARM_RL_RUN_KIND', raising=False)
    run = tmp_path.parent / (tmp_path.name + '-output')
    with pytest.raises(ValueError, match='干净'):
        write_run_record(run, repository=tmp_path, runtime={'seed': 42})
    git('init')
    git('config', 'user.email', 'test@example.com')
    git('config', 'user.name', 'Test')
    source = tmp_path / 'task.py'
    source.write_text('VALUE = 1\n')
    git('add', 'task.py')
    git('commit', '-m', 'baseline')
    write_run_record(run, repository=tmp_path, runtime={'seed': 42})
    record = json.loads((run / 'run_manifest.json').read_text())
    assert record['git_commit'] and record['git_dirty'] is False
    assert record['run_kind'] == 'train'
    source.write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='干净'):
        write_run_record(run, repository=tmp_path, runtime={'seed': 42})
    git('checkout', '--', 'task.py')
    (tmp_path / 'new.py').write_text('')
    with pytest.raises(ValueError, match='干净'):
        write_run_record(run, repository=tmp_path, runtime={'seed': 42})
    monkeypatch.setenv('REBOTARM_RL_RUN_KIND', 'smoke')
    write_run_record(run, repository=tmp_path, runtime={'seed': 42})
    assert not (run / 'git').exists()
