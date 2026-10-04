"""验证随包模型的完整性与自定义场景入口。"""
import hashlib
import pytest
from rebotarm_rl.assets import resources


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "reach_scene.xml").write_text("<mujoco/>")
    (source / "robot.xml").write_text("<mujoco model='robot'/>")
    spec = {"commit": "test", "files": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
    }}
    monkeypatch.setattr(resources, "manifest", lambda: spec)
    monkeypatch.setattr(resources, "model_directory", lambda: source)
    monkeypatch.delenv("REBOTARM_MJLAB_SCENE", raising=False)
    return source


def test_bundled_model_is_verified(bundle):
    assert resources.reach_scene_path() == bundle / "reach_scene.xml"
    (bundle / "robot.xml").write_text("corrupt")
    with pytest.raises(FileNotFoundError, match="modified"):
        resources.reach_scene_path()


def test_missing_bundle_is_rejected(bundle):
    (bundle / "reach_scene.xml").unlink()
    with pytest.raises(FileNotFoundError, match="Bundled model"):
        resources.reach_scene_path()


def test_packaged_model_ignores_external_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("REBOTARM_MJLAB_SCENE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty-cache"))
    scene = resources.reach_scene_path()
    assert scene.parent.name == "rebotarm"
    assert scene.is_file()
    assert not (tmp_path / "empty-cache").exists()


def test_explicit_custom_scene_is_not_silently_replaced(bundle, monkeypatch):
    custom = bundle / "reach_scene.xml"
    monkeypatch.setenv("REBOTARM_MJLAB_SCENE", str(custom))
    assert resources.reach_scene_path() == custom
    custom.unlink()
    with pytest.raises(FileNotFoundError, match="Configured"):
        resources.reach_scene_path()
