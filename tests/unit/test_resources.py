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
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("REBOTARM_MJLAB_SCENE", raising=False)
    return source


def test_local_bundle_is_copied_and_verified_without_source_dependency(bundle):
    scene = resources.fetch_model(bundle)
    (bundle / "reach_scene.xml").unlink()
    assert resources.reach_scene_path() == scene
    scene.write_text("corrupt")
    with pytest.raises(FileNotFoundError, match="modified"):
        resources.reach_scene_path()


def test_bad_input_is_rejected(bundle):
    (bundle / "robot.xml").write_text("bad")
    with pytest.raises(ValueError, match="checksum"):
        resources.fetch_model(bundle)
    with pytest.raises(FileNotFoundError):
        resources.reach_scene_path()


def test_missing_bundle_requires_explicit_fetch(bundle):
    with pytest.raises(FileNotFoundError, match="fetch-model"):
        resources.reach_scene_path()


def test_explicit_custom_scene_is_not_silently_replaced(bundle, monkeypatch):
    custom = bundle / "reach_scene.xml"
    monkeypatch.setenv("REBOTARM_MJLAB_SCENE", str(custom))
    assert resources.reach_scene_path() == custom
    custom.unlink()
    with pytest.raises(FileNotFoundError, match="Configured"):
        resources.reach_scene_path()
