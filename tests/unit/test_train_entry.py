"""训练入口的来源检查、环境隔离及无GPU预览。"""
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/train.py"
API = runpy.run_path(str(ENTRY))


def test_isolated_launch_and_seed_override(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/other/workspace")
    monkeypatch.setenv("REBOTARM_MJLAB_SCENE", "/other/model.xml")
    monkeypatch.setenv("REBOTARM_RL_RUN_KIND", "smoke")
    monkeypatch.setenv("REBOTARM_RL_ENVIRONMENT", "stale")
    monkeypatch.setenv("REBOTARM_RL_VALIDATION", "stale")
    cfg = API["load_experiment"]("reach_gravity_fixed")
    command, env = API["build_launch"]("reach_gravity_fixed", cfg, 19, None)
    assert command[:4] == [sys.executable, "-m", "mjlab.scripts.train", "RebotArm-Reach-GravityComp-FixedPenalties-Mjlab"]
    assert command[command.index("--agent.seed") + 1] == "19"
    assert command[command.index("--agent.run-name") + 1] == "reach_gravity_fixed_seed19"
    assert command[command.index("--log-root") + 1] == "runs/train"
    assert env["REBOTARM_RL_RUN_KIND"] == "train"
    assert not {"PYTHONPATH", "REBOTARM_MJLAB_SCENE", "REBOTARM_RL_ENVIRONMENT", "REBOTARM_RL_VALIDATION"} & env.keys()
    assert os.environ["PYTHONPATH"] == "/other/workspace"
    with pytest.raises(ValueError, match="快照"):
        API["build_launch"]("reach_gravity_fixed", cfg, 7, "missing-test-snapshot")
    with pytest.raises(ValueError):
        API["load_experiment"]("../reach_gravity_fixed")


def test_preview_from_other_directory(tmp_path):
    result = subprocess.run([sys.executable, str(ENTRY), "--experiment", "reach_gravity_fixed_smoke", "--dry-run"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "runs/smoke" in result.stdout
    assert "--agent.max-iterations 2" in result.stdout
    assert "Warp" not in result.stdout
    assert not list(tmp_path.iterdir())


def test_formal_rejects_untracked_files_before_launch(tmp_path, monkeypatch):
    check = API["require_clean_repository"]
    monkeypatch.setitem(check.__globals__, "ROOT", tmp_path)
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "init")
    check()
    (tmp_path / "experiment.toml").write_text("seed = 7\n")
    with pytest.raises(ValueError, match="干净"):
        check()


def test_default_preview_uses_current_task_and_preserves_seed():
    result = subprocess.run([sys.executable, str(ENTRY), "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "RebotArm-Reach-GravityComp-FixedPenalties-Mjlab" in result.stdout
    assert "--agent.seed 42" in result.stdout


@pytest.mark.parametrize("extra, expected", [([], 7), (["--seed", "19"], 19), (["--seed", "0"], 0)])
def test_cli_seed_overrides_history_toml_only_when_explicit(extra, expected):
    result = subprocess.run(
        [sys.executable, str(ENTRY), "--experiment", "history/reach_gravity_fixed_epochs4",
         "--dry-run", *extra], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"--agent.seed {expected}" in result.stdout
    assert f"--agent.run-name reach_gravity_fixed_epochs4_seed{expected}" in result.stdout


def test_history_configs_and_safe_names():
    for path in (ROOT / "configs/experiments/history").glob("*.toml"):
        name = "history/" + path.stem
        cfg = API["load_experiment"](name)
        command, _ = API["build_launch"](name, cfg, 17, None)
        assert command[command.index("--agent.run-name") + 1] == path.stem + "_seed17"
    for name in ("../reach_gravity_fixed", "history/../reach_gravity_fixed", "/tmp/config", "history/nested/config"):
        with pytest.raises(ValueError):
            API["load_experiment"](name)
