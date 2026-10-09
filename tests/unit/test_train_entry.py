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
    cfg = API["load_experiment"]("reach_baseline")
    command, env = API["build_launch"]("reach_baseline", cfg, 19, None)
    assert command[:4] == [sys.executable, "-m", "mjlab.scripts.train", "RebotArm-Reach-Mjlab"]
    assert command[command.index("--agent.seed") + 1] == "19"
    assert command[command.index("--agent.run-name") + 1] == "reach_baseline_seed19"
    assert command[command.index("--log-root") + 1] == "runs/train"
    assert env["REBOTARM_RL_RUN_KIND"] == "train"
    assert not {"PYTHONPATH", "REBOTARM_MJLAB_SCENE", "REBOTARM_RL_ENVIRONMENT", "REBOTARM_RL_VALIDATION"} & env.keys()
    assert os.environ["PYTHONPATH"] == "/other/workspace"
    with pytest.raises(ValueError, match="快照"):
        API["build_launch"]("reach_baseline", cfg, 7, "missing-test-snapshot")
    with pytest.raises(ValueError):
        API["load_experiment"]("../reach_baseline")


def test_preview_from_other_directory(tmp_path):
    result = subprocess.run([sys.executable, str(ENTRY), "--experiment", "reach_smoke", "--dry-run"],
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
