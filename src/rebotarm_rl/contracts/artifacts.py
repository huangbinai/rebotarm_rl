"""可移植的实验来源记录；不初始化训练框架。"""
from datetime import datetime, timezone
import hashlib
import os
import json
from pathlib import Path
import subprocess
import sys


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_run_record(directory: Path, *, repository: Path, runtime: dict) -> None:
    """训练前在后端原生配置旁保存运行来源，配置、曲线复用训练框架。"""
    def git(*args):
        result = subprocess.run(["git", "-C", str(repository), *args],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    status = git("status", "--porcelain", "--untracked-files=all")
    commit = git("rev-parse", "HEAD")
    run_kind = os.environ.get("REBOTARM_RL_RUN_KIND", "train")
    if run_kind not in {"train", "smoke"}:
        raise ValueError("REBOTARM_RL_RUN_KIND必须为train或smoke")
    if run_kind == "train" and (not commit or status is None or status):
        raise ValueError("正式训练需要已提交且干净的Git工作区；请先提交代码。短验证使用REBOTARM_RL_RUN_KIND=smoke")
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "run_kind": run_kind,
        "git_dirty": bool(status) if status is not None else None,
        "command": sys.orig_argv, "python": sys.version,
        "environment": os.environ.get("REBOTARM_RL_ENVIRONMENT", sys.prefix),
        "runtime": runtime,
        "resolved_configs": ["params/env.yaml", "params/agent.yaml"],
    }
    (directory / "run_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
