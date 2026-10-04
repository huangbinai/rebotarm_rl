"""可移植的实验来源记录；不初始化训练框架。"""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys
from .policy import REACH_V1, PolicyContract


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_run_record(directory: Path, *, repository: Path, model: dict,
                     runtime: dict, contract: PolicyContract = REACH_V1) -> None:
    """训练前在后端原生配置旁保存代码、模型、依赖与运行来源。"""
    directory.mkdir(parents=True, exist_ok=True)
    def git(*args):
        result = subprocess.run(["git", "-C", str(repository), *args],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    status = git("status", "--porcelain")
    record = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "command": sys.argv, "python": sys.version,
        "contract": contract.to_dict(), "runtime": runtime,
        "resolved_configs": ["params/env.yaml", "params/agent.yaml"],
    }
    (directory / "run_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    (directory / "model_manifest.json").write_text(json.dumps(model, indent=2) + "\n")
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()})
    (directory / "dependencies.txt").write_text("\n".join(packages) + "\n")
