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


def write_json_atomic(path: Path, value: dict) -> None:
    """Replace one JSON artifact only after successful serialization and flush."""
    import tempfile
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluation_provenance(package: Path, entrypoint: Path) -> dict:
    """Collect before GPU work; a wheel has explicit unknown Git provenance.

    Hash all installed Python sources plus the bundled model manifest so the
    identity remains useful when the source is not a Git checkout.
    """
    from importlib.metadata import PackageNotFoundError, version
    package = package.resolve()
    repository = next((p for p in package.parents if (p / '.git').exists()
                       and (p / 'src/rebotarm_rl').resolve() == package), None)
    def git(*args):
        if repository is None:
            return None
        try:
            result = subprocess.run(['git', '-C', str(repository), *args],
                                    capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    commit, status = git('rev-parse', 'HEAD'), git('status', '--porcelain')
    try:
        package_version = version('rebotarm-rl')
    except PackageNotFoundError:
        package_version = None
    sources = {str(p.relative_to(package)): file_hash(p) for p in sorted(package.rglob('*.py'))}
    return {
        'git_commit': commit, 'git_dirty': None if status is None else bool(status),
        'source_kind': 'git_checkout' if commit else 'installed_package_without_git',
        'package_version': package_version, 'source_sha256': sources,
        'model_manifest_sha256': file_hash(package / 'assets/model_manifest.json'),
        'evaluator_sha256': file_hash(entrypoint), 'command': sys.orig_argv,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }


class EvaluationReport:
    """Persist incomplete/failed evidence; publish completed JSON atomically.

    Completion is explicit. Exiting early can never produce a successful report.
    Existing output or interrupted attempts require a new output name.
    """

    def __init__(self, output: Path, record: dict):
        self.output = output
        self.pending = output.with_suffix('.incomplete.json')
        self.record = dict(record, status='incomplete')

    def __enter__(self):
        if self.output.exists() or self.pending.exists():
            raise FileExistsError(self.output)
        write_json_atomic(self.pending, self.record)
        return self

    def save_progress(self) -> None:
        write_json_atomic(self.pending, self.record)

    def complete(self) -> None:
        completed = dict(self.record, status='completed')
        write_json_atomic(self.output, completed)
        self.record = completed
        self.pending.unlink()

    def __exit__(self, exc_type, exc, traceback):
        if self.record['status'] != 'completed':
            if exc is not None:
                self.record.update(status='failed', error={'type': exc_type.__name__, 'message': str(exc)})
            self.save_progress()
        return False
