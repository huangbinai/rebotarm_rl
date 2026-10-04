"""仓库训练入口：读取命名实验，使用当前解释器启动 mjlab 原生 CLI。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def load_experiment(name: str) -> dict:
    """读取仓库内 TOML；只保存实验覆盖，不复制任务或 PPO 默认配置。"""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("实验名只能包含字母、数字、下划线和连字符")
    with (ROOT / "configs" / "experiments" / f"{name}.toml").open("rb") as stream:
        config = tomllib.load(stream)
    if set(config) != {"task", "run_kind", "seed", "args"}:
        raise ValueError("实验必须且只能包含 task、run_kind、seed、args")
    if not isinstance(config["task"], str) or config["task"].startswith("-"):
        raise ValueError("task 必须为任务名")
    if config["run_kind"] not in {"train", "smoke"}:
        raise ValueError("run_kind 必须为 train 或 smoke")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    if not isinstance(config["args"], dict):
        raise ValueError("args 必须为原生 CLI 参数表")
    reserved = {"agent.seed", "env.seed", "agent.run-name", "log-root"}
    for key, value in config["args"].items():
        if not re.fullmatch(r"(?:env|agent)\.[a-zA-Z0-9_.-]+", key) or key in reserved:
            raise ValueError(f"不支持或由入口管理的参数: {key}")
        if type(value) not in (str, int, float):
            raise ValueError(f"参数 {key} 只支持字符串或数值；复杂配置放在原生任务配置中")
    return config


def require_clean_repository() -> None:
    """在初始化 GPU 前检查正式训练来源；后端仍保留自己的检查。"""
    subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD"],
                   check=True, capture_output=True, text=True)
    result = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        raise ValueError("正式训练需要已提交且干净的 Git 工作区；请先提交代码")


def build_launch(name: str, config: dict, seed: int, environment: str | None) -> tuple[list[str], dict[str, str]]:
    """生成原生命令及隔离环境；不创建 GPU、日志或训练进程。"""
    if seed < 0:
        raise ValueError("seed 必须为非负整数")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH",
                "REBOTARM_MJLAB_SCENE", "REBOTARM_RL_ENVIRONMENT"):
        env.pop(key, None)
    env.update(MUJOCO_GL="egl", REBOTARM_RL_RUN_KIND=config["run_kind"])
    if environment is not None:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", environment):
            raise ValueError("环境标识只能包含字母、数字、下划线和连字符")
        snapshot = ROOT / "runs" / "environments" / f"{environment}.txt"
        if not snapshot.is_file() or snapshot.stat().st_size == 0:
            raise ValueError(f"依赖快照不存在或为空: {snapshot}")
        env["REBOTARM_RL_ENVIRONMENT"] = environment
    command = [sys.executable, "-m", "mjlab.scripts.train", config["task"]]
    for key, value in config["args"].items():
        command.extend([f"--{key}", str(value)])
    command.extend([
        "--agent.seed", str(seed), "--agent.run-name", f"{name}_seed{seed}",
        "--log-root", f"runs/{config['run_kind']}",
    ])
    return command, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="reach_baseline", help="configs/experiments 下的 TOML 名称")
    parser.add_argument("--seed", type=int, help="覆盖实验种子")
    parser.add_argument("--environment", help="runs/environments 下已有依赖快照名称，不含 .txt")
    parser.add_argument("--dry-run", action="store_true", help="只预览命令，不检查 Git 或启动训练")
    args = parser.parse_args()
    try:
        config = load_experiment(args.experiment)
        seed = config["seed"] if args.seed is None else args.seed
        command, env = build_launch(args.experiment, config, seed, args.environment)
        if not args.dry_run and config["run_kind"] == "train":
            require_clean_repository()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(f"实验: {args.experiment} | 类型: {config['run_kind']} | 种子: {seed}", flush=True)
    print(f"依赖环境: {args.environment or sys.prefix}", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    # 替换当前进程，使终端 Ctrl+C 和退出码直接由原生训练处理。
    os.chdir(ROOT)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
