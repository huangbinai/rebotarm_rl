"""隔离环境后启动Reach可视化回放；不训练、不连接硬件。"""
import os
from pathlib import Path
import sys


def main() -> None:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH",
                "REBOTARM_MJLAB_SCENE", "REBOTARM_RL_VALIDATION"):
        env.pop(key, None)
    env["MUJOCO_GL"] = "glfw"
    os.chdir(Path(__file__).resolve().parents[1])
    os.execve(sys.executable, [sys.executable, "-m",
              "rebotarm_rl.backends.mjlab.play", *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
