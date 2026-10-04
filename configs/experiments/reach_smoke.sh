#!/usr/bin/env bash
# 先激活独立mjlab环境；末尾附加参数可显式覆盖默认值。
set -euo pipefail
export MUJOCO_GL=egl
exec python -m mjlab.scripts.train RebotArm-Reach-Mjlab --env.scene.num-envs 8 --agent.max-iterations 1 --log-root runs/smoke "$@"
