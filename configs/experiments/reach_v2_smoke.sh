#!/usr/bin/env bash
# 新实验使用V2；先激活独立训练环境，附加参数可覆盖默认配置。
set -euo pipefail
export MUJOCO_GL=egl
exec python -m mjlab.scripts.train RebotArm-Reach-Mjlab-V2 --env.scene.num-envs 8 --agent.max-iterations 1 --log-root runs/smoke-v2 "$@"
