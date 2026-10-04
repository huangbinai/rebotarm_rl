#!/usr/bin/env bash
# Activate the mjlab environment first; appended flags explicitly override defaults.
set -euo pipefail
export MUJOCO_GL=egl
exec python -m mjlab.scripts.train RebotArm-Reach-Mjlab --env.scene.num-envs 8 --agent.max-iterations 1 --log-root runs/smoke "$@"
