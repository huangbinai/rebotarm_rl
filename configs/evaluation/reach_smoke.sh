#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "用法: bash configs/evaluation/reach_smoke.sh 权重路径 输出路径 [附加参数]" >&2
  exit 2
fi
checkpoint="$1"
output="$2"
shift 2
export MUJOCO_GL=egl
exec python -m rebotarm_rl.evaluation.paired_eval --checkpoint "$checkpoint" --output "$output" --episodes 2 --steps 20 --seed 20000 "$@"
