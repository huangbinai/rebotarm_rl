# Migration provenance

Source repository: https://github.com/huangbinai/robotarm_ros2
Source commit: 00065fb9e47730a6905af5d6268082f8c6d02db2
Subtree: rebotarm_rl/
Extraction: git subtree split --prefix=rebotarm_rl
The initial subtree commit history is preserved. Earlier code history before the
subtree existed remains available in the source repository.

Assets: model_manifest.json pins the source commit and hashes of required XML/STL
files. Cache is independent of the ROS checkout. No model files are duplicated
as manually maintained sources in this repository.

The ROS environment remains in its existing location for existing launch commands.
The RL project gets its own .venv; this directory is not committed.
Validation results are appended after migration checks.

## Completed checks so far

- 8 portable contract/resource tests passed.
- Model fetched from GitHub at pinned commit; all 14 file hashes matched.
- Built wheel and installed it into a clean temporary venv without ROS dependencies;
  resource lookup succeeded outside both repositories.
- New task code completed 8-env/1-iteration GPU precheck and 2-episode/20-step
  paired evaluation using the existing runtime. This is not the independent-env
  acceptance; that check follows after dependencies finish downloading.
- No physical hardware or remote cloud instance was operated.

Existing local runs/mjlab and runs/rl-package-smoke were copied into
runs/imported-robotarm-ros2/. Original outputs remain in the ROS checkout to avoid
breaking historical evidence paths. Both copies remain ignored by Git.

## Independent-environment acceptance (2026-10-04)

- Fresh .venv: Python 3.12, torch 2.14.1+cu130, MuJoCo 3.11.0,
  NVIDIA GeForce RTX 4060 Laptop GPU. No rclpy or rebotarm_simulation installed.
- Ran from /tmp with PYTHONPATH/AMENT_PREFIX_PATH removed: task registration passed,
  RebotArm-Reach-Mjlab completed 8 environments / 1 PPO iteration and wrote model_0.pt.
- New-environment paired evaluation: 2 episodes / 20 steps passed, all outputs finite.
  Both CPU and GPU final success counts were 0. This is pipeline validation, not
  a trained policy or convergence claim.
- 8 independent contract/resource tests passed; pip dependency check passed.
- A ROS-inherited PYTHONPATH initially caused pytest to autoload ROS plugins;
  cleared it and verified tests in the clean environment. README documents cleanup.
- mjlab 1.6.0 does not declare a cu130 extra; removed that ignored extra from the
  migrated requirements. Explicit torch CUDA wheel selection remains pinned.
- ROS regression after removal: 757 passed / 37 skipped; layering 25 passed;
  physics tests 96 passed; simulation colcon rebuild and compileall passed.
  The full-suite count changed because training implementation tests moved here.
- Original ROS training plugin was uninstalled without removing MuJoCo dependencies.
- Tests and the actual GPU smoke were local. GitHub CI and cloud jobs have not been
  claimed as executed. Isaac Lab is not yet implemented.

Detailed local logs and checkpoints are excluded from Git. Training environment
versions are recorded in requirements/migration-snapshot.txt.
