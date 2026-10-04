# rebotarm_rl

Independent reinforcement-learning project for reBotArm. Current implementation:
mjlab + MuJoCo Warp + RSL-RL Reach training and CPU/GPU paired evaluation.
Isaac Lab is a planned backend, not implemented yet.
No ROS workspace, ROS runtime, motor SDK or real-arm connection is required.

## Quick start (Ubuntu, Python 3.12, NVIDIA GPU)

The initial CUDA profile is pinned in requirements/mjlab-cu130.txt. It retains
the previously verified CUDA 13 profile; check driver compatibility before
deploying to another machine. Isaac Lab must use a separate future environment.

~~~bash
git clone https://github.com/huangbinai/rebotarm_rl.git
cd rebotarm_rl
python3.12 -m venv .venv
source .venv/bin/activate
# Keep ROS Python paths out of the independent training environment.
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
python -m pip install -r requirements/mjlab-cu130.txt
python -m pip install -e '.[test]'
python -m rebotarm_rl.resources
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
python -m mjlab.scripts.list_envs
MUJOCO_GL=egl python -m mjlab.scripts.train RebotArm-Reach-Mjlab --env.scene.num-envs 128 --agent.max-iterations 1000 --log-root runs/mjlab
~~~

The model fetcher downloads only files named in the packaged manifest, from an
immutable robotarm_ros2 commit, verifying SHA-256 for every file. Cache:
$XDG_CACHE_HOME/rebotarm_rl/models/<commit> (default ~/.cache).
Training performs no implicit downloads and rejects missing or modified baseline files.
The ROS repository remains the model source of truth. Update the manifest deliberately
when adopting a new model; do not hand-edit downloaded cache files.

Offline import of a matching model bundle is also supported:

~~~bash
python -m rebotarm_rl.resources --source /path/to/model-bundle
~~~

Custom experiments may explicitly set REBOTARM_MJLAB_SCENE=/path/to/reach_scene.xml.
Keep XML includes and meshes together. Custom models bypass baseline hash matching
and must have their own provenance recorded in experiment results.

## Playback and evaluation

~~~bash
python -m mjlab.scripts.play RebotArm-Reach-Mjlab --checkpoint-file /path/to/model.pt --env.scene.num-envs 1
MUJOCO_GL=egl python -m rebotarm_rl.evaluation.paired_eval --checkpoint /path/to/model.pt --episodes 100 --steps 250 --seed 20000 --output runs/paired_eval.json
~~~

Short training proves pipeline operation, not convergence or hardware readiness.
Current actions are six joint torques, not ROS position trajectories.
See [the interface contract](docs/policy_contract.md) before deployment.

## Layout and collaboration

- src/rebotarm_rl/backends/mjlab/tasks/: Reach observations, actions, rewards and termination.
- src/rebotarm_rl/training/rsl_rl/: PPO/network configuration; RSL-RL owns algorithm implementation.
- src/rebotarm_rl/backends/mjlab/evaluation/: fixed-target CPU MuJoCo / GPU Warp comparisons.
- src/rebotarm_rl/assets/resources.py + model_manifest.json: independently fetched pinned assets.
- requirements/: CUDA installation profile and migration environment snapshot.
- tests/: portable contracts and model-resource integrity tests.
- docs/: deployment boundaries, cloud workflow and migration provenance.

Run PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -m pytest tests -q and python -m compileall src/rebotarm_rl -q.
Use branches and reviewed pull requests. CPU CI checks contracts/resources;
GPU training and policy evaluation require a GPU runner or cloud job.
Do not commit credentials, checkpoints, caches or videos. Store training outputs
in an artifact/object store with code commit, model manifest, dependencies and seed.

This repository was extracted with git subtree split from robotarm_ros2, preserving
the history of its former rebotarm_rl/ subtree. Add this directory as a separate
Codex project; AGENTS.md describes its boundaries.

Architecture: [结构规范](docs/architecture.md), [开发规范](docs/development.md).
Named smoke commands: bash configs/experiments/reach_smoke.sh; evaluation uses configs/evaluation/reach_smoke.sh.
The core package has no mandatory simulator dependency; install the pinned GPU requirements or the mjlab extra for training.
