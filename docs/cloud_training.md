# Cloud training

1. Provision a Linux GPU machine compatible with the pinned CUDA profile.
2. Clone this repository at a specific commit; create its own Python 3.12 environment.
3. Install requirements/mjlab-cu130.txt and this project; fetch the pinned model.
4. Check CUDA availability and task registration. Run a small headless smoke job.
5. Run longer jobs via your scheduler or terminal session manager; upload checkpoints
   and logs to persistent storage rather than the ephemeral instance disk alone.
6. Record git revision, model_manifest.json, pip freeze, GPU/driver, seed, command and
   evaluation metrics with every experiment. Retain normalization and model config.
7. Evaluate against fixed targets and seeds before publishing a policy artifact.

No cloud credentials are stored here. Configure storage access through the provider's
identity/secret mechanism. Cloud machines do not need to clone/build robotarm_ros2:
the asset fetcher downloads immutable model resources directly.

Example smoke (single shell command):
MUJOCO_GL=egl python -m mjlab.scripts.train RebotArm-Reach-Mjlab --env.scene.num-envs 8 --agent.max-iterations 1 --log-root runs/smoke

The migration environment snapshot is diagnostic evidence, not a cross-platform lock.
The pinned CUDA requirements are the installation entrypoint. For exact deployment,
build and publish a tested container and pin its digest.

Isaac Lab: create an external task project when implementation begins. Pin a compatible
Isaac Lab/Isaac Sim/container version and use a separate environment. Reuse the robot
parameter provenance and policy contract; do not assume identical physics or interchangeable
checkpoints. No Isaac Lab task or cloud deployment has been implemented by this migration.


## Structured experiment artifacts
The existing mjlab train command uses the registered RecordedRunner automatically.
Each run includes run_manifest.json, model_manifest.json, dependencies.txt,
compiled_model.mjb and native params/env.yaml + params/agent.yaml.
Keep these alongside checkpoints when uploading to persistent storage.
See development.md for dirty-worktree, resume and legacy-checkpoint rules.
