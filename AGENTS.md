# Agent instructions

Read README.md, docs/policy_contract.md, docs/cloud_training.md and docs/MIGRATION.md first.
This is an independent training repository, not a ROS package. Never add imports from
rclpy, rebotarm_simulation, rebotarm_preview or rebotarmcontroller.
Do not connect to physical hardware. Torque policies are not position trajectories.
Preserve the mjlab task ID and action/observation semantics unless explicitly changing
the task; version such changes and document checkpoint incompatibilities.
Models come from the pinned model_manifest.json; do not depend on sibling directories.
No credentials/checkpoints/videos/virtual environments in Git.
Isaac Lab is planned, not implemented; keep its eventual runtime separate from mjlab.
Checks: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -m pytest tests -q; python -m compileall rebotarm_rl -q;
git diff --check. Changes to task dynamics need a GPU smoke test and paired evaluation.
Report smoke tests separately from convergence, generalization and hardware acceptance.
