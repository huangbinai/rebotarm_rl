# Reach v1 interface boundary

Task ID: RebotArm-Reach-Mjlab. Migration preserves task/controller semantics.
The executable task configuration in rebotarm_rl/tasks/reach.py is authoritative.

- Six effort actions address joint1..joint6, scale 1.0, torque units N m.
  Model actuator limits and runner clipping remain authoritative.
- Physics timestep: 0.002 s; decimation: 10; policy action period: 0.02 s.
- Actor observations concatenate relative joint positions, relative joint velocities,
  target position error and orientation error, in the order defined in reach.py.
  Both actor and critic consume the actor observation group.
- MuJoCo/mjlab native pose quaternions use wxyz; ROS quaternion messages use xyzw.
  Do not copy quaternion arrays directly into ROS messages.
- Success threshold: position error < 0.01 m and orientation error < 0.05236 rad.
- Checkpoint deployment must retain observation preprocessing/normalization,
  action scaling, joint mapping, timestep, target frame and model provenance.
- No ROS publisher, Action client or hardware driver is part of this project.
  A future local deployment adapter must validate freshness, units, limits and
  supported hardware control mode before enabling any physical execution.
- A six-torque vector cannot be sent as six positions to FollowJointTrajectory.
- CPU/GPU comparisons use the mjlab-compiled model. They do not prove equivalent
  behavior to ROS position-control simulation or physical hardware.
