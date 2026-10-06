"""显式注册mjlab插件；顶层包导入保持轻量。"""
from mjlab.tasks.registry import register_mjlab_task
from rebotarm_rl.backends.mjlab.tasks.reach.config import make_env_cfg
from rebotarm_rl.training.rsl_rl.reach_ppo import runner_cfg

from .runner import RecordedRunner
from .tasks.reach.aligned import make_aligned_env_cfg
from rebotarm_rl.training.rsl_rl.reach_ppo import aligned_runner_cfg
from rebotarm_rl.contracts.policy import REACH_ALIGNED

register_mjlab_task(
    task_id="RebotArm-Reach-Mjlab",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True, num_envs=1),
    rl_cfg=runner_cfg(),
    runner_cls=RecordedRunner,
)

register_mjlab_task(
    task_id=REACH_ALIGNED.task_id,
    env_cfg=make_aligned_env_cfg(),
    play_env_cfg=make_aligned_env_cfg(play=True, num_envs=1),
    rl_cfg=aligned_runner_cfg(), runner_cls=RecordedRunner,
)
