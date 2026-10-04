"""显式注册mjlab插件；顶层包导入保持轻量。"""
from mjlab.tasks.registry import register_mjlab_task
from rebotarm_rl.backends.mjlab.tasks.reach.config import make_env_cfg
from rebotarm_rl.training.rsl_rl.reach_ppo import runner_cfg

from .runner import RecordedRunner

register_mjlab_task(
    task_id="RebotArm-Reach-Mjlab",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True, num_envs=1),
    rl_cfg=runner_cfg(),
    runner_cls=RecordedRunner,
)
