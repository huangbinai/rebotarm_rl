"""Reach PPO 配置；算法实现使用 mjlab / RSL-RL。"""
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

def runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(128, 128), activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=3.0e-4,
            num_learning_epochs=5,
            num_mini_batches=4,
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
        ),
        # Actor和Critic各自建网；当前任务没有特权状态，使用同一公开观测组。
        obs_groups={"actor": ("actor",), "critic": ("actor",)},
        experiment_name="rebotarm_mjlab_reach",
        logger="tensorboard",
        num_steps_per_env=24,
        max_iterations=1000,
        save_interval=100,
    )


def aligned_runner_cfg():
    cfg = runner_cfg()
    cfg.actor.hidden_dims = (64, 64)
    cfg.critic.hidden_dims = (64, 64)
    cfg.actor.distribution_cfg['init_std'] = 1.0
    cfg.algorithm.entropy_coef = .001
    cfg.algorithm.learning_rate = .001
    cfg.algorithm.num_learning_epochs = 8
    cfg.experiment_name = 'rebotarm_reach'
    cfg.save_interval = 50
    return cfg
