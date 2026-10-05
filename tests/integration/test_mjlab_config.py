"""检查生效环境配置与公开契约一致，禁止时序静默漂移。"""
import pytest

def test_task_contract_matches_effective_config():
    pytest.importorskip("mjlab")
    from rebotarm_rl.backends.mjlab.tasks.reach.config import make_env_cfg
    from rebotarm_rl.backends.mjlab.validation import validate_config
    from rebotarm_rl.backends.mjlab.tasks.reach.observations import orientation_error
    from rebotarm_rl.contracts.policy import REACH_V1
    cfg = make_env_cfg()
    assert validate_config(cfg) == REACH_V1
    assert cfg.observations['actor'].terms['orientation_error'].func is orientation_error
    cfg.actions['joint_position'].scale = 2.0
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = make_env_cfg()
    cfg.decimation = 5
    with pytest.raises(ValueError):
        validate_config(cfg)
