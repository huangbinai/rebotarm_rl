import pytest

def test_task_contract_matches_effective_config():
    pytest.importorskip("mjlab")
    from rebotarm_rl.backends.mjlab.tasks.reach.config import make_env_cfg
    from rebotarm_rl.backends.mjlab.validation import validate_config
    cfg = make_env_cfg()
    validate_config(cfg)
    cfg.decimation = 5
    with pytest.raises(ValueError):
        validate_config(cfg)
