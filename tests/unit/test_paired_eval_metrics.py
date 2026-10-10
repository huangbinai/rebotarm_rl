from rebotarm_rl.evaluation.metrics import success_summary


def test_success_summary_requires_continuous_hold_and_tail():
    result = success_summary([False, True, True, False, True, True, True], 3)
    assert result['initial_success'] is False
    assert result['final_success'] is True
    assert result['first_success_step'] == 1
    assert result['longest_success_hold_steps'] == 3
    assert result['hold_success'] is True
    assert result['tail_success'] is True


def test_success_summary_rejects_short_trajectory():
    try:
        success_summary([], 1)
    except ValueError:
        pass
    else:
        raise AssertionError('empty trajectory should be rejected')
