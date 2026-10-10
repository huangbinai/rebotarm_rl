import json

import pytest

from rebotarm_rl.contracts.artifacts import file_hash
from rebotarm_rl.contracts.evaluation import RANDOM_START, validate_frozen_selection
from rebotarm_rl.contracts.policy import REACH_GRAVITY_FIXED


@pytest.fixture
def evidence(tmp_path):
    weight = tmp_path / 'model.pt'
    weight.write_bytes(b'frozen weight')
    settings = dict(target_seed=230000, initial_seed=230001, episodes=200,
                    steps=250, hold_samples=26, amplitudes_rad=[.05, .4])
    report = dict(protocol=RANDOM_START.name, role='diagnostic_validation',
                  checkpoint=str(weight), checkpoint_sha256=file_hash(weight),
                  policy_task=REACH_GRAVITY_FIXED.task_id, policy_contract=REACH_GRAVITY_FIXED.to_dict(),
                  conditions=dict(target_seed=130000, initial_seed=130001, episodes=1, steps=250,
                                  hold_samples=26, control_dt_s=.02, success_position_m=.01,
                                  success_orientation_rad=RANDOM_START.success_orientation_rad,
                                  target_radius_uniform_m=[.02, .06], joint_limit_margin_rad=.01,
                                  automatic_termination=False, target_refresh=False, deterministic_policy=True),
                  levels=[dict(amplitude_rad=a, summary={'episodes': 1}, results=[{}]) for a in [.05, .4]])
    report_path = tmp_path / 'validation.json'
    report_path.write_text(json.dumps(report))
    selection = dict(protocol=RANDOM_START.name, checkpoint=str(weight), checkpoint_sha256=file_hash(weight),
                     frozen_at='2026-10-11', decision_before_test={'minimum_tail_success_rate': .95},
                     test_settings=settings, validation_target_seeds=[130000], validation_initial_seeds=[130001],
                     validation_reports=[dict(path=str(report_path), sha256=file_hash(report_path))])
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(selection))
    return path, weight, settings, selection, report_path, report


def test_valid_legacy_selection_and_settings_binding(evidence):
    path, weight, settings, selection, report_path, report = evidence
    report.pop('role')
    report_path.write_text(json.dumps(report))
    selection['validation_reports'][0]['sha256'] = file_hash(report_path)
    path.write_text(json.dumps(selection))
    assert validate_frozen_selection(path, [weight], settings)['record'] == selection
    with pytest.raises(ValueError, match='settings'):
        validate_frozen_selection(path, [weight], dict(settings, episodes=100))
    weight.write_bytes(b'modified')
    with pytest.raises(ValueError, match='checkpoint'):
        validate_frozen_selection(path, [weight], settings)


@pytest.mark.parametrize('change', ['protocol', 'missing', 'hash', 'failed', 'seeds', 'reuse', 'conditions', 'coverage', 'weight'])
def test_invalid_validation_evidence_is_rejected(evidence, change):
    path, weight, settings, selection, report_path, report = evidence
    if change == 'protocol':
        selection['protocol'] = 'other-v1'
    elif change == 'missing':
        report_path.unlink()
    elif change == 'hash':
        report_path.write_text('{}')
    elif change == 'failed':
        report['status'] = 'failed'
    elif change == 'seeds':
        selection['validation_initial_seeds'] = [999]
    elif change == 'reuse':
        settings['initial_seed'] = 130001
    elif change == 'conditions':
        report['conditions']['target_refresh'] = True
    elif change == 'coverage':
        report['levels'].pop()
    elif change == 'weight':
        report['checkpoint_sha256'] = 'modified'
    if change in {'failed', 'conditions', 'coverage', 'weight'}:
        report_path.write_text(json.dumps(report))
        selection['validation_reports'][0]['sha256'] = file_hash(report_path)
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        validate_frozen_selection(path, [weight], settings)
