import json

import pytest

from rebotarm_rl.contracts.artifacts import EvaluationReport, evaluation_provenance, write_json_atomic


def test_failed_attempt_is_retained_without_completed_output(tmp_path):
    output = tmp_path / 'eval.json'
    with pytest.raises(RuntimeError, match='GPU failed'):
        with EvaluationReport(output, {'levels': []}) as report:
            report.record['levels'].append({'amplitude_rad': .05})
            raise RuntimeError('GPU failed')
    assert not output.exists()
    failed = json.loads(output.with_suffix('.incomplete.json').read_text())
    assert failed['status'] == 'failed'
    assert failed['levels'] == [{'amplitude_rad': .05}]
    with pytest.raises(FileExistsError):
        with EvaluationReport(output, {}):
            pass


def test_completion_is_explicit_and_atomic(tmp_path):
    output = tmp_path / 'eval.json'
    with EvaluationReport(output, {'levels': []}) as report:
        assert not output.exists()
        report.complete()
    assert json.loads(output.read_text())['status'] == 'completed'
    assert not output.with_suffix('.incomplete.json').exists()
    with pytest.raises(ValueError):
        write_json_atomic(output, {'bad': float('nan')})
    assert json.loads(output.read_text())['status'] == 'completed'
    assert not list(tmp_path.glob('*.tmp'))
    early = tmp_path / 'early.json'
    with EvaluationReport(early, {}):
        pass
    assert not early.exists()
    assert json.loads(early.with_suffix('.incomplete.json').read_text())['status'] == 'incomplete'


def test_portable_provenance_without_git(tmp_path):
    package = tmp_path / 'installed/rebotarm_rl'
    (package / 'assets').mkdir(parents=True)
    (package / 'assets/model_manifest.json').write_text('{}')
    entrypoint = package / 'evaluation.py'
    entrypoint.write_text('')
    provenance = evaluation_provenance(package, entrypoint)
    assert provenance['git_commit'] is None and provenance['git_dirty'] is None
    assert provenance['source_kind'] == 'installed_package_without_git'
    assert provenance['source_sha256']['evaluation.py']
    assert provenance['model_manifest_sha256']
