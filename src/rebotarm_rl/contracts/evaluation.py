"""Versioned evaluation conditions and frozen-test evidence (standard library only)."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .artifacts import file_hash
from .policy import REACH_GRAVITY_FIXED


@dataclass(frozen=True)
class RandomStartProtocol:
    """Task 2 evaluation defaults; does not change the policy training contract."""

    name: str = 'task2-random-start-diagnostic-v1'
    success_position_m: float = .01
    success_orientation_rad: float = math.radians(3)
    joint_limit_margin_rad: float = .01
    target_radius_m: tuple[float, float] = (.02, .06)
    steps: int = 250
    hold_samples: int = 26
    target_seed: int = 130000
    initial_seed: int = 130001


RANDOM_START = RandomStartProtocol()


@dataclass(frozen=True)
class TargetRangeProtocol:
    """Task 3 validation bands; pose/hold metrics reuse the Task 2 definition."""

    name: str = 'task3-target-range-diagnostic-v1'
    bands_m: tuple[tuple[float, float], ...] = ((.02, .06), (.06, .08), (.08, .10), (.10, .12), (.12, .16), (.16, .20))
    initial_amplitude_rad: float = .4
    target_seed: int = 330000
    initial_seed: int = 330001
    reserved_test_target_seed: int = 430000
    reserved_test_initial_seed: int = 430001
    minimum_tail_success_rate: float = .95


TARGET_RANGE = TargetRangeProtocol()


def validate_settings(settings: dict) -> None:
    """Validate exact Task 2 test settings, including sample span and RNG seeds."""
    expected = {'amplitudes_rad', 'episodes', 'steps', 'hold_samples', 'target_seed', 'initial_seed'}
    if set(settings) != expected:
        raise ValueError('Invalid test settings fields')
    for name in expected - {'amplitudes_rad'}:
        minimum = 0 if name.endswith('seed') else 1
        if type(settings[name]) is not int or settings[name] < minimum:
            raise ValueError(f'Invalid test settings: {name}')
    if not 2 <= settings['hold_samples'] <= settings['steps'] + 1:
        raise ValueError('Invalid test settings: hold_samples')
    amplitudes = settings['amplitudes_rad']
    if (not isinstance(amplitudes, list) or not amplitudes
            or any(type(a) not in (int, float) or not math.isfinite(a) or a < 0 for a in amplitudes)
            or len(set(amplitudes)) != len(amplitudes)):
        raise ValueError('Invalid test settings: amplitudes_rad')


def _read_hashed_record(path: Path, digest: str) -> dict:
    if not path.is_file() or file_hash(path) != digest:
        raise ValueError(f'Missing or modified evidence: {path}')
    return json.loads(path.read_text())


def validate_frozen_selection(path: Path, checkpoints: list[Path], settings: dict) -> dict:
    """Bind a fresh test to actual validation files, weights and immutable settings.

    Legacy v1 reports without status are supported; explicit failed/incomplete
    reports are never selection evidence. Paths in old manifests remain valid.
    """
    validate_settings(settings)
    selection = json.loads(path.read_text())
    if selection.get('protocol') != RANDOM_START.name:
        raise ValueError('Incompatible frozen selection protocol')
    if (len(checkpoints) != 1 or not checkpoints[0].is_file()
            or selection.get('checkpoint_sha256') != file_hash(checkpoints[0])
            or Path(selection.get('checkpoint', '')).resolve() != checkpoints[0].resolve()
            or selection.get('test_settings') != settings):
        raise ValueError('Frozen checkpoint or test settings do not match')
    if not selection.get('decision_before_test') or not selection.get('frozen_at'):
        raise ValueError('Missing frozen decision or timestamp')
    reports = selection.get('validation_reports')
    if not isinstance(reports, list) or not reports:
        raise ValueError('Missing validation reports')
    target_seeds, initial_seeds, selected_amplitudes = set(), set(), set()
    for reference in reports:
        report = _read_hashed_record(Path(reference['path']), reference['sha256'])
        if (report.get('protocol') != RANDOM_START.name
                or report.get('role', 'diagnostic_validation') != 'diagnostic_validation'
                or report.get('status', 'completed') != 'completed'):
            raise ValueError('Invalid validation report protocol, role or status')
        weight = Path(report['checkpoint'])
        if not weight.is_file() or file_hash(weight) != report['checkpoint_sha256']:
            raise ValueError('Validation checkpoint is missing or modified')
        if report['policy_task'] != REACH_GRAVITY_FIXED.task_id:
            raise ValueError('Validation policy task mismatch')
        REACH_GRAVITY_FIXED.require_compatible(report['policy_contract'])
        c = report['conditions']
        if (c['steps'] != settings['steps'] or c['hold_samples'] != settings['hold_samples']
                or c['control_dt_s'] != .02
                or c['success_position_m'] != RANDOM_START.success_position_m
                or c['success_orientation_rad'] != RANDOM_START.success_orientation_rad
                or c['target_radius_uniform_m'] != list(RANDOM_START.target_radius_m)
                or c['joint_limit_margin_rad'] != RANDOM_START.joint_limit_margin_rad
                or c['automatic_termination'] or c['target_refresh']
                or not c['deterministic_policy']):
            raise ValueError('Validation conditions mismatch')
        target_seeds.add(c['target_seed'])
        initial_seeds.add(c['initial_seed'])
        levels = report['levels']
        if not levels or any(len(level['results']) != c['episodes']
                             or level['summary']['episodes'] != c['episodes'] for level in levels):
            raise ValueError('Incomplete validation episodes')
        if report['checkpoint_sha256'] == selection['checkpoint_sha256']:
            selected_amplitudes.update(level['amplitude_rad'] for level in levels)
    if (target_seeds != set(selection['validation_target_seeds'])
            or initial_seeds != set(selection['validation_initial_seeds'])):
        raise ValueError('Declared validation seeds do not match evidence')
    if (settings['target_seed'] in target_seeds or settings['initial_seed'] in initial_seeds):
        raise ValueError('Test must use fresh target and initial-state seeds')
    if not set(settings['amplitudes_rad']) <= selected_amplitudes:
        raise ValueError('Selected checkpoint lacks validation coverage for test amplitudes')
    return {'path': str(path.resolve()), 'sha256': file_hash(path), 'record': selection}
