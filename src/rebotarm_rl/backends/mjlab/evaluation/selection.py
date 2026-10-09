"""Synchronous, isolated checkpoint validation using the native single-GPU evaluator."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_protocol(protocol: dict) -> dict:
    protocol = {'interval': 100, 'test_seed': 100000, **protocol}
    required = {'episodes', 'steps', 'seed', 'hold_steps', 'target_min_radius', 'target_max_radius', 'interval', 'test_seed'}
    if set(protocol) != required:
        raise ValueError('验证配置字段不完整或含未知字段')
    for name in ('episodes', 'steps', 'seed', 'hold_steps', 'interval', 'test_seed'):
        if type(protocol[name]) is not int or protocol[name] < (0 if name in ('seed', 'test_seed') else 1):
            raise ValueError(f'无效验证参数: {name}')
    if protocol['hold_steps'] > protocol['steps'] + 1:
        raise ValueError('保持样本数超过评估轨迹长度')
    low, high = protocol['target_min_radius'], protocol['target_max_radius']
    if not all(type(v) in (float, int) and math.isfinite(v) for v in (low, high)):
        raise ValueError('目标半径必须有限')
    if not 0.01 <= low <= high <= 0.06:
        raise ValueError('固定验证目标半径须满足0.01 <= min <= max <= 0.06')
    if protocol['seed'] == protocol['test_seed']:
        raise ValueError('测试种子必须与验证种子不同')
    return dict(protocol)


def candidate(checkpoint: Path, report: dict, protocol: dict, task: str) -> dict:
    """Validate result identity before allowing it into the selection ledger."""
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (report['checkpoint_sha256'] != digest or Path(report['checkpoint']).resolve() != checkpoint.resolve()
            or report['task'] != task or report['seed'] != protocol['seed']
            or report['episodes'] != protocol['episodes'] or report['steps'] != protocol['steps']):
        raise ValueError('验证报告与权重或协议不匹配')
    conditions = report['conditions']
    if (conditions['hold_steps'] != protocol['hold_steps']
            or conditions['target_position']['radius_uniform_m'] != [protocol['target_min_radius'], protocol['target_max_radius']]
            or not conditions['exclude_initial_success'] or not conditions['deterministic_policy']
            or conditions['automatic_termination']):
        raise ValueError('验证条件不匹配')
    metric = report['summary']['gpu']
    if metric['episodes_evaluated'] != protocol['episodes']:
        raise ValueError('验证回合不完整')
    count = metric['tail_success_count']
    error = metric['mean_final_position_error_m']
    if type(count) is not int or not 0 <= count <= protocol['episodes'] or not math.isfinite(error) or error < 0:
        raise ValueError('无效验证指标')
    return dict(checkpoint=checkpoint.name, checkpoint_sha256=digest,
                iteration=int(checkpoint.stem.removeprefix('model_')),
                tail_success_count=count, mean_final_position_error_m=error)


def ranking(item: dict) -> tuple:
    return (-item['tail_success_count'], item['mean_final_position_error_m'], item['iteration'])


def validate_saved_checkpoint(checkpoint: Path, task: str, protocol: dict) -> None:
    """Block training until validation finishes; never touch parent RNG/environment."""
    protocol = validate_protocol(protocol)
    directory = checkpoint.parent / 'eval' / 'validation'
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path = directory / 'selection.json'
    ledger = (json.loads(ledger_path.read_text()) if ledger_path.exists() else
              dict(task=task, protocol=protocol, backend='gpu_warp',
                   rule='maximum tail success, minimum mean final position error, earliest iteration',
                   candidates=[], best=None))
    if ledger['task'] != task or ledger['protocol'] != protocol:
        raise ValueError('同一运行禁止改变验证协议')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for item in ledger['candidates']:
        if item['checkpoint'] == checkpoint.name:
            if item['checkpoint_sha256'] != digest:
                raise ValueError('已评估权重被覆盖，禁止沿用旧选择结果')
            return
    report_path = directory / f'{checkpoint.stem}_seed{protocol["seed"]}.json'
    command = [sys.executable, '-m', 'rebotarm_rl.evaluation.paired_eval',
               '--task', task, '--checkpoint', str(checkpoint.resolve()),
               '--output', str(report_path.resolve()), '--exclude-initial-success', '--save-trajectories']
    command.extend(['--backend', 'gpu'])
    for name, value in protocol.items():
        if name in ('interval', 'test_seed'):
            continue
        command.extend(['--' + name.replace('_', '-'), str(value)])
    child_env = os.environ.copy()
    child_env.pop('REBOTARM_RL_VALIDATION', None)
    child_env['MUJOCO_GL'] = 'egl'
    with report_path.with_suffix('.log').open('w') as log:
        subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads(report_path.read_text())
    item = candidate(checkpoint, report, protocol, task)
    item['report'] = str(report_path.relative_to(checkpoint.parent))
    ledger['candidates'].append(item)
    ledger['best'] = min(ledger['candidates'], key=ranking)
    write_json(ledger_path, ledger)
    print(f"[Validation] {checkpoint.name}: tail={item['tail_success_count']}/{protocol['episodes']}; "
          f"best={ledger['best']['checkpoint']}", flush=True)


def test_selected_checkpoint(directory: Path, task: str, protocol: dict) -> None:
    """Test the frozen validation winner; never feed test results into selection."""
    ledger = json.loads((directory / 'eval/validation/selection.json').read_text())
    winner = ledger['best']
    checkpoint = directory / winner['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != winner['checkpoint_sha256']:
        raise ValueError('选定权重已改变')
    output = directory / 'eval/test' / f"{checkpoint.stem}_seed{protocol['test_seed']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'rebotarm_rl.evaluation.paired_eval',
               '--task', task, '--checkpoint', str(checkpoint.resolve()), '--backend', 'gpu',
               '--output', str(output.resolve()), '--exclude-initial-success', '--save-trajectories',
               '--selection-reason', 'Frozen validation winner; test results do not change selection']
    test_protocol = {**protocol, 'seed': protocol['test_seed']}
    for name, value in test_protocol.items():
        if name not in ('interval', 'test_seed'):
            command.extend(['--' + name.replace('_', '-'), str(value)])
    child_env = os.environ.copy()
    child_env.pop('REBOTARM_RL_VALIDATION', None)
    child_env['MUJOCO_GL'] = 'egl'
    with output.with_suffix('.log').open('w') as log:
        subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    candidate(checkpoint, json.loads(output.read_text()), test_protocol, task)
    print(f'[Test] Frozen winner {checkpoint.name}: {output}', flush=True)
