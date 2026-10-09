import hashlib
import json
from pathlib import Path
import subprocess

import pytest
from rebotarm_rl.backends.mjlab.evaluation import selection as s

P = dict(episodes=2, steps=250, seed=30000, hold_steps=25,
         target_min_radius=.02, target_max_radius=.06)


def report(path, count=1, error=.003):
    return dict(checkpoint=str(path.resolve()), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                task='task', episodes=2, steps=250, seed=30000,
                conditions=dict(hold_steps=25, target_position=dict(radius_uniform_m=[.02,.06]),
                                exclude_initial_success=True, deterministic_policy=True, automatic_termination=False),
                summary=dict(gpu=dict(episodes_evaluated=2, tail_success_count=count, mean_final_position_error_m=error)))


def test_selection_order_identity_and_failures(tmp_path, monkeypatch):
    calls=[]
    def run(command, **kwargs):
        calls.append(command)
        assert 'REBOTARM_RL_VALIDATION' not in kwargs['env']
        checkpoint=Path(command[command.index('--checkpoint')+1])
        Path(command[command.index('--output')+1]).write_text(json.dumps(report(checkpoint)))
    monkeypatch.setattr(s.subprocess,'run',run)
    monkeypatch.setenv('REBOTARM_RL_VALIDATION','{}')
    for number in [100,50]:
        path=tmp_path/f'model_{number}.pt';path.write_bytes(b'weights')
        s.validate_saved_checkpoint(path,'task',P)
    ledger=tmp_path/'eval/validation/selection.json'
    assert json.loads(ledger.read_text())['best']['checkpoint']=='model_50.pt'
    s.validate_saved_checkpoint(path,'task',P)
    assert len(calls)==2
    before=ledger.read_bytes()
    with pytest.raises(ValueError,match='协议'):
        s.validate_saved_checkpoint(path,'task',{**P,'seed':1})
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='覆盖'):
        s.validate_saved_checkpoint(path,'task',P)
    assert ledger.read_bytes()==before
    path=tmp_path/'model_200.pt';path.write_bytes(b'weights')
    def fail(*args,**kwargs):raise subprocess.CalledProcessError(1,args[0])
    monkeypatch.setattr(s.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):s.validate_saved_checkpoint(path,'task',P)
    assert ledger.read_bytes()==before


def test_invalid_metrics_and_ranking(tmp_path):
    p=tmp_path/'model_1.pt';p.write_bytes(b'x')
    a=s.candidate(p,report(p,1,.002),P,'task')
    b=s.candidate(p,report(p,2,.01),P,'task')
    assert s.ranking(b)<s.ranking(a)
    c=s.candidate(p,report(p,1,.001),P,'task')
    assert s.ranking(c)<s.ranking(a)
    for value in [float('nan'),float('inf'),-.1]:
        with pytest.raises(ValueError):s.candidate(p,report(p,error=value),P,'task')
    d=report(p);d['checkpoint_sha256']='wrong'
    with pytest.raises(ValueError):s.candidate(p,d,P,'task')
    for override in [dict(seed=-1),dict(episodes=0),dict(hold_steps=252),dict(target_min_radius=float('nan'))]:
        with pytest.raises(ValueError):s.validate_protocol({**P,**override})
