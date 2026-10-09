from pathlib import Path
from types import SimpleNamespace
import pytest


def test_periodic_final_and_holdout_order(monkeypatch, tmp_path):
    pytest.importorskip('mjlab')
    from rebotarm_rl.backends.mjlab.runner import RecordedRunner, MjlabOnPolicyRunner
    from rebotarm_rl.backends.mjlab.evaluation import selection
    runner=object.__new__(RecordedRunner)
    runner.contract=SimpleNamespace(task_id='task',to_dict=lambda:{})
    runner._validation_protocol={'interval':100}
    calls=[]
    monkeypatch.setattr(MjlabOnPolicyRunner,'save',lambda *a,**k:None)
    monkeypatch.setattr(selection,'validate_saved_checkpoint',lambda p,*a:calls.append(('validate',p.name)))
    monkeypatch.setattr(selection,'test_selected_checkpoint',lambda p,*a:calls.append(('test',p.name)))
    def learn(*a,**k):
        for n in [0,50,100,150,200,249]:runner.save(str(tmp_path/f'model_{n}.pt'))
    monkeypatch.setattr(MjlabOnPolicyRunner,'learn',learn)
    runner.learn(250)
    assert calls==[('validate','model_100.pt'),('validate','model_200.pt'),('validate','model_249.pt'),('test',tmp_path.name)]
    calls.clear()
    def failed(*a,**k):raise RuntimeError('interrupted')
    monkeypatch.setattr(MjlabOnPolicyRunner,'learn',failed)
    with pytest.raises(RuntimeError):runner.learn(250)
    assert calls==[]
