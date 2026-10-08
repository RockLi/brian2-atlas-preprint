"""Synthetic controls and matrices; these tests are not neural evidence."""
import ast
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import compare_mam_confirmation_lags as new
import compare_mam_primary_lags as old
import mam_launch_confirmation_analysis as launch
from test_mam_confirmation_terminal import controls
from test_mam_confirmation_raw import bound
from test_mam_launch_confirmation_raw import publication
from test_mam_confirmation_analysis import accepted
from test_mam_launch_confirmation_analysis import science


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture
def completed(accepted,science):
    case,digest=accepted;p,g,c,kw=copy.deepcopy(science)
    p.update(completion_sha256=digest,raw_report_sha256=new.sha(case/'raw/report.json'))
    kw.update(completion_sha=digest,raw_report_sha=new.sha(case/'raw/report.json'))
    g['command']=launch.application(digest,kw['previous'])
    c['command']=launch.command(digest,kw['previous'],kw['limit'])
    root=case/'science'
    for stage in p['catalogs']:
        write(root/stage/'catalog.json',{})
        p['catalogs'][stage]=new.sha(root/stage/'catalog.json')
    values={'pending.json':p,'guard.json':g,'controller.json':c,'intent.json':{},
        'admission.json':dict(previous_seconds=kw['previous'],limit_seconds=kw['limit'],source_catalog=kw['catalog'])}
    for name,value in values.items():write(root/name,value)
    r=launch.publish(p,g,c,**kw)
    r['evidence_sha256']={str(f.relative_to(root)):new.sha(f) for f in root.rglob('*.json')}
    write(root/'report.json',r)
    write(root/'complete.json',dict(replicate=1750,full_descriptive_analysis_complete=True,
        scientific_acceptance=False,performance_cost_acceptance=False,total_seconds=1001,
        report_sha256=new.sha(root/'report.json')))
    return case


def test_science_gate_replays_engineering_and_six_stage_decisions(completed):
    v,r=new.accepted_science(completed)
    assert v['replicate']==1750 and r['analysis_guard_passed']
    assert not r['scientific_acceptance'] and not r['formal_equivalence_acceptance']


@pytest.mark.parametrize('name',['raw/report.json','science/guard.json','science/lags/catalog.json'])
def test_changed_evidence_rejected(completed,name):
    with (completed/name).open('a') as f:f.write(' ')
    with pytest.raises(ValueError):new.accepted_science(completed)


def test_pending_analysis_creates_no_comparison(tmp_path):
    p=tmp_path/'missing';out=tmp_path/'output'
    result=new.compare(p,p,p,out,tmp_path)
    assert not result['ready'] and not result['comparison_written'] and not out.exists()


def artifact(root,identity):
    root.mkdir()
    names=['area'+str(i) for i in range(31)];areas=names+['MDP']
    levels=np.arange(31,dtype=float)
    retained=levels[:,None]-levels[None,:]
    fit=new.fit_hierarchy(retained)
    full=np.full((32,32),np.nan);full[:31,:31]=retained
    np.savez_compressed(root/'lags.npz',lag_ms=full,retained_lag_ms=retained,
        levels_ms=fit['levels_ms'],normalized_levels=fit['normalized_levels'])
    report=dict(schema='b2-mam-paper-propagation-v1',observation_seconds=100.,excluded_from_hierarchy=['MDP'],
        settings=new.SETTINGS,identity=identity,area_names=areas,hierarchy_area_names=names,
        implementation_sha256={'mam_paper_propagation.py':new.sha(Path(new.__file__).with_name('mam_paper_propagation.py'))},
        levels_ms=fit['levels_ms'].tolist(),normalized_levels=fit['normalized_levels'].tolist(),
        ordered_early_to_late=names,source_sha256={'reference_audit':'a'*64,'matrices_metadata':'b'*64})
    write(root/'lags.json',report)
    write(root/'catalog.json',{p.name:dict(bytes=p.stat().st_size,sha256=new.sha(p)) for p in root.iterdir()})
    return root


def test_new_identity_preserves_numerical_results_and_old_gate(controls,tmp_path):
    v=controls[0]
    before=artifact(tmp_path/'old',dict(simulator='Rust',model_sha256=old.RUST_MODELS[100000],seed=1729))
    after=artifact(tmp_path/'new',dict(simulator='Rust',model_sha256=v['model_sha256'],seed=v['random_keys']['runtime_input']))
    a=old.read_artifact(before,'rust');b=new.read_artifact(after,'rust',v)
    for x,y in zip(a[1:4],b[1:4]):np.testing.assert_array_equal(x,y)
    assert old.lag_stats(a[1],a[2],a[1],a[2])==new.lag_stats(b[1],b[2],a[1],a[2])
    with pytest.raises(ValueError):old.read_artifact(after,'rust')
    with pytest.raises(ValueError):new.read_artifact(before,'rust',v)


@pytest.mark.parametrize('seed',[1750,1729,16757147634959265529.0])
def test_label_or_rounded_key_rejected(controls,tmp_path,seed):
    v=controls[0];root=artifact(tmp_path/'bad',dict(simulator='Rust',model_sha256=v['model_sha256'],seed=seed))
    with pytest.raises(ValueError):new.read_artifact(root,'rust',v)


def test_array_validation_and_numerical_tail_identical():
    def tail(module):
        tree=ast.parse(Path(module.__file__).read_text())
        body=next(n.body for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read_artifact')
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in n.targets))
        return [ast.dump(n) for n in body[start:]]
    assert tail(old)==tail(new)
