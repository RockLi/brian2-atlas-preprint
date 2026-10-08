"""Explicit reference admission and unchanged arithmetic; no new simulation."""
import ast
import copy
import inspect
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import analyze_mam_reference_fc as fc
import analyze_mam_paper_fc as original
import mam_native_analysis_pipeline as p


def test_reference_fc_body_is_identical_except_for_explicit_admission():
    old=ast.parse(inspect.getsource(original.analyze)).body[0].body
    new=ast.parse(inspect.getsource(fc.analyze)).body[0].body
    assert ast.unparse(new[0])=='label_for(reference_seed)'
    new=new[1:]
    old_gate=[i for i,node in enumerate(old) if isinstance(node,ast.If)
              and ast.unparse(node.test)=="identity['seed'] != 1729"]
    new_gate=[i for i,node in enumerate(new) if isinstance(node,ast.Expr)
              and ast.unparse(node)=='admit_reference(identity, bins, reference_seed)']
    assert len(old_gate)==len(new_gate)==1 and old_gate==new_gate
    old.pop(old_gate[0]);new.pop(new_gate[0])
    assert ast.dump(ast.Module(body=old,type_ignores=[]))==ast.dump(ast.Module(body=new,type_ignores=[]))
    for name in ['sha','checked','compare_fc','synaptic_area_inputs','functional_connectivity']:
        assert getattr(fc,name) is getattr(original,name)


@pytest.mark.parametrize('seed',[1730,1731])
def test_full_reference_fc_admission(seed):
    identity=dict(simulator='NEST',**{**p.IDENTITY,'seed':seed})
    fc.admit_reference(identity,100000,seed)
    for key,value in [('seed',1729),('simulator','Rust'),('ranks',32),('threads',1),
                      ('duration_ms',10500),('dt_ms',.01),('nest_version','different')]:
        with pytest.raises(ValueError):fc.admit_reference({**identity,key:value},100000,seed)
    with pytest.raises(ValueError):fc.admit_reference(identity,10000,seed)


@pytest.mark.parametrize('seed',[1729,1750,1754,True])
def test_unadmitted_fc_seed_refused_before_input_access(tmp_path,seed):
    with pytest.raises(ValueError):fc.analyze(tmp_path,tmp_path,tmp_path,tmp_path/'out',seed)
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('seed',[1730,1731])
def test_reference_plan_has_all_six_stages_under_the_shared_budget(seed):
    source=Path('/data/brick2/fixture-source');spec=p.analysis_spec(seed)
    plan=p.stages(source,source/'norm',reference_seed=seed)
    assert [x[0] for x in plan]==['activity','cell','correlation','series','fc','lags']
    assert [x[1] for x in plan]==[2400,1800,2400,1800,180,180]
    assert plan[-2][2][plan[-2][2].index('--reference-seed')+1]==str(seed)
    assert plan[-1][2][plan[-1][2].index('--fc-audit')+1]==str(spec['output']/'fc')
    assert plan[0][2][plan[0][2].index('--end-tick')+1]=='1005000'
    for name,_,command in plan:
        assert command[command.index('--output')+1]==str(spec['output']/name)
        assert not any('native-primary-postrun-v1' in item or '/runs/'+p.LABEL in item for item in command)
    assert p.TOTAL_SECONDS==10800


@pytest.fixture(params=[1730,1731])
def reference_output(tmp_path,monkeypatch,request):
    seed=request.param;spec=p.analysis_spec(seed);raw=tmp_path/'raw';out=tmp_path/'out'
    (out/'activity').mkdir(parents=True);raw.mkdir()
    original_spec=p.analysis_spec
    monkeypatch.setattr(p,'analysis_spec',lambda value:{**original_spec(value),'raw':raw,'output':out})
    results={}
    for rank in range(48):
        root=raw/('node'+p.NODES[rank//8].rsplit('-',1)[-1])/'runs'/spec['label'];root.mkdir(parents=True,exist_ok=True)
        (root/f'rank{rank}.json').write_text(json.dumps(dict(event_bytes=8,event_sha256='a'*64)))
        results[str(root/f'rank{rank}.events.bin')]=dict(bytes=8,sha256='a'*64)
    (raw/'summary.json').write_text(json.dumps(dict(physical_50ms_bin_counts=[0]*2010,spikes=48,terminal_tick_events=48)))
    baseline=dict(schema='b2-native-mam-activity-v1',simulation=spec['identity'],parameters_sha256=p.PARAMETERS,
        neurons=p.NEURONS,population_count=254,bounded_memory=True,observed_spikes=0,result_files=results,
        window=dict(start_tick=5000,end_tick=1005000,seconds=100.,spike_tick_offset=0,endpoint='[start,end)',raster_end_tick=105000))
    (out/'activity/activity.json').write_text(json.dumps(baseline))
    for name in ['activity-arrays.npz','activity-overview.png','spike-rasters.png']:
        (out/'activity'/name).write_bytes(b'explicit metadata-only fixture')
    return seed,raw,out,baseline


def catalog(directory):
    (directory/'catalog.json').write_text(json.dumps({q.name:dict(bytes=q.stat().st_size,sha256=p.sha(q))
        for q in directory.iterdir() if q.name!='catalog.json'}))


def test_reference_activity_is_bound_to_new_seed_and_paths(reference_output):
    seed,raw,out,baseline=reference_output;catalog(out/'activity')
    p.validate_output('activity',reference_seed=seed)
    with pytest.raises(ValueError):p.validate_output('activity')
    baseline['simulation']['seed']=1729
    (out/'activity/activity.json').write_text(json.dumps(baseline));catalog(out/'activity')
    with pytest.raises(ValueError):p.validate_output('activity',reference_seed=seed)


@pytest.mark.parametrize('name',['fc','lags'])
def test_reference_interarea_validation_binds_series_matrix_and_reference(reference_output,tmp_path,name):
    seed,raw,out,baseline=reference_output;source=tmp_path/'source'
    paths=[out/'series/time-series.json',out/'series/time-series.npz',out/'fc/fc.json',
           source/'interarea/matrices/matrices.json',source/'interarea/fc-reference/report.json',
           source/'interarea/lag-reference/report.json']
    for path in paths:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'explicit fixture')
    inputs=dict(series_report=p.sha(out/'series/time-series.json'),series_arrays=p.sha(out/'series/time-series.npz'),
        matrices_metadata=p.sha(source/'interarea/matrices/matrices.json'))
    r=dict(identity=dict(simulator='NEST',**p.analysis_spec(seed)['identity']),scientific_equivalence=False,
           observation_seconds=100.,area_names=['area'+str(i) for i in range(32)],source_sha256=inputs)
    if name=='fc':
        r.update(schema='b2-mam-paper-fc-v1',validated_rate_input=True,normalization_bins_exact=True,equal_observation_duration=True)
        inputs['reference_report']=p.sha(source/'interarea/fc-reference/report.json')
    else:
        r.update(schema='b2-mam-paper-propagation-v1',excluded_from_hierarchy=['MDP'],hierarchy_area_names=['area'+str(i) for i in range(31)])
        inputs.update(fc_report=p.sha(out/'fc/fc.json'),reference_audit=p.sha(source/'interarea/lag-reference/report.json'))
    directory=out/name;directory.mkdir(exist_ok=True)
    (directory/(name+'.json')).write_text(json.dumps(r));(directory/(name+'.npz')).write_bytes(b'explicit fixture');catalog(directory)
    p.validate_output(name,reference_seed=seed,source=source)
    r['source_sha256']['series_arrays']='0'*64
    (directory/(name+'.json')).write_text(json.dumps(r));catalog(directory)
    with pytest.raises(ValueError,match='inputs differ'):p.validate_output(name,reference_seed=seed,source=source)


def test_reference_budget_cannot_mix_seed_or_ignore_audit_cost(reference_output):
    seed,raw,out,_=reference_output;spec=p.analysis_spec(seed)
    from mam_launch_native_full_reference import PROTOCOL_SHA
    r=dict(schema='b2-mam-native-primary-output-audit-v1',passed=True,audit_guard_passed=True,
        raw_output_audit_passed=True,terminal_resource_audit_passed=True,label=spec['label'],seed=seed,
        parameters_sha256=p.PARAMETERS,retained_prefix_spikes=spec['prior_spikes'],all_first_2500ms_event_prefixes_exact=True,
        duration_ms=100500,event_bytes=96*2**30,audit_guard_sha256='a'*64,shared_analysis_budget_seconds=10800,
        analysis_wall_seconds_consumed=5000.)
    (raw/'summary.json').write_text(json.dumps(r))
    budget=dict(label=spec['label'],seed=seed,protocol_sha256=PROTOCOL_SHA,summary_sha256=p.sha(raw/'summary.json'),
                raw_guard_sha256='a'*64,shared_analysis_budget_seconds=10800,previous_analysis_seconds=5100.)
    assert p.raw_gate(budget,reference_seed=seed)==r
    budget['seed']=1729
    with pytest.raises(ValueError):p.raw_gate(budget,reference_seed=seed)
    budget['seed']=seed;budget['previous_analysis_seconds']=100.
    with pytest.raises(ValueError,match='budget differs'):p.raw_gate(budget,reference_seed=seed)
