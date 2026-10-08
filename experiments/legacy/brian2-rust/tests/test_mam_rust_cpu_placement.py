import copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import mam_rust_cpu_placement as placement
import mam_launch_rust_performance as launcher
from test_mam_rust_benchmark_terminal import controls,edit_done
from mam_rust_benchmark_terminal import audit


def test_candidate_preserves_physical_core_count_socket_and_numa():
    topology=placement.candidate_topologies(ROOT/'mpi-evidence')
    assert len(topology)==4
    assert all([int(row[0]) for row in host]==placement.ALTERNATE_CPUS for host in topology)
    old=launcher.launch_options(Path('/unused'));new=launcher.launch_options(Path('/unused'),cpu_ids=placement.ALTERNATE_CPUS)
    assert {k:v for k,v in old.items() if k!='guard_cpu_ids'}=={k:v for k,v in new.items() if k!='guard_cpu_ids'}


def test_old_and_alternate_admissions_require_exact_revision():
    assert placement.admitted_cpus({})==placement.CPUS
    assert placement.admitted_cpus(dict(cpu_placement_revision=placement.revision()))==placement.ALTERNATE_CPUS
    row=placement.revision();row['cpu_ids']=[2,14,26,38,50,62,74,86]
    with pytest.raises(ValueError):placement.admitted_cpus(dict(cpu_placement_revision=row))


@pytest.mark.parametrize('mutate',[
    lambda row:row['hosts'][0]['topology'].__setitem__(1,['1','1','1','1','Y']),
    lambda row:row['hosts'][0]['samples'][0].update({'1':50}),
    lambda row:row['hosts'][0].update(own_units='b2mpi-live.service'),
    lambda row:row['hosts'].pop(),
])
def test_candidate_changed_geometry_busy_cpu_or_running_unit_rejected(tmp_path,monkeypatch,mutate):
    source=ROOT/'mpi-evidence/rust-cpu-placement-diagnosis-v1'
    target=tmp_path/source.name;target.mkdir()
    row=json.loads((source/'candidate.json').read_text());mutate(row)
    (target/'candidate.json').write_text(json.dumps(row));(target/'report.json').write_bytes((source/'report.json').read_bytes())
    monkeypatch.setattr(placement,'CANDIDATE_SHA',placement.sha(target/'candidate.json'))
    with pytest.raises(ValueError):placement.candidate_topologies(tmp_path)


def alternate_controls(controls):
    admission,launch,hosts=controls;ids=placement.ALTERNATE_CPUS
    admission['cpu_placement_revision']=placement.revision();admission['resources']['cpu_ids']=ids
    for host in hosts:
        for guard in host['guards'].values():
            for stage in ['before','after']:guard[stage]['cpuset.cpus.effective']=','.join(map(str,ids))
    output=hosts[0]['leader_outputs'];runtime=json.loads(output['runtime_json']);runtime['rank_cpu_ids']=ids*4
    summary=json.loads(output['summary_json']);summary['mpi']=runtime
    output['runtime_json']=json.dumps(runtime);output['summary_json']=json.dumps(summary)
    import hashlib
    def rehash(done):
        for name,key in [('summary.json','summary_json'),('mpi-runtime.json','runtime_json')]:
            raw=output[key].encode();done['leader_output_synchronization']['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    edit_done(controls,rehash)
    return controls


def test_terminal_accepts_new_binding_without_changing_science_or_cost_claims(controls):
    result=audit(*alternate_controls(controls))
    assert result['terminal_resource_audit_passed']
    assert not result['scientific_acceptance'] and not result['performance_cost_acceptance']


@pytest.mark.parametrize('mutate',[
    lambda c:c[0].pop('cpu_placement_revision'),
    lambda c:c[0]['resources'].update(cpu_ids=placement.CPUS),
    lambda c:c[2][0]['guards']['proxy-0']['before'].update({'cpuset.cpus.effective':','.join(map(str,placement.CPUS))}),
    lambda c:c[0]['cpu_placement_revision'].update(maximum_neural_runs=2),
])
def test_terminal_rejects_unrecorded_or_mismatched_cpu_relocation(controls,mutate):
    alternate_controls(controls);mutate(controls)
    with pytest.raises(ValueError):audit(*controls)


def test_previous_window_cannot_be_reused_after_new_attempt(tmp_path,monkeypatch):
    (tmp_path/'alternate-cpus-v3').mkdir()
    with pytest.raises(ValueError,match='already attempted'):placement.previous_window_gate(tmp_path)


def test_new_cpu_preflight_retains_full_checks():
    package=json.loads((ROOT/'mpi-evidence/primary-run/package.json').read_text())
    topology=placement.candidate_topologies(ROOT/'mpi-evidence')[0]
    code=launcher.preflight_code(0,package,{},topology,cpu_ids=placement.ALTERNATE_CPUS)
    import ast
    ast.parse(code)
    assert 'for c in [1, 13, 25, 37, 49, 61, 73, 85]' in code
    assert 'max(busy.values())<50' in code and "m['MemAvailable']>=(576 if leader else 320)*2**30" in code
    assert 'digest(path)==item' in code and 'signal.alarm(40)' in code
