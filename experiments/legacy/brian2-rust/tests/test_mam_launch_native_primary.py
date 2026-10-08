import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    import mam_launch_native_primary
    return mam_launch_native_primary


def test_missing_primary_audit_never_contacts_hosts_or_creates_outputs(module, tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('remote access before Rust prerequisite')
    monkeypatch.setattr(module, 'remote', forbidden)
    result = module.run(tmp_path/'evidence', tmp_path/'t7')
    assert result['ready'] is False and result['launch_started'] is False
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def prerequisite(module, tmp_path):
    # Reuse genuine guard/admission identity with explicitly synthetic terminal
    # counters from the resource test fixture, not a primary acceptance claim.
    spec = importlib.util.spec_from_file_location('resource_fixture', ROOT/'tests/test_mam_primary_resources.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    admission, launch, collected, runtime = fixture.records.__wrapped__()
    directory = tmp_path/'resources'
    directory.mkdir()
    inputs = dict(admission=admission, launch=launch, collected=collected, runtime=runtime)
    hashes = {}
    for name, value in inputs.items():
        p = directory/(name+'.json')
        p.write_text(json.dumps(value))
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (directory/'input-sha256.json').write_text(json.dumps(hashes))
    (directory/'report.json').write_text(json.dumps(module.audit_rust_resources(**inputs)))
    evidence = tmp_path/'evidence'
    (evidence/'primary-output-audit').mkdir(parents=True)
    output = dict(schema='b2-mam-primary-output-audit-v1',internal_output_audit_passed=True,
        terminal_resource_audit_passed=True,model_sha256=module.MODEL,
        retained_10500ms_prefix=dict(exact=True,spikes=633265154))
    guard = dict(admitted=True,returncode=0,after={'memory.events':'max 0\noom 0\noom_kill 0\noom_group_kill 0'})
    (evidence/'primary-output-audit/full-report.json').write_text(json.dumps(output))
    (evidence/'primary-output-audit/full-guard.json').write_text(json.dumps(guard))
    return evidence, directory


def test_composed_rust_gate_returns_hashes_only_after_all_checks(module, prerequisite):
    evidence, directory = prerequisite
    result = module.rust_gate(evidence, directory)
    assert len(result) == 3 and all(len(h) == 64 for h in result.values())


@pytest.mark.parametrize('fault', ['missing-output','failed-output','wrong-model','short-prefix',
    'failed-guard','pressure','changed-control','changed-resource-decision'])
def test_incomplete_primary_cannot_unlock_native(module, prerequisite, fault):
    evidence, directory = prerequisite
    output = evidence/'primary-output-audit/full-report.json'
    guard = evidence/'primary-output-audit/full-guard.json'
    if fault == 'missing-output':
        output.unlink()
        assert module.rust_gate(evidence, directory) is None
        return
    if fault in ['failed-output','wrong-model','short-prefix']:
        value = json.loads(output.read_text())
        if fault == 'failed-output': value['internal_output_audit_passed'] = False
        elif fault == 'wrong-model': value['model_sha256'] = '0'*64
        else: value['retained_10500ms_prefix']['spikes'] -= 1
        output.write_text(json.dumps(value))
    elif fault in ['failed-guard','pressure']:
        value = json.loads(guard.read_text())
        if fault == 'failed-guard': value['returncode'] = 1
        else: value['after']['memory.events'] = value['after']['memory.events'].replace('max 0','max 1')
        guard.write_text(json.dumps(value))
    elif fault == 'changed-control':
        with (directory/'runtime.json').open('a') as f: f.write('\n')
    else:
        p = directory/'report.json'
        value = json.loads(p.read_text());value['terminal_resource_audit_passed'] = False
        p.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.rust_gate(evidence, directory)


def test_launch_preserves_full_condition_and_every_planned_limit(module, tmp_path):
    plan = json.loads((ROOT/'mpi-evidence/primary-native-admission/planned-budget.json').read_text())
    options = module.launch_options(tmp_path/'log')
    assert options['nodes'] == plan['nodes'] and options['ranks_per_node'] == 8
    assert options['timeout'] == plan['planned_hard_wall_seconds'] == 54000
    assert options['guard_memory_mib'] == plan['planned_memory_mib_per_proxy']
    assert options['guard_cpu_percent'] == plan['planned_cpu_percent_per_proxy']
    assert options['guard_cpu_count'] == len(set(options['guard_cpu_ids'])) == 32
    assert options['guard_file_mib'] == plan['planned_single_file_limit_mib']
    assert options['guard_min_free_gib'] == plan['minimum_runtime_disk_reserve_gib']
    assert options['guard_script'] == module.PROJECT+'/mpi_resource_guard.py'
    app = options['application']
    flags = {'--duration-ms':'100500','--max-duration-ms':'100500','--chunk-ms':'50',
             '--threads':'4','--ranks':'48','--seed':'1729','--max-neurons':'4200000',
             '--max-edges':'25000000000','--max-chunk-spikes':'2000000',
             '--max-spikes-per-rank':'268435456','--layout-sha256':module.LAYOUT_SHA}
    for flag, value in flags.items():
        assert app.count(flag) == 1 and app[app.index(flag)+1] == value
    assert module.PROJECT+'/mam_nest_rank_affinity.py' in app
    assert app.index(module.PROJECT+'/mam_nest_rank_affinity.py') < app.index(module.PROJECT+'/mam_nest_reference.py')
    assert options['guard_volume'] == '/' and options['guard_allow_root_volume']


def test_all_six_preflight_programs_bind_staged_identity_and_topology(module):
    base = ROOT/'mpi-evidence/primary-native-layout'
    staged = json.loads((base/'stage.json').read_text())
    layout = json.loads((base/'layout.json').read_text())
    inventory = json.loads((base/'inventory.json').read_text())['nodes']
    for index in range(6):
        code = module.preflight_code(index, staged[index], layout['hosts'][index]['selected_topology'],
                                     inventory[index]['mpi_sha256'])
        ast.parse(code)
        assert '576*2**30' in code if index == 0 else '320*2**30' in code
        assert 'free>=192*2**30' in code and 'assert not units.strip()' in code
        assert 'selected_topology_verified=True' in code


def test_explicit_seed_changes_only_seed_and_result_locations(module, tmp_path):
    label = 'nest-mam-full-reference-v1-metastable-seed1730-100500ms'
    original = module.launch_options(tmp_path/'log')
    updated = module.launch_options(tmp_path/'log', label=label, seed=1730)
    old_app, new_app = original.pop('application'), updated.pop('application')
    assert original == updated
    changed = {i for i,(a,b) in enumerate(zip(old_app,new_app,strict=True)) if a != b}
    assert changed == {old_app.index(f)+1 for f in ['--seed','--output','--receipt-directory']}
    assert module.PROJECT+'/mam_nest_reference.py' in new_app
    assert module.PROJECT+'/mpi_resource_guard.py' == updated['guard_script']


@pytest.mark.parametrize('label,seed', [('../escape',1730), ('wrong-seed1731-100500ms',1730),
    ('valid-seed1730-100500ms',True), ('valid-seed0-100500ms',0)])
def test_invalid_run_identity_rejected(module,tmp_path,label,seed):
    with pytest.raises(ValueError): module.launch_options(tmp_path,label=label,seed=seed)
