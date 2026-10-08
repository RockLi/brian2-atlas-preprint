"""Prepare a full chi=1 control with exact frozen-input and model-delta audits.

Run in a bounded preparation service. This does not admit a simulation.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'examples'), str(ROOT/'tools')]
from audit_mam_ground_parameters import compare_parameters, BASELINE_SHA


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def bits(value):
    return struct.pack('>d', value).hex()


def audit_model_delta(old, model, baseline_parameters, ground_parameters):
    parameter_audit = compare_parameters(baseline_parameters, ground_parameters)
    if (model['definition'] != old['definition'] or model['run'] != old['run']
            or model['schema'] != old['schema']):
        raise ValueError('undeclared definition/run/schema change')
    restored = dict(model['instance'])
    restored['synapses'] = []
    seen = set()
    changed = 0
    for d, new, prior in zip(model['definition']['synapses'], model['instance']['synapses'],
                             old['instance']['synapses'], strict=True):
        ordinal = int(d['name'].removeprefix('mam_projection_'))
        if d['name'] != f'mam_projection_{ordinal}' or ordinal in seen:
            raise ValueError('projection identity changed')
        seen.add(ordinal)
        bp = baseline_parameters['projections'][ordinal]
        gp = ground_parameters['projections'][ordinal]
        topology = dict(new['topology'])
        initializers = dict(topology['initializers'])
        weight = dict(initializers['w'])
        for field, key in [('mean', 'weight_mean_pA'), ('std', 'weight_sd_pA')]:
            if (weight[field] != bits(gp[key]*1e-12)
                    or prior['topology']['initializers']['w'][field] != bits(bp[key]*1e-12)):
                raise ValueError('weight initializer differs from frozen parameters')
            changed += weight[field] != prior['topology']['initializers']['w'][field]
            weight[field] = prior['topology']['initializers']['w'][field]
        initializers['w'] = weight
        topology['initializers'] = initializers
        restored['synapses'].append(dict(new, topology=topology))
    if seen != set(range(len(ground_parameters['projections']))):
        raise ValueError('incomplete projection universe')
    if restored != old['instance']:
        raise ValueError('undeclared instance/initial-state/topology/delay change')
    return dict(parameter_audit=parameter_audit,weight_initializer_fields_changed=changed,
                unchanged_definition_run_initial_state_topology_seeds_delays_and_external_input=True,
                only_declared_condition_changes=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'baseline-parameters', 'parameters', 'protocol', 'output', 'runner']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    os.environ.update(B2_MAX_NEURONS='4200000', B2_MAX_INITIAL_VALUES='20000000',
                      B2_MAX_IR_BYTES='536870912')
    import brian2 as b
    from mpi_multi_area import make_model
    from brian2_rust.protocol import attach_protocol, verify_protocol
    from brian2_rust.distributed import write_mpi_project, _verify_artifact
    start = time.monotonic()
    protocol = json.loads(a.protocol.read_text())
    assert protocol['condition'] == 'stabilized-ground' and protocol['duration_ms'] == 2500
    assert digest(a.parameters) == protocol['parameters_sha256']
    assert digest(a.baseline_parameters) == BASELINE_SHA
    assert digest(a.baseline/'model.json') == '75ff4455b1345a2d0713cf88c0e655840333fab93d4407c61bf3fb692c651fcc'
    params = json.loads(a.parameters.read_text())
    base_params = json.loads(a.baseline_parameters.read_text())
    compare_parameters(base_params, params)
    old = json.loads((a.baseline/'model.json').read_text())
    verify_protocol(old)
    assert old['instance']['rng_seed'] == protocol['seed'] == 1729
    assert old['run']['duration'] == bits(2.5)
    a.output.mkdir(exist_ok=False)
    b.set_device('rust_standalone', runner=a.runner)
    areas = list(dict.fromkeys(p['area'] for p in params['populations']))
    model, _ = make_model(params, areas, steps=1, seed=protocol['seed'],
                         max_neurons=4200000, max_recurrent_edges=25000000000,
                         nest_grid=True, nest_poisson_start=True)
    model['run'] = copy.deepcopy(old['run'])
    for pop in model['definition']['populations']:
        pop['steps'] = pop['monitor']['window_steps'] = 25000
    attach_protocol(model)
    verify_protocol(model)
    result = audit_model_delta(old, model, base_params, params)
    assert model['instance']['neuron_count'] == 4129924
    assert sum(s['topology']['edge_count'] for s in model['instance']['synapses']) == 24126516728
    del old
    (a.output/'model.json').write_text(json.dumps(model)+'\n')
    placement = json.loads((a.baseline/'placement.json').read_text())
    write_mpi_project(model, a.output/'mpi', ranks=32,
                      population_owners=placement['population_owners'], compact_populations=True,
                      prebuild_shared_topology=True, runner=a.runner)
    manifest = _verify_artifact(a.output/'mpi')
    assert manifest['population_compaction']['compacted']
    assert manifest['topology_prebuild']['shared_projections'] == 24
    shutil.copyfile(a.baseline/'placement.json', a.output/'placement.json')
    shutil.copyfile(a.parameters, a.output/'parameters.json')
    shutil.copyfile(a.protocol, a.output/'condition-protocol.json')
    result.update(label=a.output.name,condition='stabilized-ground',
        model_sha256=digest(a.output/'model.json'),baseline_model_sha256=digest(a.baseline/'model.json'),
        parameters_sha256=digest(a.parameters),protocol_sha256=digest(a.protocol),seed=protocol['seed'],
        neurons=4129924,recurrent_edges=24126516728,populations=254,duration_seconds=2.5,
        prepare_seconds=time.monotonic()-start,plan_sha256=manifest['plan_sha256'],
        source_bytes=(a.output/'mpi/main.rs').stat().st_size,
        source_hashes={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),ROOT/'tools/audit_mam_ground_parameters.py',ROOT/'examples/mpi_multi_area.py']},
        scope='Preparation only; fresh simulation resource admission and scientific acceptance remain required.')
    (a.output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
