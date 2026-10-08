"""Regenerate the full MAM and prove the two declared NEST semantic changes.

Preparation only: host admission and runtime guards remain separate requirements.
Run inside the existing bounded preparation service, not on an unguarded host.
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
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'examples')]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(B2_MAX_NEURONS='4200000', B2_MAX_INITIAL_VALUES='20000000',
                      B2_MAX_IR_BYTES='536870912')
    import brian2 as b
    from mpi_multi_area import make_model
    from brian2_rust.protocol import attach_protocol, verify_protocol
    from brian2_rust.distributed import write_mpi_project, _verify_artifact
    started = time.monotonic()
    args.output.mkdir(exist_ok=False)
    old = json.loads((args.baseline / 'model.json').read_text())
    verify_protocol(old)
    parameters = json.loads(args.parameters.read_text())
    assert digest(args.parameters) == 'ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e'
    assert old['instance']['neuron_count'] == 4129924
    assert sum(s['topology']['edge_count'] for s in old['instance']['synapses']) == 24126516728
    assert old['run']['duration'] == struct.pack('>d', 2.5).hex()
    b.set_device('rust_standalone', runner=args.runner)
    areas = list(dict.fromkeys(p['area'] for p in parameters['populations']))
    model, _ = make_model(parameters, areas, steps=1, seed=old['instance']['rng_seed'],
                          max_neurons=4200000, max_recurrent_edges=25000000000,
                          nest_grid=True, nest_poisson_start=True)
    model['run'] = copy.deepcopy(old['run'])
    for pop in model['definition']['populations']:
        pop['steps'] = 25000
        pop['monitor']['window_steps'] = 25000
    attach_protocol(model)
    verify_protocol(model)

    # Restore only the two specific semantic changes, then compare everything.
    # Large unchanged arrays stay shared until the equality check has finished.
    restored_instance = dict(model['instance'])
    restored_instance['populations'] = []
    periods = []
    for new, prior in zip(model['instance']['populations'], old['instance']['populations'], strict=True):
        replacement = dict(new)
        refractory = dict(new['refractory'])
        assert prior['refractory']['period_ticks'] == 20 and refractory['period_ticks'] == 21
        assert prior['refractory']['period'] == struct.pack('>d', .002).hex()
        assert abs(struct.unpack('>d', bytes.fromhex(refractory['period']))[0] - .0021) < 1e-18
        periods.append({'old': prior['refractory']['period'], 'new': refractory['period']})
        for key in ['period', 'period_ticks']:
            refractory[key] = prior['refractory'][key]
        replacement['refractory'] = refractory
        restored_instance['populations'].append(replacement)
    assert restored_instance == old['instance'], 'undeclared instance difference'
    restored_definition = copy.deepcopy(model['definition'])
    gates = []

    def restore_gate(node, path):
        if isinstance(node, dict):
            if node.get('op') == 'ge' and node.get('left', {}).get('op') == 'tick_to_f64':
                assert node['left']['arg'] == {'op': 'timestep', 'time': {'op': 'load', 'name': 't'}, 'dt': {'op': 'load', 'name': 'dt'}}
                assert node['right'] == {'op': 'cast', 'dtype': 'f64', 'arg': {'op': 'integer', 'dtype': 'i64', 'value': '11'}}
                node['right']['arg']['value'] = '10'
                gates.append(path)
            else:
                for key, value in node.items():
                    restore_gate(value, path + '/' + key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                restore_gate(value, path + '/' + str(index))

    for index, pop in enumerate(restored_definition['populations']):
        before = len(gates)
        for code in pop['code_objects']:
            if code['kind'] == 'run_regularly':
                restore_gate(code, 'populations/' + str(index) + '/' + code['name'])
        assert len(gates) == before + 1, 'expected one external input gate per full-model population'
    assert restored_definition == old['definition'], 'undeclared definition difference'
    assert len(gates) == len(periods) == 254
    assert model['run'] == old['run'] and model['schema'] == old['schema']
    del restored_instance, restored_definition, old
    (args.output / 'model.json').write_text(json.dumps(model) + '\n')
    placement = json.loads((args.baseline / 'placement.json').read_text())
    write_mpi_project(model, args.output / 'mpi', ranks=32,
                      population_owners=placement['population_owners'],
                      compact_populations=True, prebuild_shared_topology=True,
                      runner=args.runner)
    manifest = _verify_artifact(args.output / 'mpi')
    assert manifest['population_compaction']['compacted']
    assert manifest['topology_prebuild']['shared_projections'] == 24
    shutil.copyfile(args.baseline / 'placement.json', args.output / 'placement.json')
    admission = json.loads((args.baseline / 'admission.json').read_text())
    admission.update(label=args.output.name,
        scope='Both audited NEST semantic options. Original conservative 2.5 s capacity envelope retained; activity may change and hard limits still apply.',
        baseline=str(args.baseline), semantic_activity_uncertainty=True)
    (args.output / 'admission.json').write_text(json.dumps(admission, indent=2) + '\n')
    report = dict(label=args.output.name, model_sha256=digest(args.output / 'model.json'),
        baseline_model_sha256=digest(args.baseline / 'model.json'),
        parameters_sha256=digest(args.parameters), seed=model['instance']['rng_seed'],
        populations=254, neurons=4129924, recurrent_edges=24126516728,
        duration_seconds=2.5, refractory_changes=periods, poisson_gate_paths=gates,
        only_declared_semantic_changes=True, unchanged_initial_state_topology_and_other_parameters=True,
        prepare_seconds=time.monotonic()-started, plan_sha256=manifest['plan_sha256'],
        source_bytes=(args.output / 'mpi/main.rs').stat().st_size,
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT/'examples/mpi_multi_area.py', ROOT/'python/brian2_rust/multi_area_semantics.py']})
    (args.output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['refractory_changes', 'poisson_gate_paths']}, indent=2))


if __name__ == '__main__':
    main()
