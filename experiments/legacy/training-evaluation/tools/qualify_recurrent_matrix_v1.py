"""Small matrix R Q0 only: independent graph oracle plus same-case Atlas.

No full-size fixture generation or performance mode is exposed. The remote-only
entry point preserves each failed run and uses the fixed pre-execution manifest.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'protocol/recurrent-matrix-v1.manifest.json'
sys.path.insert(0, str(ROOT/'adapters'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def phase(folder, started, name, **details):
    with (folder/'phases.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(phase=name, elapsed_s=time.monotonic()-started, **details))+'\n')


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    return value


def q0_cases():
    from atlas_graph_qualification import graph_fixture
    base = json.loads((ROOT/'fixtures/q0.json').read_text())
    dense = graph_fixture(base, shared=False)
    # Frozen R p=1 recurrence uses target-major bank order; feedforward banks
    # remain source-major. This one anchor exercises both zero-copy layouts.
    p = dense['projections'][-1]
    dense['weights'][-1] = [dense['weights'][-1][i] for i in p['parameter_ids']]
    p['parameter_ids'] = list(range(p['parameter_count']))
    dense['id'] = 'Q0-matrix-dense'
    sparse = copy.deepcopy(dense)
    p = sparse['projections'][-1]
    chosen = [i for i, (source, target) in enumerate(zip(p['sources'], p['targets']))
              if (source+2*target) % 3 != 0]
    sparse['weights'][-1] = [sparse['weights'][-1][i] for i in chosen]
    p['sources'] = [p['sources'][i] for i in chosen]
    p['targets'] = [p['targets'][i] for i in chosen]
    p['parameter_count'] = len(chosen)
    p['parameter_ids'] = list(range(len(chosen)))
    sparse['id'] = 'Q0-matrix-sparse'
    result = {}
    for name, case in [('dense', dense), ('sparse', sparse)]:
        result[name] = case
        boundary = copy.deepcopy(case)
        boundary['initial'] = [[1/.95, 1/.95-1e-7, 1/.95+1e-7, -.5, 1/.95, 0],
                               [.2, -.3, 0, 1.8, .7, -.8]]
        boundary['id'] += '-boundary'
        result[name+'_boundary'] = boundary
    return result


def native(case, mark):
    from atlas_adapter import AtlasAdapter
    adapter = AtlasAdapter(case, ROOT/'snapshot/brian2-rust', ROOT/'runtime/b2-train')
    updates = []
    for count in range(1, 4):
        mark('atlas_diagnostic_state_prefix_replay')
        trace = adapter.state_trace()
        if trace['status'] != 'executed':
            return dict(status=trace['status'], failure=dict(step=count, trace=trace))
        mark('atlas_public_adam_step')
        execution = adapter.call('train')
        if execution['status'] != 'executed':
            return dict(status=execution['status'], failure=dict(step=count, execution=execution))
        mark('diagnostic_conversion')
        value = execution['result']
        updates.append(dict(loss=value['loss'], logits=value['logits'], states=trace['states'], spikes=value['spikes'],
                            gradients=value['gradients'], initial_vjp=value['initial_gradients'],
                            weights=value['state']['weights'], first=value['state']['first_moment'],
                            second=value['state']['second_moment'], step=value['state']['step'],
                            tape_count_matches_runtime=execution['tape_count_matches_runtime']))
        adapter.records.clear()
        del execution, value, trace
    mark('atlas_public_sgd_step')
    sgd = AtlasAdapter(case, ROOT/'snapshot/brian2-rust', ROOT/'runtime/b2-train', optimizer='sgd').call('train')
    if sgd['status'] != 'executed':
        return dict(status=sgd['status'], failure=dict(sgd=sgd))
    out = {key: updates[0][key] for key in ('loss', 'logits', 'states', 'spikes', 'gradients', 'initial_vjp')}
    out.update(engine='atlas', version='frozen runtime/source', status='executed', updates=updates,
               one_sgd_update=sgd['result']['state']['weights'],
               identity=dict(projection='exact same fixed unique-edge graph', state_observation='small Q0 public prefix replay before each Adam step',
                             local_surrogate='no arbitrary-cotangent native API; CE-induced VJP qualified'),
               timing_role='diagnostic Q0 only; no performance evidence')
    return out


def comparisons(raw, case, mark):
    import numpy as np
    from atlas_graph_qualification import forward_vjp
    from qualify_graph_competitor import compare, check_forward
    from oracle import adam
    mark('independent_numpy_oracle')
    reference = forward_vjp(case)
    checks = check_forward(raw, reference)
    checks['sgd_bank_count'] = dict(passed=len(raw['one_sgd_update']) == len(case['weights']))
    for i, (w, g, new) in enumerate(zip(case['weights'], reference['gradients'], raw['one_sgd_update'])):
        checks[f'sgd_{i}'] = compare(new, np.asarray(w)-.001*g)
    if 'local_surrogate_vjp' in raw:
        local = raw['local_surrogate_vjp']
        margins, cotangents = np.asarray(local['margins']), np.asarray(local['cotangents'])
        checks['surrogate_forward'] = compare(local['values'], (margins > 0).astype(float), True)
        checks['surrogate_vjp'] = compare(local['vjp'], cotangents/(1+5*np.abs(margins))**2)
    weights = [np.asarray(w, dtype=np.float64) for w in case['weights']]
    first = [np.zeros_like(w) for w in weights]
    second = [np.zeros_like(w) for w in weights]
    checks['three_adam_steps'] = dict(passed=len(raw['updates']) == 3)
    for count, update in enumerate(raw['updates'], 1):
        mark('independent_numpy_oracle')
        if count != 1:
            reference = forward_vjp(case, weights)
        checks.update({f'adam{count}_{key}': result for key, result in check_forward(update, reference).items()})
        weights, first, second = adam(weights, reference['gradients'], first, second, count)
        for key, expected in [('weights', weights), ('first', first), ('second', second)]:
            checks[f'adam{count}_{key}_bank_count'] = dict(passed=len(update[key]) == len(expected))
            for i, (actual, want) in enumerate(zip(update[key], expected)):
                checks[f'adam{count}_{key}_{i}'] = compare(actual, want)
        checks[f'adam{count}_counter'] = dict(passed=bool(np.all(np.asarray(update['step']) == count)))
        if 'tape_count_matches_runtime' in update:
            checks[f'adam{count}_native_tape_formula'] = dict(passed=update['tape_count_matches_runtime'])
    return checks


def atlas_pair(raw, reference):
    from qualify_graph_competitor import compare, check_forward
    checks = {f'atlas_{key}': value for key, value in check_forward(raw, reference).items()}
    checks['atlas_sgd_bank_count'] = dict(passed=len(raw['one_sgd_update']) == len(reference['one_sgd_update']))
    for i, (a, b) in enumerate(zip(raw['one_sgd_update'], reference['one_sgd_update'])):
        checks[f'atlas_sgd_{i}'] = compare(a, b)
    checks['atlas_adam_step_count'] = dict(passed=len(raw['updates']) == len(reference['updates']) == 3)
    for count, (actual, want) in enumerate(zip(raw['updates'], reference['updates']), 1):
        checks.update({f'atlas_adam{count}_{key}': value for key, value in check_forward(actual, want).items()})
        for key in ('weights', 'first', 'second'):
            checks[f'atlas_adam{count}_{key}_bank_count'] = dict(passed=len(actual[key]) == len(want[key]))
            for i, (a, b) in enumerate(zip(actual[key], want[key])):
                checks[f'atlas_adam{count}_{key}_{i}'] = compare(a, b)
    return checks


def dependencies():
    fixed = [Path(__file__), MANIFEST, ROOT/'adapters/recurrent_matrix_v1.py', ROOT/'adapters/atlas_graph_qualification.py',
            ROOT/'adapters/atlas_adapter.py', ROOT/'adapters/torch_adapter.py', ROOT/'adapters/jax_adapter.py',
            ROOT/'adapters/oracle.py', ROOT/'tools/qualify_graph_competitor.py', ROOT/'fixtures/q0.json',
            ROOT/'protocol/recurrent-fixture-r1.json', ROOT/'protocol/finite-engine-cases.json',
            ROOT/'runtime/b2-train', ROOT/'environment/cpu-lock.txt', ROOT/'environment/jax-lock.txt',
            ROOT/'environment/hardware.json']
    native_sources = [p for p in sorted((ROOT/'snapshot/brian2-rust/python/brian2_rust').glob('training*')) if p.is_file()]
    if not native_sources:
        raise FileNotFoundError('Frozen native training source snapshot is missing')
    return fixed + native_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True, choices=['atlas', 'snntorch_fp64', 'spikingjelly', 'spyx', 'brainx_state'])
    parser.add_argument('--storage', choices=['auto', 'dense'], default='auto')
    parser.add_argument('--compile', action='store_true', help='Explicit Torch fullgraph view; rejection never falls back to eager')
    parser.add_argument('--atlas-reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('Model execution restricted to authorized 100.90.28.27')
    if args.compile and args.engine not in ('snntorch_fp64', 'spikingjelly'):
        parser.error('--compile is only the explicit Torch fullgraph view; JAX is always JIT')
    if args.engine != 'atlas' and args.atlas_reference is None:
        parser.error('A qualified exact-case Atlas Q0 reference is required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(schema='recurrent-matrix-q0-v1', engine=args.engine, storage=args.storage, compile=args.compile,
                  status='unqualified', performance_run=False, cases={},
                  scope='four fixed small cases, all states/spikes/logits/loss/CE-VJP/initial VJP, all gradients, SGD and three Adam steps',
                  resource_scope='whole qualification process tree, including diagnostics/oracle/cache; JAX is not confined to one core')
    frozen = {}
    try:
        phase(args.output, start, 'source_and_manifest_verification')
        frozen = {str(path.relative_to(ROOT)): digest(path) for path in dependencies()}
        manifest = json.loads(MANIFEST.read_text())
        if manifest['status'] != 'prepared_before_any_matrix_execution':
            raise ValueError('Unexpected matrix-v1 declaration status')
        baseline_report = None
        if args.engine != 'atlas':
            baseline_report = json.loads((args.atlas_reference/'result.json').read_text())
            if (baseline_report.get('schema') != 'recurrent-matrix-q0-v1' or baseline_report.get('status') != 'qualified'
                    or baseline_report.get('engine') != 'atlas' or not baseline_report.get('identities_unchanged')):
                raise ValueError('Atlas anchor did not qualify')
            if baseline_report.get('identities') != frozen:
                raise ValueError('Atlas anchor source/runtime identities differ')
            report['atlas_reference'] = dict(path=str(args.atlas_reference), result_sha256=digest(args.atlas_reference/'result.json'))
        phase(args.output, start, 'small_fixture_preparation')
        variants = q0_cases()
        for name, case in variants.items():
            folder = args.output/name
            folder.mkdir(exist_ok=False)
            write(folder/'case.json', case)
            case_sha = digest(folder/'case.json')
            mark = lambda name_, variant=name: phase(args.output, start, name_, variant=variant)
            result = dict(case_sha256=case_sha, status='unqualified')
            try:
                from recurrent_matrix_v1 import topology
                result['matrix_plans'] = topology(case, args.storage)[2]
                atlas_raw = None
                if baseline_report is not None:
                    baseline = baseline_report['cases'][name]
                    if (baseline['status'] != 'qualified' or baseline['case_sha256'] != case_sha
                            or digest(args.atlas_reference/name/'case.json') != case_sha):
                        raise ValueError('Atlas reference used different case arrays')
                    if digest(args.atlas_reference/name/'raw.json') != baseline['raw_sha256']:
                        raise ValueError('Atlas raw reference changed')
                    atlas_raw = json.loads((args.atlas_reference/name/'raw.json').read_text())
                if args.engine == 'atlas':
                    raw = native(case, mark)
                else:
                    from recurrent_matrix_v1 import run_case
                    raw = run_case(case, args.engine, steps=3, compiled=args.compile, storage=args.storage, phase=mark)
                mark('raw_artifact_save')
                write(folder/'raw.json', safe(raw))
                result['raw_sha256'] = digest(folder/'raw.json')
                if raw.get('status', 'executed') != 'executed':
                    result.update(status=raw['status'], reason='Exact same-case Atlas public request rejected')
                else:
                    checks = comparisons(raw, case, mark)
                    if atlas_raw is not None:
                        mark('direct_atlas_comparison')
                        checks.update(atlas_pair(raw, atlas_raw))
                    mark('checks_artifact_save')
                    write(folder/'checks.json', safe(checks))
                    passed = all(value['passed'] for value in checks.values())
                    from qualify_graph_competitor import finite
                    result.update(status='qualified' if passed else ('semantic_mismatch' if finite(raw) else 'numerical_divergence'),
                                  failed=[key for key, value in checks.items() if not value['passed']],
                                  check_count=len(checks), checks_sha256=digest(folder/'checks.json'),
                                  implementation={key: raw[key] for key in ('engine', 'version', 'torch', 'jax', 'optax', 'device', 'identity', 'timing_role') if key in raw})
                del raw, atlas_raw
            except Exception as error:
                # No string-based OOM inference: an unavailable sparse/compiled
                # path is a software rejection of this view, not engine capacity.
                result.update(status='dependency_error' if isinstance(error, ImportError) else 'software_rejected',
                              error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
            report['cases'][name] = result
            mark('case_result_save')
            write(folder/'result.json', safe(result))
            print(json.dumps(dict(case=name, engine=args.engine, status=result['status'])), flush=True)
        report['status'] = 'qualified' if all(v['status']=='qualified' for v in report['cases'].values()) else 'unqualified'
    except Exception as error:
        report.update(status='software_rejected', error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
    finally:
        report['identities'] = frozen
        try:
            report['identities_unchanged'] = bool(frozen) and frozen == {str(path.relative_to(ROOT)): digest(path) for path in dependencies()}
        except OSError:
            report['identities_unchanged'] = False
        if not report['identities_unchanged']:
            report.update(status='unqualified', reason='Source/runtime/manifest identity incomplete or changed')
        report['elapsed_s'] = time.monotonic()-start
        phase(args.output, start, 'terminal_report_save', status=report['status'])
        write(args.output/'result.json', safe(report))
    return 0 if report['status'] == 'qualified' else 2


if __name__ == '__main__':
    raise SystemExit(main())
