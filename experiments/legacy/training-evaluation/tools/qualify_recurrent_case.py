"""Full frozen R-case admission/qualification; never a performance benchmark."""
import argparse
import json
import math
from pathlib import Path
import socket
import sys
import time
import traceback

from generate_recurrent_cases import ROOT, SEEDS, CONTRACT, CASE_SOURCE, cases, paths, contract, digest
sys.path.insert(0, str(ROOT/'adapters'))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def mark(folder, start, phase, **details):
    with (folder/'phases.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-start, phase=phase, **details), allow_nan=False)+'\n')
        stream.flush()


def archive(folder, stem, raw):
    import numpy as np
    arrays = {}
    for key, value in raw.items():
        if key in ('weights', 'first', 'second', 'gradients'):
            arrays.update({f'{key}_{i}': np.asarray(v) for i, v in enumerate(value)})
        else:
            arrays[key] = np.asarray(value)
    dest = folder/f'{stem}.npz'
    if dest.exists():
        raise FileExistsError(dest)
    np.savez_compressed(dest, **arrays)
    return dict(path=dest.name, sha256=digest(dest), bytes=dest.stat().st_size,
                arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()})


def compare(a, b, exact=False):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return dict(passed=False, actual_shape=list(a.shape), expected_shape=list(b.shape))
    finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
    delta = np.abs(a-b)
    return dict(passed=finite and bool(np.array_equal(a, b) if exact else np.all(delta <= 1e-10+1e-8*np.abs(b))),
                max_abs=float(delta.max(initial=0)) if finite else None, finite=finite)


def compare_banks(actual, expected, name):
    checks = {name+'_bank_count': dict(passed=len(actual) == len(expected))}
    checks.update({f'{name}_{i}': compare(a, b) for i, (a, b) in enumerate(zip(actual, expected))})
    return checks


def compare_forward(actual, reference, *, final_only=False):
    checks = {key: compare(actual[key], reference[key], key == 'spikes')
              for key in ('loss', 'logits', 'spikes', 'initial_vjp')}
    checks['final_state' if final_only else 'states'] = compare(actual['final_state'], reference['states'][:, -1]) if final_only else compare(actual['states'], reference['states'])
    checks.update(compare_banks(actual['gradients'], reference['gradients'], 'gradient'))
    return checks


def checks_status(checks):
    if all(item['passed'] for item in checks.values()):
        return 'qualified'
    return 'numerical_divergence' if any(item.get('finite') is False for item in checks.values()) else 'semantic_mismatch'


def atlas(case, args, report, start):
    import numpy as np
    from atlas_adapter import AtlasAdapter, admission, failure_status
    from atlas_graph_qualification import forward_vjp
    from oracle import adam
    mark(args.output, start, 'atlas_setup')
    adapter = AtlasAdapter(case, ROOT/'snapshot/brian2-rust', ROOT/'runtime/b2-train')
    report['steps'] = []
    for count in range(1, 2 if args.mode == 'admission' else 4):
        mark(args.output, start, 'atlas_exact_admission', step=count)
        check = admission(adapter.trainer.plan, adapter.trainer.state, case['inputs'], case['labels'], operation='train', initial=None)
        entry = dict(step=count, optimizer_step_before=adapter.trainer.state['step'], admission=check)
        write(args.output/f'admission-step-{count}.json', entry)
        report['steps'].append(entry)
        if check['status'] != 'admitted':
            report.update(status='budget_rejected', rejected_step=count, passed_native_steps=count-1,
                          boundary='exact Atlas software request/tape admission, not physical memory')
            return
        if args.mode == 'admission':
            report.update(status='admitted_initial_only', native_executed=False,
                          limitation='Only the initial request is known. Later Adam-state admission requires a real native update; no later rejection is fabricated.')
            return
        mark(args.output, start, 'atlas_public_execute', step=count)
        before = time.perf_counter_ns()
        try:
            result = adapter.trainer.execute(case['inputs'], case['labels'], operation='train', initial=None)
        except Exception as error:
            report.update(status=failure_status(error), error_type=type(error).__name__, error=str(error), failed_step=count)
            return
        entry['public_api_ns_diagnostic_only'] = time.perf_counter_ns()-before
        report['native_executed'] = True
        report['native_steps_executed'] = count
        actual = dict(loss=result['loss'], logits=result['logits'], spikes=result['spikes'], final_state=result['final_membrane'],
                      initial_vjp=result['initial_gradients'], gradients=result['gradients'], weights=result['state']['weights'],
                      first=result['state']['first_moment'], second=result['state']['second_moment'])
        mark(args.output, start, 'oracle_validation', step=count)
        if count == 1:
            weights = [np.asarray(w, dtype=np.float64) for w in case['weights']]
            first = [np.zeros_like(w) for w in weights]
            second = [np.zeros_like(w) for w in weights]
        reference = forward_vjp(case, weights)
        checks = compare_forward(actual, reference, final_only=True)
        weights, first, second = adam(weights, reference['gradients'], first, second, count)
        for key, expected in [('weights', weights), ('first', first), ('second', second)]:
            checks.update(compare_banks(actual[key], expected, key))
        checks['optimizer_counter'] = dict(passed=result['state']['step'] == count)
        checks['native_tape_formula'] = dict(passed=result['tape_bytes'] == check['exact_native_tape_bytes'])
        entry['checks'] = checks
        write(args.output/f'checks-step-{count}.json', checks)
        mark(args.output, start, 'artifact_save', step=count)
        entry['artifact'] = archive(args.output, f'actual-step-{count}', actual)
        write(args.output/f'qualified-step-{count}.json', entry)
        if not all(item['passed'] for item in checks.values()):
            report['status'] = checks_status(checks)
            return
        del result, actual, reference
    report.update(status='qualified', native_executed=True,
                  scope='full-size final state, complete spikes, CE loss/initial VJP/parameter gradients and three native Adam updates; prior small graph all-state qualification retained')


def competitor(case, args, report, start):
    import numpy as np
    from graph_competitor import run_case
    from atlas_graph_qualification import forward_vjp
    from oracle import adam
    mark(args.output, start, 'engine_adapter_execution_including_diagnostic_conversion')
    raw = run_case(case, args.engine, steps=3, compiled=args.compile)
    report['implementation'] = {key: raw[key] for key in ('engine', 'version', 'identity', 'timing_role')}
    mark(args.output, start, 'oracle_validation', step=0)
    reference = forward_vjp(case)
    checks = compare_forward(raw, reference)
    local = raw['local_surrogate_vjp']
    margin, cot = np.asarray(local['margins']), np.asarray(local['cotangents'])
    checks['local_surrogate_forward'] = compare(local['values'], (margin > 0).astype(float), True)
    checks['local_surrogate_vjp'] = compare(local['vjp'], cot/(1+5*np.abs(margin))**2)
    checks.update(compare_banks(raw['one_sgd_update'], [np.asarray(w)-.001*g for w, g in zip(case['weights'], reference['gradients'])], 'sgd'))
    weights = [np.asarray(w, dtype=np.float64) for w in case['weights']]
    first = [np.zeros_like(w) for w in weights]
    second = [np.zeros_like(w) for w in weights]
    checks['three_adam_updates'] = dict(passed=len(raw['updates']) == 3)
    for count, actual in enumerate(raw['updates'], 1):
        mark(args.output, start, 'oracle_validation', step=count)
        if count != 1:
            reference = forward_vjp(case, weights)
        checks.update({f'update{count}_{key}': value for key, value in compare_forward(actual, reference).items()})
        weights, first, second = adam(weights, reference['gradients'], first, second, count)
        for key, expected in [('weights', weights), ('first', first), ('second', second)]:
            checks.update(compare_banks(actual[key], expected, f'update{count}_{key}'))
        checks[f'update{count}_counter'] = dict(passed=bool(np.all(np.asarray(actual['step']) == count)))
    report['checks'] = checks
    write(args.output/'checks.json', dict(complete_numeric_checks=True, checks=checks,
                                        artifact_archival_complete=False, qualified_numerically=all(c['passed'] for c in checks.values())))
    mark(args.output, start, 'artifact_save', step=0)
    fields = ('loss', 'logits', 'states', 'spikes', 'gradients', 'initial_vjp')
    artifacts = [archive(args.output, 'initial-qualification', {key: raw[key] for key in fields})]
    for count, actual in enumerate(raw['updates'], 1):
        mark(args.output, start, 'artifact_save', step=count)
        artifacts.append(archive(args.output, f'adam-update-{count}', {key: actual[key] for key in (*fields, 'weights', 'first', 'second')}))
    report.update(artifacts=artifacts, status=checks_status(checks),
                  initial_spike_count=int(np.asarray(raw['spikes']).sum()),
                  scope='full-size states/spikes/loss, CE initial/parameter VJP, local surrogate VJP, SGD and three Adam steps')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id', choices=tuple(cases()), required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--engine', choices=['atlas', 'snntorch_fp64', 'spikingjelly', 'spyx', 'brainx_state'], required=True)
    parser.add_argument('--mode', choices=['admission', 'qualify'], required=True)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('Recurrent evaluation is restricted to 100.90.28.27')
    if args.mode == 'admission' and args.engine != 'atlas':
        parser.error('Admission-only mode is the exact Atlas software guard, not a competitor admission claim')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(schema='recurrent-case-qualification-v1', case_id=args.case_id, seed=args.seed, engine=args.engine,
                  mode=args.mode, status='unqualified', performance_run=False, compile=args.compile,
                  timeout_interpretation='Phase-specific qualification outcome; oracle/serialization/adapter diagnostic conversion timeout is not engine performance/capacity evidence',
                  hardware_scope='remote Mac Studio; same E1 settings; JAX pool not constrained to one CPU; all-worker RSS is not training-only memory')
    files = [Path(__file__), CONTRACT, CASE_SOURCE, ROOT/'tools/generate_recurrent_cases.py', ROOT/'tools/generate_dense_large.py', ROOT/'adapters/graph_competitor.py',
             ROOT/'adapters/atlas_graph_qualification.py', ROOT/'adapters/atlas_adapter.py', ROOT/'adapters/torch_adapter.py',
             ROOT/'adapters/jax_adapter.py', ROOT/'adapters/oracle.py', ROOT/'runtime/b2-train']
    frozen = {str(path.relative_to(ROOT)): digest(path) for path in files}
    try:
        contract()
        mark(args.output, start, 'fixture_read_and_verify')
        path, manifest_path = paths(args.case_id, args.seed)
        manifest = json.loads(manifest_path.read_text())
        if digest(path) != manifest['arrays_sha256'] or manifest['identity']['contract_sha256'] != digest(CONTRACT):
            raise ValueError('Common arrays or frozen recurrent contract changed')
        case = json.loads(path.read_text())
        report.update(arrays_sha256=manifest['arrays_sha256'], manifest_sha256=digest(manifest_path), actual_recurrent_edges=manifest['recurrent_edges'])
        if args.engine == 'atlas':
            atlas(case, args, report, start)
        else:
            competitor(case, args, report, start)
    except Exception as error:
        from atlas_adapter import failure_status
        status = failure_status(error)
        if 'out of memory' in str(error).lower():
            status = 'oom'
        report.update(status=status, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        report['elapsed_s'] = time.monotonic()-start
        report['identities'] = frozen
        report['identities_unchanged'] = frozen == {str(path.relative_to(ROOT)): digest(path) for path in files}
        if not report['identities_unchanged']:
            report.update(status='unqualified', invalidation='Frozen implementation changed during execution')
        mark(args.output, start, 'terminal_report_save', status=report['status'])
        write(args.output/'result.json', report)
        print(json.dumps({key: report[key] for key in ('case_id', 'seed', 'engine', 'mode', 'status', 'elapsed_s')}), flush=True)
    return 0 if report['status'] in ('qualified', 'admitted_initial_only', 'budget_rejected') else 2


if __name__ == '__main__':
    raise SystemExit(main())
