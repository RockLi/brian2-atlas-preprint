"""One remote E1-large worker: admission, three real Adam steps, then timing.

The common arrays, batch and time length never change to admit a rejected case.
Qualification, exact request counting, array archival and oracle execution are
outside public-API timing, but all consume the supervising 360-second budget.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'adapters'))
from generate_dense_large import RULE, SEEDS, digest


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check(actual, expected, exact=False):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return dict(passed=False, actual_shape=list(a.shape), expected_shape=list(b.shape))
    finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
    delta = np.abs(a-b)
    return dict(passed=finite and bool(np.array_equal(a, b) if exact else np.all(delta <= 1e-10+1e-8*np.abs(b))),
                max_abs=float(delta.max(initial=0)) if finite else None, finite=finite, shape=list(b.shape))


def bank_checks(actual, expected, prefix):
    result = {prefix+'_bank_count': dict(passed=len(actual) == len(expected))}
    result.update({f'{prefix}_{i}': check(a, b) for i, (a, b) in enumerate(zip(actual, expected))})
    return result


def save_arrays(folder, stem, raw):
    """Lossless raw numerical evidence; JSON metadata never embeds giant arrays."""
    import numpy as np
    arrays = {}
    for key, value in raw.items():
        if key in ('weights', 'first', 'second', 'gradients'):
            arrays.update({f'{key}_{i}': np.asarray(bank) for i, bank in enumerate(value)})
        else:
            arrays[key] = np.asarray(value)
    path = folder/(stem+'.npz')
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **arrays)
    return dict(path=path.name, sha256=digest(path), bytes=path.stat().st_size,
                arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()})


def classify(error):
    message = str(error).lower()
    if isinstance(error, MemoryError) or 'out of memory' in message:
        return 'oom'
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return 'dependency_error'
    if isinstance(error, TimeoutError):
        return 'timeout'
    if 'nonfinite' in message:
        return 'numerical_divergence'
    if 'budget exceeded' in message or '64 mib input limit' in message:
        return 'budget_rejected'
    return 'software_rejected'


def milestone(folder, started, phase, **extra):
    with (folder/'milestones.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-started, phase=phase, **extra), allow_nan=False)+'\n')
        stream.flush()


def atlas_call(adapter, folder, phase, index):
    """Preflight current state once, then measure the complete public execute."""
    from atlas_adapter import admission, failure_status
    before = time.perf_counter_ns()
    initial = adapter.case.get('initial')
    admitted = admission(adapter.trainer.plan, adapter.trainer.state, adapter.case['inputs'], adapter.case['labels'],
                         operation='train', initial=initial)
    record = dict(phase=phase, index=index, operation='train', admission=admitted,
                  admission_ns=time.perf_counter_ns()-before, status=admitted['status'],
                  optimizer_step_before=adapter.trainer.state['step'])
    # This survives a timeout inside the public call and retains prior admitted
    # steps when a later full-precision optimizer-state JSON becomes too large.
    with (folder/'atlas-admissions.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False)+'\n')
        stream.flush()
    if admitted['status'] != 'admitted':
        return record
    before = time.perf_counter_ns()
    try:
        result = adapter.trainer.execute(adapter.case['inputs'], adapter.case['labels'], operation='train', initial=initial)
        record.update(public_api_ns=time.perf_counter_ns()-before, status='executed', result=result,
                      tape_count_matches_runtime=result['tape_bytes'] == admitted['exact_native_tape_bytes'])
    except Exception as error:
        record.update(public_api_ns=time.perf_counter_ns()-before, status=failure_status(error),
                      error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
    return record


def atlas_worker(case, args, report, started):
    import numpy as np
    import oracle
    from atlas_adapter import AtlasAdapter
    folder = args.output
    cold = time.perf_counter_ns()
    adapter = AtlasAdapter(case, ROOT/'snapshot/brian2-rust', ROOT/'runtime/b2-train')
    report['constructor_ns'] = time.perf_counter_ns()-cold
    report['runtime_sha256'] = digest(ROOT/'runtime/b2-train')
    weights = [np.asarray(w, dtype=np.float64) for w in case['weights']]
    first = [np.zeros_like(w) for w in weights]
    second = [np.zeros_like(w) for w in weights]
    report['qualification_steps'] = []
    for count in range(1, 4):
        milestone(folder, started, 'qualification_public_call', step=count)
        raw = atlas_call(adapter, folder, 'qualification', count)
        metadata = {key: value for key, value in raw.items() if key != 'result'}
        if raw['status'] != 'executed':
            write_json(folder/f'qualification-step-{count}.json', metadata)
            report.update(status=raw['status'], rejected_or_failed_step=count,
                          qualification_status='incomplete', passed_qualification_steps=count-1)
            report['qualification_steps'].append(metadata)
            return
        actual = raw['result']
        arrays = dict(loss=actual['loss'], logits=actual['logits'], spikes=actual['spikes'],
                      final_state=actual['final_membrane'], initial_vjp=actual['initial_gradients'],
                      gradients=actual['gradients'], weights=actual['state']['weights'],
                      first=actual['state']['first_moment'], second=actual['state']['second_moment'])
        metadata['artifact'] = save_arrays(folder, f'qualification-step-{count}', arrays)
        milestone(folder, started, 'qualification_oracle', step=count)
        ref = oracle.forward_vjp(case, weights)
        checks = {key: check(arrays[key], ref[key], key == 'spikes') for key in ('loss', 'logits', 'spikes', 'initial_vjp')}
        checks['final_state'] = check(arrays['final_state'], ref['states'][:, -1])
        checks.update(bank_checks(arrays['gradients'], ref['gradients'], 'gradient'))
        weights, first, second = oracle.adam(weights, ref['gradients'], first, second, count)
        for key, expected in [('weights', weights), ('first', first), ('second', second)]:
            checks.update(bank_checks(arrays[key], expected, key))
        checks['optimizer_counter'] = dict(passed=actual['state']['step'] == count)
        checks['native_tape_count'] = dict(passed=raw['tape_count_matches_runtime'])
        metadata['checks'] = checks
        metadata['qualified'] = all(c['passed'] for c in checks.values())
        write_json(folder/f'qualification-step-{count}.json', metadata)
        report['qualification_steps'].append(metadata)
        if not metadata['qualified']:
            report.update(status='semantic_mismatch', qualification_status='failed')
            return
        del raw, actual, arrays, ref
    report['qualification_status'] = 'passed_three_actual_public_Adam_updates'
    del weights, first, second, adapter
    gc.collect()
    adapter = AtlasAdapter(case, ROOT/'snapshot/brian2-rust', ROOT/'runtime/b2-train')
    report['performance_run'] = True
    with (folder/'raw-steps.jsonl').open('x') as log:
        for index in range(60):
            phase = 'warmup' if index < 10 else 'measured'
            result = atlas_call(adapter, folder, phase, index)
            row = {key: value for key, value in result.items() if key != 'result'}
            if result['status'] == 'executed':
                value = float(result['result']['loss'])
                row['loss'] = value if math.isfinite(value) else {'nonfinite': repr(value)}
                row['optimizer_step'] = result['result']['state']['step']
            log.write(json.dumps(row, allow_nan=False)+'\n')
            log.flush()
            if result['status'] != 'executed':
                report.update(status=result['status'], failed_performance_index=index)
                return
            if not math.isfinite(value):
                report['status'] = 'numerical_divergence'
                return
            if not result['tape_count_matches_runtime']:
                report['status'] = 'semantic_mismatch'
                return
            if index >= 10:
                report['measured_seconds'].append(result['public_api_ns']/1e9)
            # The public return includes full spikes, gradients and optimizer
            # state. Do not keep this previous return alive across the next API
            # call; only the trainer's necessary current state remains live.
            del result
    report.update(status='completed', median_s=statistics.median(report['measured_seconds']))


def competitor_worker(case, args, report, started):
    import numpy as np
    from benchmark_competitor import torch_setup, jax_setup
    import oracle
    folder = args.output
    cold = time.perf_counter()
    if args.engine in ('spyx', 'brainx_state'):
        step, reset, metadata = jax_setup(case, args.engine)
    else:
        step, reset, metadata = torch_setup(case, args.engine, args.compile, args.layerwise)
    report['implementation'] = metadata
    if hasattr(step, 'performance_mode'):
        step.performance_mode()
    report['qualification_steps'] = []
    ref0 = None
    for count in range(1, 4):
        milestone(folder, started, 'qualification_actual_public_call', step=count)
        before = time.perf_counter()
        loss = step()
        elapsed = time.perf_counter()-before
        if count == 1:
            report['cold_construction_and_first_update_s'] = time.perf_counter()-cold
        loss = float(loss)
        if not math.isfinite(loss):
            raise ValueError('nonfinite public loss during qualification')
        actual = {key: [np.array(bank, copy=True) for bank in banks] for key, banks in step.state().items()}
        actual['loss'] = loss
        artifact = save_arrays(folder, f'actual-public-step-{count}', actual)
        milestone(folder, started, 'qualification_oracle', step=count)
        if count == 1:
            weights = [np.asarray(w, dtype=np.float64) for w in case['weights']]
            first = [np.zeros_like(w) for w in weights]
            second = [np.zeros_like(w) for w in weights]
        ref = oracle.forward_vjp(case, weights)
        if count == 1:
            ref0 = ref
        checks = dict(loss=check(loss, ref['loss']))
        weights, first, second = oracle.adam(weights, ref['gradients'], first, second, count)
        for key, expected in [('weights', weights), ('first', first), ('second', second)]:
            checks.update(bank_checks(actual[key], expected, key))
        row = dict(step=count, actual_public_s=elapsed, artifact=artifact, checks=checks,
                   qualified=all(c['passed'] for c in checks.values()))
        write_json(folder/f'qualification-step-{count}.json', row)
        report['qualification_steps'].append(row)
        if not row['qualified']:
            report.update(status='semantic_mismatch', qualification_status='failed_actual_public_path')
            return
        del actual
    # Native graph diagnostics are a separate call after the unprewarmed actual
    # path has been tested; this call is not a performance sample.
    milestone(folder, started, 'qualification_diagnostics')
    reset()
    raw = step(True)
    checks = {key: check(raw[key], ref0[key], key == 'spikes')
              for key in ('loss', 'logits', 'states', 'spikes', 'initial_vjp')}
    checks.update(bank_checks(raw['gradients'], ref0['gradients'], 'gradient'))
    diagnostic = dict(checks=checks, artifact=save_arrays(folder, 'first-step-diagnostics', raw))
    write_json(folder/'diagnostics.json', diagnostic)
    report['diagnostic_checks'] = checks
    if not all(c['passed'] for c in checks.values()):
        report.update(status='semantic_mismatch', qualification_status='failed_diagnostics')
        return
    report['qualification_status'] = 'passed_three_actual_public_Adam_updates_and_full_diagnostics'
    del raw, ref0, ref, weights, first, second
    reset()
    if hasattr(step, 'performance_mode'):
        step.performance_mode()
    gc.collect()
    report['performance_run'] = True
    milestone(folder, started, 'performance')
    with (folder/'raw-steps.jsonl').open('x') as log:
        for index in range(60):
            before = time.perf_counter()
            loss = step()
            elapsed = time.perf_counter()-before
            scalar = float(loss)
            row = dict(index=index, phase='warmup' if index < 10 else 'measured', elapsed_s=elapsed,
                       loss=scalar if math.isfinite(scalar) else {'nonfinite': repr(scalar)})
            log.write(json.dumps(row, allow_nan=False)+'\n')
            log.flush()
            if not math.isfinite(scalar):
                report['status'] = 'numerical_divergence'
                return
            if index >= 10:
                report['measured_seconds'].append(elapsed)
    report.update(status='completed', median_s=statistics.median(report['measured_seconds']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True, choices=['atlas', 'snntorch_fp64', 'spikingjelly_frontier', 'spyx', 'brainx_state'])
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--layerwise', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('E1-large computation is restricted to the user-selected 100.90.28.27 host')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(schema='e1-large-worker-v1', engine=args.engine, seed=args.seed, compile=args.compile,
                  layerwise=args.layerwise, status='unqualified', performance_run=False, measured_seconds=[],
                  warmup_steps=10, measured_steps=50, per_seed_budget_s=360, aggregate_per_view_budget_s=1800,
                  budget_scope='supervisor wall includes interpreter, imports, fixture I/O, exact admission, qualification, oracle, serialization, compilation, and timing',
                  timing='complete coordinator public API; qualification/oracle/exact preflight and evidence I/O excluded from API time, included in case budget',
                  resource_scope='same disclosed environment settings as E1-small; OMP=1 does not constrain the JAX worker pool to one core; supervisor RSS covers the whole worker including qualification and compiler/diagnostic caches, not training-only memory',
                  no_microbatch=True, rule=RULE)
    files = [Path(__file__), ROOT/'tools/generate_dense_large.py', ROOT/'tools/benchmark_competitor.py',
             ROOT/'adapters/atlas_adapter.py', ROOT/'adapters/torch_adapter.py', ROOT/'adapters/jax_adapter.py', ROOT/'adapters/oracle.py']
    before_hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    try:
        manifest_path = ROOT/'fixtures/e1-large/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['rule'] != RULE:
            raise ValueError('E1-large frozen generation rule changed')
        case_path = manifest_path.parent/f'seed-{args.seed}.json'
        if digest(case_path) != manifest['files'][case_path.name]['sha256']:
            raise ValueError('Common-array checksum differs')
        case = json.loads(case_path.read_text())
        report.update(arrays_sha256=digest(case_path), fixture_manifest_sha256=digest(manifest_path), identities=before_hashes)
        milestone(args.output, started, 'fixture_loaded')
        if args.engine == 'atlas':
            atlas_worker(case, args, report, started)
        else:
            competitor_worker(case, args, report, started)
    except Exception as error:
        report.update(status=classify(error), error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
    finally:
        report['wall_s'] = time.monotonic()-started
        report['identities_unchanged'] = before_hashes == {str(path.relative_to(ROOT)): digest(path) for path in files}
        if not report['identities_unchanged']:
            report.update(status='unqualified', invalidation='Implementation changed during execution')
        write_json(args.output/'result.json', report)
        print(json.dumps(dict(engine=args.engine, seed=args.seed, status=report['status'], wall_s=report['wall_s'])), flush=True)
    return 0 if report['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
