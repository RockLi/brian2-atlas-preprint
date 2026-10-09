"""Qualify actual competitor graph cells against independent NumPy VJP.

Run this on the authorized 100.90.28.27 evaluation host. Never overwrites a
previous output directory. This reports qualification, not benchmark speed.
"""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapters'))
from atlas_graph_qualification import graph_fixture, forward_vjp


def compare(actual, expected, exact=False):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return dict(passed=False, actual_shape=list(a.shape), expected_shape=list(b.shape))
    delta = np.abs(a - b)
    return dict(passed=bool(np.array_equal(a, b) if exact else np.all(delta <= 1e-10 + 1e-8 * np.abs(b))),
                max_abs=float(delta.max(initial=0)), shape=list(b.shape))


def serializable(value):
    """Preserve failed numeric evidence explicitly without invalid JSON tokens."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if isinstance(value, dict):
        return {key: serializable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(child) for child in value]
    return value


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(child) for child in value)
    return True


def check_forward(raw, ref):
    checks = {key: compare(raw[key], ref[key], key == 'spikes')
              for key in ('loss', 'logits', 'states', 'spikes', 'initial_vjp')}
    checks['gradient_bank_count'] = dict(passed=len(raw['gradients']) == len(ref['gradients']))
    for i, (a, b) in enumerate(zip(raw['gradients'], ref['gradients'])):
        checks[f'gradient_{i}'] = compare(a, b)
    return checks


def qualify(case, engine, compiled):
    from graph_competitor import run_case
    raw = run_case(case, engine, steps=3, compiled=compiled)
    ref = forward_vjp(case)
    checks = check_forward(raw, ref)
    for i, (w, g, update) in enumerate(zip(case['weights'], ref['gradients'], raw['one_sgd_update'])):
        checks[f'sgd_{i}'] = compare(update, np.asarray(w) - .001 * g)
    local = raw['local_surrogate_vjp']
    margin, cotangent = np.asarray(local['margins']), np.asarray(local['cotangents'])
    checks['surrogate_forward'] = compare(local['values'], (margin > 0).astype(float), True)
    checks['surrogate_vjp'] = compare(local['vjp'], cotangent / (1 + 5 * np.abs(margin)) ** 2)
    weights = [np.asarray(w, dtype=np.float64) for w in case['weights']]
    first = [np.zeros_like(w) for w in weights]
    second = copy.deepcopy(first)
    checks['adam_update_count'] = dict(passed=len(raw['updates']) == 3)
    for step, update in enumerate(raw['updates'], 1):
        ref = forward_vjp(case, weights)
        checks.update({f'adam_{step}_{key}': value for key, value in check_forward(update, ref).items()})
        # Independent FP64 Adam recurrence; no framework optimizer is consulted.
        for i, gradient in enumerate(ref['gradients']):
            first[i] = .9 * first[i] + .1 * gradient
            second[i] = .999 * second[i] + .001 * gradient * gradient
            weights[i] = weights[i] - .001 * (first[i] / (1 - .9 ** step)) / (np.sqrt(second[i] / (1 - .999 ** step)) + 1e-8)
        for key, expected in [('weights', weights), ('first', first), ('second', second)]:
            checks[f'adam_{step}_{key}_bank_count'] = dict(passed=len(update[key]) == len(expected))
            for i, (a, b) in enumerate(zip(update[key], expected)):
                checks[f'adam_{step}_{key}_{i}'] = compare(a, b)
        checks[f'adam_{step}_counter'] = dict(passed=bool(np.all(np.asarray(update['step']) == step)))
    return raw, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=['snntorch_fp64', 'spikingjelly', 'spyx', 'brainx_state'], required=True)
    parser.add_argument('--compile', action='store_true', help='Compile Torch graph cell forward with fullgraph=True; JAX variants always JIT')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=['all', 'recurrent', 'shared', 'recurrent_boundary', 'shared_boundary'], default='all')
    args = parser.parse_args()
    dest = args.output
    dest.mkdir(parents=True, exist_ok=False)
    base = json.loads((ROOT / 'fixtures/q0.json').read_text())
    cases = {}
    for shared in (False, True):
        name = 'shared' if shared else 'recurrent'
        case = graph_fixture(base, shared=shared)
        cases[name] = case
        boundary = copy.deepcopy(case)
        boundary['initial'] = [[1/.95, 1/.95-1e-7, 1/.95+1e-7, -.5, 1/.95, 0],
                               [.2, -.3, 0, 1.8, .7, -.8]]
        cases[name + '_boundary'] = boundary
    if args.case != 'all':
        cases = {args.case: cases[args.case]}
    report = dict(engine=args.engine, compile=args.compile, schema='competitor-graph-qualification-v1',
                  scope='synchronous same-tick recurrent/tied graph, full states/spikes/loss, CE-induced VJP, local arbitrary-cotangent surrogate VJP, one SGD and three Adam steps',
                  oracle='independent NumPy analytic graph reverse VJP; no hard-spike finite difference',
                  not_yet_qualified=['delay', 'trainable dynamics', 'checkpoint restore', 'arbitrary external whole-network cotangent'],
                  atol=1e-10, rtol=1e-8, performance_run=False, cases={})
    before = time.perf_counter()
    for name, case in cases.items():
        artifact = dict(case=case)
        try:
            raw, checks = qualify(case, args.engine, args.compile)
            artifact.update(raw=raw, checks=checks)
            passed = all(check['passed'] for check in checks.values())
            status = 'qualified' if passed else ('semantic_mismatch' if finite(raw) else 'numerical_divergence')
            report['cases'][name] = dict(status=status, passed=passed,
                                         failed=[key for key, check in checks.items() if not check['passed']])
        except Exception as error:
            status = 'dependency_error' if isinstance(error, (ImportError, ModuleNotFoundError)) else 'software_rejected'
            if isinstance(error, MemoryError) or 'out of memory' in str(error).lower():
                status = 'oom'
            artifact.update(status=status, error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
            report['cases'][name] = dict(status=status, passed=False, error=str(error))
        (dest / f'{name}.json').write_text(json.dumps(serializable(artifact), indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(engine=args.engine, case=name, **report['cases'][name])), flush=True)
    report['qualification_status'] = 'qualified' if all(r['passed'] for r in report['cases'].values()) else 'unqualified'
    report['elapsed_s'] = time.perf_counter() - before
    files = [Path(__file__), ROOT/'adapters/graph_competitor.py', ROOT/'adapters/atlas_graph_qualification.py',
             ROOT/'adapters/torch_adapter.py', ROOT/'adapters/jax_adapter.py', ROOT/'fixtures/q0.json']
    report['identities'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    report['case_artifact_sha256'] = {name: hashlib.sha256((dest/f'{name}.json').read_bytes()).hexdigest() for name in cases}
    (dest/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('engine', 'qualification_status', 'elapsed_s')}), flush=True)
    return 0 if report['qualification_status'] == 'qualified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
