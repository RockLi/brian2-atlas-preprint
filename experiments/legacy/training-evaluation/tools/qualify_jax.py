"""Standalone JAX dense Q0 validator against independent NumPy analytic VJP."""
import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapters'))
from oracle import forward_vjp, adam


def compare(actual, expected, exact=False):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return dict(passed=False, actual_shape=list(a.shape), expected_shape=list(b.shape))
    delta = np.abs(a - b)
    return dict(passed=bool(np.array_equal(a, b) if exact else np.all(delta <= 1e-10 + 1e-8 * np.abs(b))), max_abs=float(delta.max(initial=0)), shape=list(a.shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=['spyx', 'brainx_state'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'report.json').exists():
        raise FileExistsError(dest / 'report.json')
    base = json.loads((ROOT / 'fixtures/q0.json').read_text())
    cases = {'base_negative_count_input': base}
    duplicate = copy.deepcopy(base); duplicate['inputs'] *= 2; duplicate['labels'] *= 2
    cases['batch_duplicate'] = duplicate
    initial = copy.deepcopy(base)
    initial['initial'] = [[1/.95, 1/.95-1e-7, 1/.95+1e-7, -.5, 1/.95, 0], [.2, -.3, 0, 1.8, .7, -.8]]
    cases['initial_threshold_boundary'] = initial
    isolated = copy.deepcopy(base); isolated['inputs'] = isolated['inputs'][1:]; isolated['labels'] = isolated['labels'][1:]
    cases['single_sample_no_carry'] = isolated
    report = dict(engine=args.engine, scope='dense synchronous Q0, independent local surrogate VJP and one SGD/three Adam updates',
                  oracle='NumPy explicit analytic reverse VJP, independent from adapter; no hard-spike finite difference',
                  cases={}, not_yet_qualified=['recurrent', 'delay', 'tied_weights', 'trainable_dynamics', 'checkpoint_restore', 'full_network_arbitrary_spike_cotangent'],
                  atol=1e-10, rtol=1e-8, performance_run=False)
    start = time.perf_counter()
    for name, case in cases.items():
        try:
            from jax_adapter import run_case
            raw = run_case(case, args.engine, steps=3)
            ref = forward_vjp(case)
            checks = {key: compare(raw[key], ref[key], key == 'spikes') for key in ['loss', 'logits', 'states', 'spikes', 'initial_vjp']}
            for i, (a, b) in enumerate(zip(raw['gradients'], ref['gradients'])):
                checks[f'gradient_{i}'] = compare(a, b)
            for i, (w, g, update) in enumerate(zip(case['weights'], ref['gradients'], raw['one_sgd_update'])):
                checks[f'sgd_{i}'] = compare(update, np.asarray(w) - .001 * g)
            local = raw['local_surrogate_vjp']
            margin, cot = np.asarray(local['margins']), np.asarray(local['cotangents'])
            checks['local_surrogate_forward'] = compare(local['values'], (margin > 0).astype(float), True)
            checks['local_surrogate_vjp'] = compare(local['vjp'], cot / (1 + 5 * np.abs(margin)) ** 2)
            weights = [np.asarray(w) for w in case['weights']]
            first = [np.zeros_like(w) for w in weights]; second = copy.deepcopy(first)
            for step, update in enumerate(raw['updates'], 1):
                gradients = forward_vjp(case, weights)['gradients']
                weights, first, second = adam(weights, gradients, first, second, step)
                for key, expected in [('weights', weights), ('first', first), ('second', second)]:
                    for i, (a, b) in enumerate(zip(update[key], expected)):
                        checks[f'adam_{step}_{key}_{i}'] = compare(a, b)
                checks[f'adam_{step}_counter'] = dict(passed=update['step'] == step)
            (dest / (name + '.json')).write_text(json.dumps(dict(case=case, raw=raw, checks=checks), indent=2, allow_nan=False) + '\n')
            report['cases'][name] = dict(passed=all(x['passed'] for x in checks.values()), checks=checks)
        except Exception as error:
            report['cases'][name] = dict(passed=False, error=str(error), traceback=traceback.format_exc())
    report['dense_qualification_status'] = 'passed' if all(r['passed'] for r in report['cases'].values()) else 'failed'
    report['qualification_status'] = 'partial' if report['dense_qualification_status'] == 'passed' else 'unqualified'
    report['elapsed_s'] = time.perf_counter() - start
    report['identities'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'adapters/oracle.py',ROOT/'adapters/jax_adapter.py',ROOT/'fixtures/q0.json',Path(__file__)]}
    (dest / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'engine': args.engine, 'dense': report['dense_qualification_status'], 'overall': report['qualification_status'], 'elapsed_s': report['elapsed_s'], 'cases': {n: r['passed'] for n, r in report['cases'].items()}}))
    return 0 if report['dense_qualification_status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
