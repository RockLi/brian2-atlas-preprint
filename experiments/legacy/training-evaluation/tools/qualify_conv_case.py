"""E2 true-conv Q0/full-size qualification and exact Atlas admission only."""
import argparse
import copy
import json
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace

from generate_conv_cases import ROOT, CONTRACT, SEEDS, digest
from qualify_recurrent_case import archive, compare, compare_forward, compare_banks, checks_status, mark as record_phase, write, atlas as qualify_atlas_full_graph
sys.path.insert(0, str(ROOT/'adapters'))
PHASE_ROOT = None


def mark(folder, start, phase, **details):
    record_phase(folder, start, phase, **details)
    if PHASE_ROOT is not None and Path(folder) != PHASE_ROOT:
        record_phase(PHASE_ROOT, start, phase, **{'variant': Path(folder).name, **details})


def native_q0(case):
    from atlas_adapter import AtlasAdapter
    adapter = AtlasAdapter(case, ROOT/'snapshot/brian2-rust', ROOT/'runtime/b2-train')
    updates = []
    for count in range(1, 4):
        trace = adapter.state_trace()
        if trace['status'] != 'executed':
            return dict(status=trace['status'], failure=dict(step=count, trace=trace))
        execution = adapter.call('train')
        if execution['status'] != 'executed':
            return dict(status=execution['status'], failure=dict(step=count, execution=execution))
        result = execution['result']
        updates.append(dict(loss=result['loss'], logits=result['logits'], states=trace['states'], spikes=result['spikes'],
                            gradients=result['gradients'], initial_vjp=result['initial_gradients'],
                            weights=result['state']['weights'], first=result['state']['first_moment'], second=result['state']['second_moment'],
                            step=result['state']['step'], elapsed_s=execution['public_api_ns']/1e9,
                            tape_count_matches_runtime=execution['tape_count_matches_runtime']))
        adapter.records.clear()
    out = {key: updates[0][key] for key in ('loss', 'logits', 'states', 'spikes', 'gradients', 'initial_vjp')}
    sgd = AtlasAdapter(case, ROOT/'snapshot/brian2-rust', ROOT/'runtime/b2-train', optimizer='sgd').call('train')
    if sgd['status'] != 'executed':
        return dict(status=sgd['status'], failure=dict(sgd=sgd))
    out.update(engine='atlas', version='frozen source/runtime', status='executed', updates=updates,
               one_sgd_update=sgd['result']['state']['weights'],
               identity=dict(projection='explicit shared OIHW graph parameter bank', state_observation='bounded T=16 native public prefix replay before every Adam update',
                             local_surrogate_observability='CE-induced whole-network VJP only; no external arbitrary-cotangent native API'),
               timing_role='small Q0 diagnostics including prefix replay; no performance score')
    return out


def qualify(case, args, report, start):
    import numpy as np
    from atlas_graph_qualification import forward_vjp
    from oracle import adam
    mark(args.output, start, 'engine_adapter_execution_including_diagnostic_conversion')
    if args.engine == 'atlas':
        raw = native_q0(case)
        if raw['status'] != 'executed':
            report.update(status=raw['status'], native_failure=raw)
            return
    else:
        from conv_competitor import run_case
        raw = run_case(case, args.engine, compiled=args.compile, steps=3)
    report['implementation'] = {key: raw[key] for key in ('engine', 'version', 'torch', 'jax', 'optax', 'parameter_dtype', 'device', 'compile', 'identity', 'qualification_cold_s', 'timing_role') if key in raw}
    mark(args.output, start, 'oracle_validation', step=0)
    reference = forward_vjp(case)
    checks = compare_forward(raw, reference)
    checks.update(compare_banks(raw['one_sgd_update'], [np.asarray(w)-.001*g for w, g in zip(case['weights'], reference['gradients'])], 'sgd'))
    if 'local_surrogate_vjp' in raw:
        local = raw['local_surrogate_vjp']
        margin, cot = np.asarray(local['margins']), np.asarray(local['cotangents'])
        checks['surrogate_forward'] = compare(local['values'], (margin > 0).astype(float), True)
        checks['surrogate_vjp'] = compare(local['vjp'], cot/(1+5*np.abs(margin))**2)
    weights = [np.asarray(w, dtype=np.float64) for w in case['weights']]
    first = [np.zeros_like(w) for w in weights]
    second = [np.zeros_like(w) for w in weights]
    checks['three_adam_steps'] = dict(passed=len(raw['updates']) == 3)
    for count, update in enumerate(raw['updates'], 1):
        mark(args.output, start, 'oracle_validation', step=count)
        if count != 1:
            reference = forward_vjp(case, weights)
        checks.update({f'adam{count}_{key}': value for key, value in compare_forward(update, reference).items()})
        weights, first, second = adam(weights, reference['gradients'], first, second, count)
        for key, expected in [('weights', weights), ('first', first), ('second', second)]:
            checks.update(compare_banks(update[key], expected, f'adam{count}_{key}'))
        checks[f'adam{count}_counter'] = dict(passed=bool(np.all(np.asarray(update['step']) == count)))
        if 'tape_count_matches_runtime' in update:
            checks[f'adam{count}_native_tape_formula'] = dict(passed=update['tape_count_matches_runtime'])
    report['checks'] = checks
    write(args.output/'checks.json', dict(complete_numeric_checks=True, archival_complete=False, checks=checks))
    fields = ('loss', 'logits', 'states', 'spikes', 'gradients', 'initial_vjp')
    mark(args.output, start, 'artifact_save', step=0)
    artifacts = [archive(args.output, 'initial-qualification', {key: raw[key] for key in fields})]
    artifacts.append(archive(args.output, 'one-sgd-update', {'weights': raw['one_sgd_update']}))
    if 'local_surrogate_vjp' in raw:
        write(args.output/'local-surrogate-vjp.json', raw['local_surrogate_vjp'])
    for count, update in enumerate(raw['updates'], 1):
        mark(args.output, start, 'artifact_save', step=count)
        artifacts.append(archive(args.output, f'adam-step-{count}', {key: update[key] for key in (*fields, 'weights', 'first', 'second')}))
    report.update(status=checks_status(checks), artifacts=artifacts,
                  qualification_scope='all states/spikes/logits/loss, initial VJP, every shared kernel/dense gradient, native SGD and three complete native Adam steps',
                  parameter_counts=[len(w) for w in case['weights']], logical_edge_counts=[len(p['sources']) for p in case['projections']])


def dependencies():
    return [Path(__file__), CONTRACT, ROOT/'tools/generate_conv_cases.py', ROOT/'tools/generate_dense_large.py',
            ROOT/'tools/qualify_recurrent_case.py', ROOT/'tools/generate_recurrent_cases.py',
            ROOT/'adapters/conv_competitor.py', ROOT/'adapters/atlas_adapter.py', ROOT/'adapters/atlas_graph_qualification.py',
            ROOT/'adapters/torch_adapter.py', ROOT/'adapters/jax_adapter.py', ROOT/'adapters/oracle.py', ROOT/'runtime/b2-train',
            ROOT/'environment/cpu-lock.txt', ROOT/'environment/jax-lock.txt', ROOT/'environment/hardware.json']


def main():
    global PHASE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['q0', 'admission', 'full-qualification'], required=True)
    parser.add_argument('--engine', choices=['atlas', 'snntorch_fp64', 'spikingjelly', 'spyx', 'brainx_state'], required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('E2 computation is restricted to 100.90.28.27')
    if args.mode == 'admission' and args.engine != 'atlas':
        parser.error('Exact Atlas admission is not a competitor capacity claim')
    if args.mode == 'q0' and args.seed != 11:
        parser.error('Frozen Q0 seed is 11; formal full cases retain all five seeds')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    PHASE_ROOT = args.output
    start = time.monotonic()
    report = dict(schema='e2-convolution-qualification-v1', mode=args.mode, engine=args.engine, seed=args.seed, compile=args.compile,
                  status='unqualified', performance_run=False, variants={},
                  interpretation='Diagnostic qualification/admission only. Oracle, diagnostic conversion or save timeout is not an engine performance/capacity result.')
    frozen = {str(path.relative_to(ROOT)): digest(path) for path in dependencies()}
    try:
        mark(args.output, start, 'fixture_read_and_verify')
        scale = 'q0' if args.mode == 'q0' else 'full'
        path = ROOT/f'fixtures/e2-conv-r1/{scale}/seed-{args.seed}.json'
        manifest_path = path.with_name(f'seed-{args.seed}.manifest.json')
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity']['contract_sha256'] != digest(CONTRACT) or manifest['arrays_sha256'] != digest(path):
            raise ValueError('Frozen E2 common arrays changed')
        case = json.loads(path.read_text())
        report.update(arrays_sha256=digest(path), manifest_sha256=digest(manifest_path), counts=manifest['counts'])
        if args.mode == 'q0':
            variants = {'zero_initial': case}
            boundary = copy.deepcopy(case)
            pattern = [1/.95, 1/.95-1e-7, 1/.95+1e-7, -.5, .2, 0.]
            boundary['initial'] = [[pattern[(i+sample)%len(pattern)] for i in range(sum(case['sizes'][1:]))] for sample in range(len(case['labels']))]
            variants['nonzero_threshold_boundary'] = boundary
            for name, variant in variants.items():
                child = args.output/name
                child.mkdir(exist_ok=False)
                write(child/'case.json', variant)
                mark(args.output, start, 'q0_variant', variant=name, phase_source=str(child/'phases.jsonl'))
                result = dict(engine=args.engine, variant=name, performance_run=False)
                settings = SimpleNamespace(**{**vars(args), 'output': child})
                qualify(variant, settings, result, start)
                result['case_sha256'] = digest(child/'case.json')
                write(child/'result.json', result)
                report['variants'][name] = result
            report['status'] = 'qualified' if all(result['status']=='qualified' for result in report['variants'].values()) else 'unqualified'
        elif args.engine == 'atlas':
            settings = SimpleNamespace(**{**vars(args), 'mode': 'admission' if args.mode == 'admission' else 'qualify'})
            qualify_atlas_full_graph(case, settings, report, start)
            report['scope_note'] = 'Full-size native final membrane only; all-state tied conv Q0 must pass separately. No full-shape prefix replay.'
        else:
            qualify(case, args, report, start)
    except Exception as error:
        from atlas_adapter import failure_status
        report.update(status='oom' if 'out of memory' in str(error).lower() else failure_status(error),
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        report['identities'] = frozen
        report['identities_unchanged'] = frozen == {str(path.relative_to(ROOT)): digest(path) for path in dependencies()}
        if not report['identities_unchanged']:
            report.update(status='unqualified', reason='Frozen implementation changed during run')
        report['elapsed_s'] = time.monotonic()-start
        mark(args.output, start, 'terminal_report_save', status=report['status'])
        write(args.output/'result.json', report)
        print(json.dumps({key: report[key] for key in ('mode', 'engine', 'seed', 'status', 'elapsed_s')}), flush=True)
    return 0 if report['status'] in ('qualified', 'admitted_initial_only', 'budget_rejected') else 2


if __name__ == '__main__':
    raise SystemExit(main())
