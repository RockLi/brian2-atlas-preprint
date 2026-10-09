"""Bounded E2 stages: small tied-conv Q0, five exact admissions, qualification.

There is deliberately no performance stage in this queue. Native conv operators
must pass Q0 before the optional five-seed full-size qualification can run.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import time

from generate_conv_cases import ROOT, CONTRACT, SEEDS, digest
from qualify_conv_case import dependencies
from run_recurrent_queue import launch, assert_no_other_phase, write

VIEWS = [('atlas', 'cpu', 'atlas', []), ('snntorch', 'cpu', 'snntorch_fp64', []),
         ('spikingjelly', 'cpu', 'spikingjelly', ['--compile']),
         ('spyx', 'jax', 'spyx', []), ('brainstate', 'jax', 'brainx_state', [])]


def no_e2_overlap():
    import psutil
    own = psutil.Process()
    excluded = {own.pid, *[p.pid for p in own.parents()]}
    names = {'run_conv_queue.py', 'qualify_conv_case.py', 'generate_conv_cases.py'}
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            if p.pid not in excluded and p.info['cmdline'] and any(Path(arg).name in names for arg in p.info['cmdline']):
                raise RuntimeError('Another E2 job is still running; do not overlap compute: '+str(p.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def q0_gate(qroot, view):
    name, _, engine, flags = view
    path = qroot/f'{name}-seed-11/result.json'
    if not path.exists():
        return dict(qualified=False, reason='Missing matching true-conv Q0', path=str(path))
    report = json.loads(path.read_text())
    changed = [rel for rel, h in report.get('identities', {}).items() if not (ROOT/rel).is_file() or digest(ROOT/rel) != h]
    passed = report.get('status') == 'qualified' and report.get('mode') == 'q0' and report.get('engine') == engine and bool(report.get('compile')) == ('--compile' in flags)
    return dict(qualified=passed and not changed, changed=changed, report_sha256=digest(path),
                reason=None if passed and not changed else 'True-conv Q0 failed or implementation differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['q0', 'admission', 'full-qualification'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--q0-root', type=Path, default=ROOT/'evidence/e2-q0-v1')
    args = parser.parse_args()
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('E2 execution is restricted to 100.90.28.27')
    assert_no_other_phase()
    no_e2_overlap()
    policy = json.loads(CONTRACT.read_text())
    if policy['status'] != 'frozen_before_fixture_generation_or_E2_execution':
        raise ValueError('E2 contract is not frozen')
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    views = VIEWS[:1] if args.phase == 'admission' else VIEWS
    seeds = (11,) if args.phase == 'q0' else SEEDS
    scale = 'q0' if args.phase == 'q0' else 'full'
    order = []
    for offset, seed in enumerate(seeds):
        for view in views[offset % len(views):]+views[:offset % len(views)]:
            order.append(dict(seed=seed, view=view))
    source_files = [*dependencies(), Path(__file__), ROOT/'tools/run_recurrent_queue.py',
                    ROOT/'snapshot/brian2-rust/python/brian2_rust/training_graph.py']
    frozen = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
    gates = {view[0]: q0_gate(args.q0_root, view) for view in views} if args.phase == 'full-qualification' else {}
    write(folder/'freeze.json', dict(schema='e2-convolution-queue-v1', phase=args.phase, order=order, policy=policy,
                                    identities=frozen, gates=gates, performance_run=False,
                                    per_seed_s=360, per_view_s=1800, preparation_per_seed_s=300,
                                    resource_disclosure='same E1 settings; JAX pool is not confined to one core; whole-worker RSS contains diagnostics and compiler caches'))
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(ROOT/'cache/matplotlib'), JAX_PLATFORM_NAME='cpu',
               JAX_ENABLE_X64='true', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    cpu = str(ROOT/'environment/cpu/bin/python')
    guard = policy['execution']['process_tree_rss_guard_bytes']
    prepared, records, prep_records = {}, [], []
    spent = {view[0]: 0. for view in views}
    for item in order:
        seed = item['seed']
        view = item['view']
        name, environment, engine, flags = view
        if {str(path.relative_to(ROOT)): digest(path) for path in source_files} != frozen:
            records.append(dict(seed=seed, view=name, status='unqualified', reason='Frozen implementation changed'))
            continue
        if seed not in prepared:
            prep_name = f'prepare-{scale}-seed-{seed}'
            result = launch(prep_name, [cpu, str(ROOT/'tools/generate_conv_cases.py'), '--scale', scale, '--seed', str(seed)],
                            300, guard, folder, folder/prep_name, env, 'fixture_generation_and_serialization')
            prep_records.append(result)
            prepared[seed] = result['exit_code'] == 0 and result['termination_reason'] == 'exited'
        if not prepared[seed]:
            records.append(dict(seed=seed, view=name, status='not_executed', reason='Frozen common fixture preparation failed'))
            continue
        if args.phase == 'full-qualification' and not gates[name]['qualified']:
            records.append(dict(seed=seed, view=name, status='unqualified', reason=gates[name]))
            continue
        cap = min(360, 1800-spent[name])
        if cap <= 0:
            records.append(dict(seed=seed, view=name, status='timeout', phase='not_launched', reason='Per-view aggregate qualification budget exhausted'))
            continue
        job = f'{name}-seed-{seed}'
        jobenv = env.copy()
        jobenv.update(TORCHINDUCTOR_CACHE_DIR=str(folder/'compiler-cache'/job/'torch'),
                      JAX_COMPILATION_CACHE_DIR=str(folder/'compiler-cache'/job/'jax'), TMPDIR=str(folder/'tmp'/job))
        Path(jobenv['TMPDIR']).mkdir(parents=True, exist_ok=False)
        command = [str(ROOT/f'environment/{environment}/bin/python'), str(ROOT/'tools/qualify_conv_case.py'),
                   '--mode', args.phase, '--engine', engine, '--seed', str(seed), '--output', str(folder/job), *flags]
        print('START', args.phase, job, flush=True)
        row = launch(job, command, cap, guard, folder, folder/job, jobenv, 'interpreter_imports')
        spent[name] += row['elapsed_s']
        row.update(seed=seed, view=name)
        result_path = folder/job/'result.json'
        if row['termination_reason'] in ('timeout', 'resource_limit'):
            row['status'] = row['termination_reason']
        elif result_path.exists():
            result = json.loads(result_path.read_text())
            row.update(status=result['status'], result_sha256=digest(result_path), result_path=str(result_path))
        else:
            row.update(status='software_rejected', reason='Worker exited without terminal result; no automatic OOM inference')
        records.append(row)
        write(folder/'progress.json', dict(records=records, preparation=prep_records, spent_seconds=spent), replace=True)
        print('END', job, row['status'], row.get('termination_phase', {}), flush=True)
    write(folder/'terminal.json', dict(supervisor_completed=True, finite_slots=len(order), phase=args.phase,
                                      records=records, preparation=prep_records, spent_seconds=spent, performance_run=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
