"""Sequential admission-first or 225-slot bounded recurrent qualification queue.

This queue is diagnostic only. Timeout/guard classification names its phase.
Native Atlas starts its own session, so cleanup follows descendant identities
(PID plus create_time), never assumes one worker process group contains them.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from generate_recurrent_cases import ROOT, SEEDS, CONTRACT, CASE_SOURCE, cases, paths, contract, digest

VIEWS = [('atlas', 'cpu', 'atlas', []), ('snntorch', 'cpu', 'snntorch_fp64', []),
         ('spikingjelly', 'cpu', 'spikingjelly', ['--compile']),
         ('spyx', 'jax', 'spyx', []), ('brainstate', 'jax', 'brainx_state', [])]


def write(path, value, replace=False):
    with Path(path).open('w' if replace else 'x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def latest_phase(output, fallback):
    path = output/'phases.jsonl'
    if not path.exists():
        return dict(phase=fallback)
    try:
        rows = path.read_text().splitlines()
        return json.loads(rows[-1]) if rows else dict(phase=fallback)
    except (OSError, json.JSONDecodeError):
        return dict(phase=fallback, phase_tail_incomplete=True)


def assert_no_other_phase():
    import psutil
    parent = psutil.Process()
    excluded = {parent.pid, *[p.pid for p in parent.parents()]}
    scripts = {'run_recurrent_queue.py', 'qualify_recurrent_case.py', 'generate_recurrent_cases.py',
               'benchmark_e1.py', 'benchmark_competitor.py', 'benchmark_dense_large.py', 'run_dense_large_queue.py',
               'run_remote_benchmark_v2.py', 'run_remote_phase.py', 'run_graph_phase.py',
               'qualify_dense.py', 'qualify_jax.py', 'qualify_graph_competitor.py', 'generate_dense_large.py'}
    conflicts = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            if p.pid not in excluded and p.info['cmdline'] and any(Path(arg).name in scripts for arg in p.info['cmdline']):
                conflicts.append(dict(pid=p.pid, command=p.info['cmdline']))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    if conflicts:
        raise RuntimeError('Another evaluation phase is running: '+json.dumps(conflicts))


def launch(name, command, timeout, memory_guard, folder, output, env, fallback_phase):
    import psutil
    record = dict(name=name, command=command, timeout_s=timeout, rss_guard_bytes=memory_guard,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    started = time.monotonic()
    tracked = {}
    peak = 0

    def remember(process):
        try:
            created = process.create_time()
            identity = (process.pid, created)
            if identity not in tracked:
                # A newly observed child with a reused PID gets a distinct
                # identity; historical process objects never authorize killing
                # the replacement. same_live() also uses psutil.is_running(),
                # which verifies the object identity against the live PID.
                if any(item['pid'] == process.pid for item in tracked.values()):
                    record.setdefault('pid_reuse_observed', []).append(process.pid)
                tracked[identity] = dict(pid=process.pid, create_time=created, process=process)
        except psutil.Error:
            pass

    def same_live(item):
        try:
            p = item['process']
            return p.create_time() == item['create_time'] and p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    def capture_tree(root):
        remember(root)
        # Retain identities after reparenting; descendants can start new
        # sessions or outlive the worker that originally owned them.
        for item in list(tracked.values()):
            if same_live(item):
                try:
                    for child in item['process'].children(recursive=True):
                        remember(child)
                except psutil.Error:
                    pass

    def stop_owned(root):
        # Stop the owned tree before the final census, so its members cannot
        # keep creating new children while we terminate their parents.
        for _ in range(4):
            before = len(tracked)
            capture_tree(root)
            for item in list(tracked.values()):
                if same_live(item):
                    try:
                        item['process'].suspend()
                    except psutil.Error:
                        pass
            capture_tree(root)
            if len(tracked) == before:
                break
        killed = []
        for item in sorted(tracked.values(), key=lambda entry: entry['pid'] == root.pid):
            if same_live(item):
                try:
                    item['process'].kill()
                    killed.append(dict(pid=item['pid'], create_time=item['create_time']))
                except psutil.Error:
                    pass
        # Popen owns the direct child's wait status; letting psutil reap that
        # PID first can make Popen lose the signal exit code. Only descendants
        # are waited here, then the caller records proc.wait() for the root.
        psutil.wait_procs([item['process'] for item in tracked.values() if item['pid'] != root.pid], timeout=3)
        record['terminated_owned_identities'] = killed

    with (folder/f'{name}.log').open('x') as log, (folder/f'{name}-resources.jsonl').open('x') as samples:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        root_process = psutil.Process(proc.pid)
        remember(root_process)
        while proc.poll() is None:
            capture_tree(root_process)
            rss = {}
            threads = {}
            for item in list(tracked.values()):
                if same_live(item):
                    try:
                        rss[str(item['pid'])] = item['process'].memory_info().rss
                        threads[str(item['pid'])] = item['process'].num_threads()
                    except psutil.Error:
                        pass
            total = sum(rss.values())
            peak = max(peak, total)
            samples.write(json.dumps(dict(elapsed_s=time.monotonic()-started, rss_by_pid=rss, threads_by_pid=threads,
                                          aggregate_rss=total), separators=(',', ':'))+'\n')
            samples.flush()
            reason = 'resource_limit' if total > memory_guard else ('timeout' if time.monotonic()-started >= timeout else None)
            if reason:
                record.update(termination_reason=reason, termination_phase=latest_phase(output, fallback_phase),
                              interpretation='qualification/supervision boundary in the recorded phase; not engine performance or physical OOM')
                stop_owned(root_process)
                break
            time.sleep(.05)
        record['exit_code'] = proc.wait()
    # Normal parent exit can still leave native descendants in other sessions.
    capture_tree(root_process)
    if any(same_live(item) for item in tracked.values()):
        stop_owned(root_process)
    remaining = [dict(pid=item['pid'], create_time=item['create_time']) for item in tracked.values() if same_live(item)]
    record.update(elapsed_s=time.monotonic()-started, termination_reason=record.get('termination_reason', 'exited'),
                  process_identities=[dict(pid=item['pid'], create_time=item['create_time']) for item in tracked.values()],
                  remaining_owned_processes=remaining, peak_contemporaneous_rss=peak,
                  rss_scope='whole qualification tree, includes oracle/diagnostic conversion/compiler caches; not training-only memory')
    write(folder/f'{name}-terminal.json', record)
    if remaining:
        raise RuntimeError('Owned descendants remain after cleanup; do not start the next case: '+json.dumps(remaining))
    return record


def q0_gate(view, graph_q0, atlas_q0):
    name, _, engine, flags = view
    if engine == 'atlas':
        if not atlas_q0.exists():
            return dict(qualified=False, reason='Missing prior Atlas graph Q0', path=str(atlas_q0))
        report = json.loads(atlas_q0.read_text())
        results = report.get('results', [])
        if len(results) < 2 or any(row.get('status') != 'qualified' for row in results):
            return dict(qualified=False, reason='Atlas recurrent/tied Q0 not qualified')
        if report.get('oracle_sha256') != digest(ROOT/'adapters/atlas_graph_qualification.py'):
            return dict(qualified=False, reason='Atlas graph oracle changed')
        for row in results:
            execution = row.get('execution', {})
            if execution.get('adapter_sha256') != digest(ROOT/'adapters/atlas_adapter.py') or execution.get('runner_sha256') != digest(ROOT/'runtime/b2-train'):
                return dict(qualified=False, reason='Atlas graph adapter/runtime changed')
        return dict(qualified=True, report_sha256=digest(atlas_q0))
    path = graph_q0/engine/'report.json'
    if not path.exists():
        return dict(qualified=False, reason='Missing matching competitor graph Q0', path=str(path))
    report = json.loads(path.read_text())
    changed = [rel for rel, h in report.get('identities', {}).items() if not (ROOT/rel).is_file() or digest(ROOT/rel) != h]
    passed = report.get('qualification_status') == 'qualified' and report.get('engine') == engine and bool(report.get('compile')) == ('--compile' in flags)
    return dict(qualified=passed and not changed, changed=changed, report_sha256=digest(path),
                reason=None if passed and not changed else 'Graph Q0 or implementation identity mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['admission', 'qualify'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--admission-root', type=Path, default=ROOT/'evidence/recurrent-admission-v1')
    parser.add_argument('--graph-q0', type=Path, default=ROOT/'evidence/remote-graph-qualification-v1')
    parser.add_argument('--atlas-q0', type=Path, default=ROOT/'evidence/remote-qualify-v1/atlas-graph.json')
    args = parser.parse_args()
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('Recurrent evaluation is restricted to the user-selected 100.90.28.27 host')
    assert_no_other_phase()
    policy = contract()
    memory_guard = policy['queue']['supervisor_rss_guard_bytes']
    specs = cases()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    views = VIEWS[:1] if args.phase == 'admission' else VIEWS
    plan = []
    for case_id in specs:
        for offset, seed in enumerate(SEEDS):
            for view in views[offset % len(views):]+views[:offset % len(views)]:
                plan.append(dict(case_id=case_id, seed=seed, view=view))
    files = [CONTRACT, CASE_SOURCE, Path(__file__), ROOT/'tools/generate_recurrent_cases.py', ROOT/'tools/generate_dense_large.py', ROOT/'tools/qualify_recurrent_case.py',
             ROOT/'adapters/graph_competitor.py', ROOT/'adapters/atlas_graph_qualification.py', ROOT/'adapters/atlas_adapter.py',
             ROOT/'adapters/torch_adapter.py', ROOT/'adapters/jax_adapter.py', ROOT/'adapters/oracle.py', ROOT/'runtime/b2-train']
    identities = {str(path.relative_to(ROOT)): digest(path) for path in files}
    gates = {view[0]: q0_gate(view, args.graph_q0, args.atlas_q0) for view in views} if args.phase == 'qualify' else {}
    write(folder/'freeze.json', dict(schema='recurrent-queue-v1', phase=args.phase, finite_slots=len(plan), plan=plan,
                                    identities=identities, gates=gates, policy=policy,
                                    environment_locks={p.name: digest(p) for p in (ROOT/'environment').glob('*lock.txt')},
                                    hardware_sha256=digest(ROOT/'environment/hardware.json'), performance_run=False))
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(ROOT/'cache/matplotlib'), JAX_PLATFORM_NAME='cpu',
               JAX_ENABLE_X64='true', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    cpu = str(ROOT/'environment/cpu/bin/python')
    records, preparation = [], []
    spent = {(case_id, view[0]): 0. for case_id in specs for view in views}
    fixtures = {}
    for entry in plan:
        case_id, seed, view = entry['case_id'], entry['seed'], entry['view']
        name, environment, engine, flags = view
        key = (case_id, seed)
        job = f'{case_id}-seed-{seed}-{name}'
        if {str(path.relative_to(ROOT)): digest(path) for path in files} != identities:
            records.append(dict(case_id=case_id, seed=seed, view=name, status='unqualified', reason='Frozen implementation changed'))
            continue
        if key not in fixtures:
            path, manifest_path = paths(case_id, seed)
            prep_name = f'prepare-{case_id}-seed-{seed}'
            prep_start = time.monotonic()
            if not manifest_path.exists():
                print('PREPARE', prep_name, flush=True)
                result = launch(prep_name, [cpu, str(ROOT/'tools/generate_recurrent_cases.py'), '--case-id', case_id, '--seed', str(seed)],
                                300, memory_guard, folder, folder/prep_name, env, 'fixture_generation')
                preparation.append(result)
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                fixtures[key] = manifest if digest(path) == manifest['arrays_sha256'] and manifest['identity']['contract_sha256'] == digest(CONTRACT) else None
            else:
                fixtures[key] = None
            preparation.append(dict(name=prep_name+'-verification', elapsed_s=time.monotonic()-prep_start,
                                    status='verified' if fixtures[key] else 'unavailable', accounting='shared preparation plus verification wall; do not add twice to launch record'))
        manifest = fixtures[key]
        if manifest is None:
            records.append(dict(case_id=case_id, seed=seed, view=name, status='not_executed', reason='Frozen common fixture unavailable'))
            continue
        if args.phase == 'qualify' and not gates[name]['qualified']:
            records.append(dict(case_id=case_id, seed=seed, view=name, status='unqualified', reason=gates[name]))
            continue
        prior = args.admission_root/f'{case_id}-seed-{seed}-atlas'/'result.json'
        if args.phase == 'qualify' and engine == 'atlas' and prior.exists():
            rejected = json.loads(prior.read_text())
            if rejected.get('status') == 'budget_rejected' and rejected.get('arrays_sha256') == manifest['arrays_sha256'] and rejected.get('identities') == {str(path.relative_to(ROOT)): digest(path) for path in [ROOT/'tools/qualify_recurrent_case.py', CONTRACT, CASE_SOURCE, ROOT/'tools/generate_recurrent_cases.py', ROOT/'tools/generate_dense_large.py', ROOT/'adapters/graph_competitor.py', ROOT/'adapters/atlas_graph_qualification.py', ROOT/'adapters/atlas_adapter.py', ROOT/'adapters/torch_adapter.py', ROOT/'adapters/jax_adapter.py', ROOT/'adapters/oracle.py', ROOT/'runtime/b2-train']}:
                records.append(dict(case_id=case_id, seed=seed, view=name, status='budget_rejected', phase='atlas_exact_admission',
                                    reused_initial_admission_sha256=digest(prior), reused_initial_admission_path=str(prior),
                                    interpretation='exact initial Atlas software rejection retained; does not skip any competing engine'))
                continue
        cap = min(360, 1800-spent[(case_id, name)])
        if cap <= 0:
            records.append(dict(case_id=case_id, seed=seed, view=name, status='timeout', phase='not_launched', reason='Per-case-view aggregate budget exhausted'))
            continue
        jobenv = env.copy()
        jobenv.update(TORCHINDUCTOR_CACHE_DIR=str(folder/'compiler-cache'/job/'torch'),
                      JAX_COMPILATION_CACHE_DIR=str(folder/'compiler-cache'/job/'jax'),
                      TMPDIR=str(folder/'tmp'/job))
        Path(jobenv['TMPDIR']).mkdir(parents=True, exist_ok=False)
        command = [str(ROOT/f'environment/{environment}/bin/python'), str(ROOT/'tools/qualify_recurrent_case.py'),
                   '--case-id', case_id, '--seed', str(seed), '--engine', engine, '--mode', args.phase,
                   '--output', str(folder/job), *flags]
        print('START', job, args.phase, flush=True)
        row = launch(job, command, cap, memory_guard, folder, folder/job, jobenv, 'interpreter_imports')
        spent[(case_id, name)] += row['elapsed_s']
        row.update(case_id=case_id, seed=seed, view=name)
        result_path = folder/job/'result.json'
        if row['termination_reason'] in ('timeout', 'resource_limit'):
            row['status'] = row['termination_reason']
        elif result_path.exists():
            result = json.loads(result_path.read_text())
            row.update(status=result['status'], result_sha256=digest(result_path), result_path=str(result_path))
        else:
            row.update(status='software_rejected', termination_phase=latest_phase(folder/job, 'interpreter_imports'), reason='Worker exited without terminal report; not automatically classified OOM')
        records.append(row)
        write(folder/'progress.json', dict(records=records, preparation=preparation,
                                          spent_seconds={f'{case_id}/{name}': value for (case_id, name), value in spent.items()}), replace=True)
        print('END', job, row['status'], row.get('termination_phase', {}), flush=True)
    write(folder/'terminal.json', dict(supervisor_completed=True, finite_slots=len(plan), records=records, preparation=preparation,
                                      spent_seconds={f'{case_id}/{name}': value for (case_id, name), value in spent.items()}, performance_run=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
