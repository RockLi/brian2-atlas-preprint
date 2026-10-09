"""Prepare and run the frozen seven-view E1-large queue, sequentially remotely.

Five full-size seeds per view; 360 seconds per seed and 1800 per view. A rejected
Atlas request stays budget_rejected and never turns into an OOM or reduced case.
Do not launch alongside the E1-small or other evaluation compute phases.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from generate_dense_large import ROOT, RULE, SEEDS, digest

VIEWS = [('atlas', 'cpu', 'atlas', []),
         ('sj-layerwise', 'cpu', 'spikingjelly_frontier', ['--layerwise']),
         ('snn-layerwise', 'cpu', 'snntorch_fp64', ['--layerwise']),
         ('spyx', 'jax', 'spyx', []),
         ('brainstate', 'jax', 'brainx_state', []),
         ('sj-compile', 'cpu', 'spikingjelly_frontier', ['--layerwise', '--compile']),
         ('snn-compile', 'cpu', 'snntorch_fp64', ['--layerwise', '--compile'])]


def write_json(path, value, *, replace=False):
    with Path(path).open('w' if replace else 'x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def no_competing_phase():
    import psutil
    own = psutil.Process()
    ancestors = {own.pid, *[p.pid for p in own.parents()]}
    names = ('benchmark_e1.py', 'benchmark_competitor.py', 'benchmark_dense_large.py',
             'run_dense_large_queue.py', 'run_remote_benchmark_v2.py', 'run_remote_phase.py',
             'run_graph_phase.py', 'qualify_dense.py', 'qualify_jax.py', 'qualify_graph_competitor.py')
    conflicts = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            if p.pid not in ancestors and p.info['cmdline'] and any(Path(arg).name in names for arg in p.info['cmdline']):
                conflicts.append(dict(pid=p.pid, command=p.info['cmdline']))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    if conflicts:
        raise RuntimeError('An evaluation compute phase is still active; wait before launching E1-large: '+json.dumps(conflicts))


def launch(name, command, cap, folder, env):
    import psutil
    started = time.monotonic()
    record = dict(name=name, command=command, timeout_s=cap,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    peak = 0
    max_threads = 0
    tracked_children = {}
    with (folder/f'{name}.log').open('x') as log, (folder/f'{name}-resources.jsonl').open('x') as samples:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env, start_new_session=True)
        record['pid'] = proc.pid
        while proc.poll() is None:
            if time.monotonic()-started >= cap:
                # Atlas native requests open their own session; killing only
                # the Python worker's group would leave that request running.
                try:
                    for child in psutil.Process(proc.pid).children(recursive=True):
                        tracked_children[(child.pid, child.create_time())] = child
                except psutil.Error:
                    pass
                for child in reversed(list(tracked_children.values())):
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                record['termination_reason'] = 'timeout'
                break
            try:
                parent = psutil.Process(proc.pid)
                family = [parent, *parent.children(recursive=True)]
                for child in family[1:]:
                    tracked_children[(child.pid, child.create_time())] = child
                rss = {str(p.pid): p.memory_info().rss for p in family if p.is_running()}
                threads = parent.num_threads()
                peak = max(peak, sum(rss.values()))
                max_threads = max(max_threads, threads)
                samples.write(json.dumps(dict(elapsed_s=time.monotonic()-started, aggregate_rss=sum(rss.values()),
                                              root_threads=threads, rss_by_pid=rss), separators=(',', ':'))+'\n')
                samples.flush()
            except psutil.Error:
                pass
            time.sleep(.05)
        record['exit_code'] = proc.wait()
    record.update(elapsed_s=time.monotonic()-started, termination_reason=record.get('termination_reason', 'exited'),
                  peak_contemporaneous_job_rss=peak, max_observed_root_threads=max_threads,
                  rss_scope='entire subprocess tree including qualification/oracle and retained compiler/diagnostic caches; not training-only memory or capacity evidence')
    remaining_descendants = []
    for child in tracked_children.values():
        try:
            if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                remaining_descendants.append(child.pid)
                child.kill()
        except psutil.Error:
            pass
    record['remaining_owned_descendants_cleaned'] = remaining_descendants
    # A compiler worker must not leak into the next view's timing. This check
    # only observes this subprocess group, and cleanup targets that group only.
    remaining = []
    for p in psutil.process_iter(['pid']):
        try:
            if os.getpgid(p.pid) == proc.pid:
                remaining.append(p.pid)
        except (ProcessLookupError, PermissionError):
            continue
    record['remaining_own_group_at_exit'] = remaining
    if remaining:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        record['cleaned_up_own_group'] = True
    write_json(folder/f'{name}-terminal.json', record)
    return record


def qualification_gate(qroot, view):
    name, _, engine, flags = view
    path = qroot/f'q0-{name}/report.json'
    if not path.exists():
        return dict(qualified=False, reason='Missing matching remote Q0', path=str(path))
    report = json.loads(path.read_text())
    if report.get('dense_qualification_status') != 'passed':
        return dict(qualified=False, reason='Matching Q0 did not pass', report_sha256=digest(path))
    changed = [rel for rel, h in report['identities'].items() if not (ROOT/rel).is_file() or digest(ROOT/rel) != h]
    if changed:
        return dict(qualified=False, reason='Qualified implementation changed', changed=changed, report_sha256=digest(path))
    if report.get('engine') != engine:
        return dict(qualified=False, reason='Q0 engine differs from requested view', actual=report.get('engine'), expected=engine)
    if engine not in ('atlas', 'spyx', 'brainx_state'):
        sample = json.loads((path.parent/'base_negative_count_input.json').read_text())
        if bool(sample['raw'].get('compile')) != ('--compile' in flags):
            return dict(qualified=False, reason='Q0 compile implementation differs from requested view')
    return dict(qualified=True, report_sha256=digest(path), path=str(path), scope='matching remote dense Q0 plus per-seed large three-Adam qualification required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'evidence/remote-e1-large-v1')
    parser.add_argument('--q0-root', type=Path, default=ROOT/'evidence/remote-qualify-v1')
    args = parser.parse_args()
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('E1-large execution is restricted to the user-selected 100.90.28.27 host')
    no_competing_phase()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    qroot = args.q0_root.resolve()
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(ROOT/'cache/matplotlib'), JAX_PLATFORM_NAME='cpu', JAX_ENABLE_X64='true',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    plan = []
    for offset, seed in enumerate(SEEDS):
        for view in VIEWS[offset:]+VIEWS[:offset]:
            plan.append(dict(seed=seed, view=view))
    gates = {view[0]: qualification_gate(qroot, view) for view in VIEWS}
    scripts = ['tools/generate_dense_large.py', 'tools/benchmark_dense_large.py', 'tools/run_dense_large_queue.py',
               'tools/benchmark_competitor.py', 'adapters/atlas_adapter.py', 'adapters/torch_adapter.py', 'adapters/jax_adapter.py', 'adapters/oracle.py']
    hashes = {name: digest(ROOT/name) for name in scripts}
    write_json(folder/'freeze.json', dict(schema='e1-large-frozen-queue-v1', host=socket.gethostname(), rule=RULE, order=plan,
                                         scripts=hashes, gates=gates, runtime_sha256=digest(ROOT/'runtime/b2-train'),
                                         source_manifest_sha256=digest(ROOT/'sources/snapshot-manifest.json'),
                                         environment_locks={path.name: digest(path) for path in (ROOT/'environment').glob('*lock.txt')},
                                         hardware_sha256=digest(ROOT/'environment/hardware.json'),
                                         per_seed_budget_s=360, per_view_budget_s=1800, shared_fixture_preparation_cap_s=300,
                                         budget_scope='all subprocess wall including imports, qualification/oracle, preflight, serialization and compilation; shared preparation separately accounted',
                                         warmup_steps=10, measured_steps=50, no_microbatch=True, no_batch_or_time_reduction=True,
                                         cpu_resource='requested BLAS/OMP/Torch threads=1; JAX pool recorded and not constrained to one core; no common-single-thread, affinity, capacity-advantage or whole-machine exclusivity claim'))
    cpu = str(ROOT/'environment/cpu/bin/python')
    prep = launch('fixture-preparation', [cpu, str(ROOT/'tools/generate_dense_large.py')], 300, folder, env)
    records = []
    spent = {view[0]: 0. for view in VIEWS}
    manifest_path = ROOT/'fixtures/e1-large/manifest.json'
    if prep['exit_code'] != 0 or not manifest_path.exists():
        for entry in plan:
            records.append(dict(seed=entry['seed'], view=entry['view'][0], status='not_executed', reason='Shared fixture preparation failed'))
        write_json(folder/'terminal.json', dict(supervisor_completed=True, preparation=prep, records=records, spent_seconds=spent))
        return 2
    manifest_hash = digest(manifest_path)
    write_json(folder/'fixture-identity.json', dict(manifest_sha256=manifest_hash))
    for entry in plan:
        seed, view = entry['seed'], entry['view']
        name, environment, engine, flags = view
        gate = gates[name]
        if not gate['qualified']:
            records.append(dict(seed=seed, view=name, status='unqualified', reason=gate))
            write_json(folder/'progress.json', dict(records=records, spent_seconds=spent), replace=True)
            continue
        if {rel: digest(ROOT/rel) for rel in hashes} != hashes or digest(manifest_path) != manifest_hash:
            records.append(dict(seed=seed, view=name, status='unqualified', reason='Frozen scripts or fixture manifest changed after queue creation'))
            write_json(folder/'progress.json', dict(records=records, spent_seconds=spent), replace=True)
            continue
        remaining = 1800-spent[name]
        if remaining <= 0:
            records.append(dict(seed=seed, view=name, status='timeout', reason='Per-view 1800 second aggregate budget exhausted before launch'))
            write_json(folder/'progress.json', dict(records=records, spent_seconds=spent), replace=True)
            continue
        job = f'{name}-seed-{seed}'
        command = [str(ROOT/f'environment/{environment}/bin/python'), str(ROOT/'tools/benchmark_dense_large.py'),
                   '--engine', engine, '--seed', str(seed), '--output', str(folder/job), *flags]
        jobenv = env.copy()
        # Every independent seed gets a new cache directory and an unprewarmed
        # actual public path, including compilation failures and compilation time.
        jobenv.update(TORCHINDUCTOR_CACHE_DIR=str(folder/'compiler-cache'/job/'torch'),
                      JAX_COMPILATION_CACHE_DIR=str(folder/'compiler-cache'/job/'jax'),
                      TRITON_CACHE_DIR=str(folder/'compiler-cache'/job/'triton'), TMPDIR=str(folder/'tmp'/job))
        Path(jobenv['TMPDIR']).mkdir(parents=True, exist_ok=False)
        print('START', job, flush=True)
        record = launch(job, command, min(360, remaining), folder, jobenv)
        spent[name] += record['elapsed_s']
        record.update(seed=seed, view=name)
        result_path = folder/job/'result.json'
        if record['termination_reason'] == 'timeout':
            record['status'] = 'timeout'
            record['partial_evidence_directory'] = str(folder/job)
        elif result_path.exists():
            result = json.loads(result_path.read_text())
            record.update(status=result['status'], result_sha256=digest(result_path), result_path=str(result_path),
                          median_s=result.get('median_s'), performance_run=result.get('performance_run', False))
        else:
            record.update(status='software_rejected', reason='Worker exited without terminal result')
        records.append(record)
        write_json(folder/'progress.json', dict(records=records, spent_seconds=spent), replace=True)
        print('END', job, record['status'], record['elapsed_s'], flush=True)
    write_json(folder/'terminal.json', dict(supervisor_completed=True, preparation=prep, records=records,
                                          spent_seconds=spent, finite_denominator=len(plan),
                                          run_status='completed' if all(r['status']=='completed' for r in records) else 'finite_queue_exhausted_with_noncompletion_outcomes'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
