"""Remote-only second serial queue: bounded Q0 and exact admission, no ranking.

Waits for the existing M/F/E1-large queue and all its compute processes to exit.
Keeps source identities fixed while waiting; every child remains independently
responsible for its scientific outcome. No fixture reduction or retry occurs.
"""
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

from generate_dense_large import ROOT, digest
from run_recurrent_queue import launch

HOST = 'rock-mac-studio-1.local'
OUT = ROOT/'evidence/qualification-followup-r1'
PREREQUISITE = ROOT/'evidence/cpu-followup-r1/terminal.json'
MAX_WAIT_S = 21600
MAX_EXECUTION_S = 43200
RSS_GUARD = 64*1024**3
FILES = [
    'tools/run_qualification_followup_queue.py', 'tools/run_recurrent_queue.py',
    'tools/generate_dense_large.py', 'tools/generate_recurrent_cases.py',
    'tools/qualify_recurrent_case.py', 'tools/run_conv_queue.py',
    'tools/generate_conv_cases.py', 'tools/qualify_conv_case.py',
    'tools/qualify_e4.py', 'tools/qualify_metal_q0.py',
    'adapters/conv_competitor.py', 'adapters/graph_competitor.py',
    'adapters/atlas_graph_qualification.py', 'adapters/atlas_adapter.py',
    'adapters/torch_adapter.py', 'adapters/jax_adapter.py', 'adapters/oracle.py',
    'protocol/recurrent-fixture-r1.json', 'protocol/e2-convolution-r1.json',
    'protocol/finite-engine-cases.json', 'protocol/semantic-contracts.json',
    'protocol/execution-plan-r2.json', 'fixtures/q0.json',
    'evidence/h0r/contract.json', 'environment/cpu-lock.txt',
    'environment/jax-lock.txt', 'environment/hardware.json', 'runtime/b2-train',
]


def save(name, value):
    target = OUT/name
    temporary = target.with_suffix(target.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(target)


def conflicts():
    import psutil
    own = psutil.Process()
    excluded = {own.pid, *[p.pid for p in own.parents()]}
    tokens = {
        'run_cpu_followup_queue.py', 'run_qualification_followup_queue.py',
        'run_remote_benchmark_v2.py', 'benchmark_e1.py', 'benchmark_competitor.py',
        'benchmark_dense_large.py', 'run_dense_large_queue.py',
        'capability_migration.py', 'frontier_capability.py',
        'run_graph_phase.py', 'run_conv_queue.py', 'run_recurrent_queue.py',
        'qualify_conv_case.py', 'qualify_recurrent_case.py',
        'qualify_e4.py', 'qualify_metal_q0.py', 'b2-train',
    }
    found = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            argv = process.info['cmdline'] or []
            if process.pid not in excluded and any(Path(a).name in tokens for a in argv):
                # Native runner belongs to this frozen evaluation only.
                if not any(str(ROOT) in a for a in argv) and 'b2-train' in [Path(a).name for a in argv]:
                    continue
                found.append(dict(pid=process.pid, argv=argv))
        except psutil.Error:
            pass
    return found


def main():
    if socket.gethostname() != HOST:
        raise RuntimeError('All numerical work is restricted to the user-selected Mac Studio.')
    OUT.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    # Record native API/shader identities as well as each independent adapter.
    files = list(FILES)
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'snapshot/brian2-rust/python/brian2_rust').glob('training*')) if p.is_file()]
    hashes = {rel: digest(ROOT/rel) for rel in sorted(set(files))}
    save('freeze.json', dict(schema='qualification-followup-r1', host=HOST, files=hashes,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prerequisite=str(PREREQUISITE.relative_to(ROOT)), max_wait_s=MAX_WAIT_S,
        maximum_execution_s=MAX_EXECUTION_S, process_tree_rss_guard_bytes=RSS_GUARD,
        order=['E4-Q0', 'H0R-Metal-Q0', 'E2-Q0', 'E2-five-exact-admissions', 'R-45-exact-admissions'],
        performance_run=False,
        exclusions=['E2 full qualification/performance', 'R225 qualification/performance', 'A1 training', 'CUDA'],
        note='Fresh bounded capability/admission evidence. A worker exit is not a scientific pass.'))
    began = time.monotonic()
    while True:
        if PREREQUISITE.exists() and not conflicts():
            previous = json.loads(PREREQUISITE.read_text())
            names = [r.get('name') for r in previous.get('records', [])]
            if previous.get('status') != 'finite_followup_queue_exited' or 'e1-large' not in names:
                save('terminal.json', dict(status='prerequisite_incomplete', previous_status=previous.get('status'), records=[]))
                return 2
            save('barrier.json', dict(status='released', waited_s=time.monotonic()-began,
                prerequisite_sha256=digest(PREREQUISITE)))
            break
        if time.monotonic()-began >= MAX_WAIT_S:
            save('terminal.json', dict(status='prerequisite_pending', records=[], conflicts=conflicts()))
            return 2
        time.sleep(15)
    cpu = str(ROOT/'environment/cpu/bin/python')
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
        PYTHONPATH=str(ROOT/'snapshot')+':'+str(ROOT/'snapshot/brian2-rust/python'),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        JAX_PLATFORM_NAME='cpu', JAX_ENABLE_X64='true')
    execution_start = time.monotonic()
    records = []

    def run(name, arguments, cap, output):
        if any(digest(ROOT/rel) != value for rel, value in hashes.items()):
            raise RuntimeError('Frozen source changed; do not run the changed revision.')
        if conflicts():
            raise RuntimeError('Another evaluation compute process is active.')
        if shutil.disk_usage(ROOT).free < 50*1024**3:
            raise RuntimeError('50 GiB free-disk floor reached.')
        remaining = MAX_EXECUTION_S-(time.monotonic()-execution_start)
        if remaining < cap:
            raise RuntimeError('Insufficient remaining execution budget for the declared phase cap.')
        jobenv = env.copy()
        jobenv.update(TMPDIR=str(OUT/'tmp'/name),
            TORCHINDUCTOR_CACHE_DIR=str(OUT/'compiler-cache'/name/'torch'),
            JAX_COMPILATION_CACHE_DIR=str(OUT/'compiler-cache'/name/'jax'))
        Path(jobenv['TMPDIR']).mkdir(parents=True, exist_ok=False)
        print('START', name, flush=True)
        record = launch(name, [cpu, *arguments], cap, RSS_GUARD, OUT, ROOT/output, jobenv, 'child_suite')
        record.update(scientific_success='Read child numerical/admission reports; not inferred from exit status.')
        records.append(record)
        save('progress.json', records)
        print('END', name, record['exit_code'], record['termination_reason'], flush=True)
        return record['exit_code'] == 0 and record['termination_reason'] == 'exited'

    try:
        prepared = run('e4-prepare', ['tools/qualify_e4.py', 'prepare'], 60, 'evidence/e4')
        if prepared:
            checked = run('e4-selfcheck', ['tools/qualify_e4.py', 'selfcheck', '--run-id', 'r1'], 180, 'evidence/e4')
            if checked:
                run('e4-qualify', ['tools/qualify_e4.py', 'qualify', '--run-id', 'r1', '--allow-host', HOST], 660, 'evidence/e4/runs/r1')
        run('metal-q0', ['tools/qualify_metal_q0.py', 'run', '--allow-host', HOST, '--run-id', 'q0-r1', '--state-trace'], 660, 'evidence/h0r/runs/q0-r1')
        run('e2-q0', ['tools/run_conv_queue.py', '--phase', 'q0', '--output', 'evidence/e2-q0-v1'], 2200, 'evidence/e2-q0-v1')
        run('e2-admission', ['tools/run_conv_queue.py', '--phase', 'admission', '--output', 'evidence/e2-admission-v1'], 3400, 'evidence/e2-admission-v1')
        run('recurrent-admission', ['tools/run_recurrent_queue.py', '--phase', 'admission', '--output', 'evidence/recurrent-admission-v1'], 30600, 'evidence/recurrent-admission-v1')
    except Exception as error:
        save('terminal.json', dict(status='coordinator_stopped', error_type=type(error).__name__, error=str(error), records=records, complete_evaluation=False))
        raise
    save('terminal.json', dict(status='finite_qualification_queue_exited', records=records,
        performance_run=False, complete_evaluation=False,
        note='Inspect each child report. Original-model migration, A1/A2 and remaining finite obligations are not completed by this queue.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
