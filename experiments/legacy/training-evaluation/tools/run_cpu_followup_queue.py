"""Serialized remote continuation; never competes with the active E1-small run.

This coordinator records process completion separately from scientific success.
It preserves each child suite's logs and never changes fixtures or retries a
failed scientific cell. All installations use new private environments.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/cpu-followup-r1'
HOST = 'rock-mac-studio-1.local'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    temporary.replace(path)


def active_evaluation_jobs():
    import psutil
    own = psutil.Process()
    exempt = {own.pid, *[p.pid for p in own.parents()]}
    tokens = {'run_remote_benchmark_v2.py', 'benchmark_e1.py',
              'benchmark_competitor.py', 'benchmark_dense_large.py',
              'run_dense_large_queue.py', 'run_graph_phase.py'}
    found = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid in exempt:
            continue
        try:
            if any(Path(arg).name in tokens for arg in process.info['cmdline'] or []):
                found.append(process.info)
        except psutil.Error:
            pass
    return found


def launch(name, argv, cap, environment):
    import psutil
    print('START', name, flush=True)
    begin = time.monotonic()
    record = dict(name=name, argv=argv, cap_s=cap,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    descendants = {}
    with (OUT / (name + '.log')).open('x') as stream:
        proc = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        record['pid'] = proc.pid
        while proc.poll() is None:
            try:
                for child in psutil.Process(proc.pid).children(recursive=True):
                    descendants[(child.pid, child.create_time())] = child
            except psutil.Error:
                pass
            if time.monotonic() - begin >= cap:
                record['termination_reason'] = 'coordinator_phase_timeout'
                # Nested child suites own separate groups, so keep identities
                # while live and terminate only descendants of this exact job.
                for child in reversed(list(descendants.values())):
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(1)
        record['exit_code'] = proc.wait()
    leftovers = []
    for child in descendants.values():
        try:
            if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                leftovers.append(child.pid)
                child.kill()
        except psutil.Error:
            pass
    record.update(remaining_owned_descendants_cleaned=leftovers,
                  elapsed_s=time.monotonic() - begin,
                  termination_reason=record.get('termination_reason', 'exited'),
                  scientific_success='Read the child suite reports; exit code is not scientific completion.')
    save(name + '-terminal.json', record)
    print('END', name, record['exit_code'], record['termination_reason'], flush=True)
    return record


def main():
    if socket.gethostname() != HOST:
        raise RuntimeError('Run only on the user-authorized Mac Studio.')
    OUT.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    cpu = str(ROOT / 'environment/cpu/bin/python')
    uv = shutil.which('uv')
    if uv is None:
        raise RuntimeError('uv is unavailable in the remote PATH')
    scripts = ['capability_migration.py', 'frontier_capability.py',
               'generate_dense_large.py', 'benchmark_dense_large.py',
               'run_dense_large_queue.py', 'run_cpu_followup_queue.py']
    hashes = {f'tools/{name}': sha(ROOT / 'tools' / name) for name in scripts}
    for path in sorted((ROOT / 'evidence/frontier/dependencies').glob('*.lock')):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    hashes['evidence/frontier/F-HH1-contract.json'] = sha(ROOT / 'evidence/frontier/F-HH1-contract.json')
    save('freeze.json', dict(schema='cpu-followup-r1', host=HOST, files=hashes,
                            order=['original-Brian-smoke-12', 'F-HH1-three-private-environments', 'E1-large-35-slots'],
                            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            prerequisite='E1-small supervisor terminal with 35 records and no active E1 job',
                            max_barrier_wait_s=14400, minimum_free_disk_bytes=50*1024**3,
                            resource_note='Sequential evaluation compute only; ordinary host processes may remain. Requested OMP/BLAS/Torch one thread does not establish a one-core JAX budget.'))
    begin = time.monotonic()
    prerequisite = ROOT / 'evidence/remote-benchmark-v1/terminal.json'
    while True:
        if prerequisite.exists() and not active_evaluation_jobs():
            previous = json.loads(prerequisite.read_text())
            if not previous.get('supervisor_completed') or len(previous['records']) != 35:
                raise RuntimeError('E1-small finite denominator was not preserved')
            save('barrier.json', dict(status='released', waited_s=time.monotonic()-begin,
                                     prerequisite_sha256=sha(prerequisite)))
            break
        if time.monotonic()-begin > 14400:
            save('terminal.json', dict(status='prerequisite_pending', executed=[]))
            return 2
        time.sleep(10)
    if shutil.disk_usage(ROOT).free < 50*1024**3:
        save('terminal.json', dict(status='resource_pending', reason='50 GiB free-disk floor', executed=[]))
        return 2
    if any(sha(ROOT / rel) != value for rel, value in hashes.items()):
        raise RuntimeError('Frozen followup file changed while waiting')
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', UV_CACHE_DIR=str(ROOT/'cache/uv'),
               TMPDIR=str(ROOT/'cache/tmp'), MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    records = []
    def run(name, argv, cap, jobenv=None):
        if any(sha(ROOT / rel) != value for rel, value in hashes.items()):
            raise RuntimeError('Frozen followup source changed before launch')
        record = launch(name, argv, cap, env if jobenv is None else jobenv)
        records.append(record)
        save('progress.json', records)
        return record
    brian_env = env.copy()
    brian_env['PYTHONPATH'] = str(ROOT/'snapshot') + ':' + str(ROOT/'snapshot/brian2-rust/python')
    run('migration-smoke', [cpu, 'tools/capability_migration.py', 'smoke', '--run-id', 'original-r1',
                           '--timeout', '600', '--allow-host', HOST], 7500, brian_env)
    locks = {'jaxley': 'jaxley-py312-macos-arm64.lock',
             'braincell': 'braincell-py312-macos-arm64.lock',
             'brian2modelfitting': 'brian2modelfitting-published-py312.lock'}
    for engine, lock in locks.items():
        destination = ROOT / ('environment/f-' + engine)
        if destination.exists():
            raise FileExistsError('Refusing to modify a pre-existing frontier environment: '+str(destination))
        created = run('f-'+engine+'-venv', [uv, 'venv', '--python', '/opt/homebrew/bin/python3.12', str(destination)], 120)
        if created['exit_code'] != 0:
            continue
        interpreter = str(destination/'bin/python')
        installed = run('f-'+engine+'-install', [uv, 'pip', 'sync', '--python', interpreter,
                        str(ROOT/'evidence/frontier/dependencies'/lock), '--require-hashes'], 1200)
        if installed['exit_code'] != 0:
            continue
        run('f-'+engine+'-probe', [interpreter, 'tools/frontier_capability.py', 'run', '--engine', engine,
                                 '--run-id', 'r1', '--allow-host', HOST], 660)
    run('e1-large', [cpu, 'tools/run_dense_large_queue.py', '--output', 'evidence/remote-e1-large-v1',
                     '--q0-root', 'evidence/remote-qualify-v1'], 13500)
    save('terminal.json', dict(status='finite_followup_queue_exited', records=records,
                               complete_evaluation=False,
                               note='Each child suite requires separate outcome/evidence validation; other finite obligations remain.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
