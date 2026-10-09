"""Bounded graph qualifications on the specifically authorized remote host."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('Run this qualification on the user-selected 100.90.28.27 host')
    folder = ROOT / 'evidence/remote-graph-qualification-v1'
    folder.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
               JAX_PLATFORM_NAME='cpu', JAX_ENABLE_X64='true', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', TORCHINDUCTOR_CACHE_DIR=str(ROOT/'cache/graph-v1/torchinductor'),
               JAX_COMPILATION_CACHE_DIR=str(ROOT/'cache/graph-v1/jax'))
    jobs = [('snntorch_fp64', 'cpu', []), ('spikingjelly', 'cpu', ['--compile']),
            ('spyx', 'jax', []), ('brainx_state', 'jax', [])]
    files = ['tools/qualify_graph_competitor.py', 'tools/run_graph_phase.py', 'adapters/graph_competitor.py',
             'adapters/atlas_graph_qualification.py', 'adapters/torch_adapter.py', 'adapters/jax_adapter.py', 'fixtures/q0.json']
    def identities():
        return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    frozen = identities()
    (folder/'freeze.json').write_text(json.dumps(dict(host=socket.gethostname(), jobs=jobs, identities=frozen,
                                                    timeout_per_engine_s=300, performance_run=False), indent=2)+'\n')
    results = []
    for engine, environment, flags in jobs:
        command = [str(ROOT/f'environment/{environment}/bin/python'), str(ROOT/'tools/qualify_graph_competitor.py'),
                   '--engine', engine, '--output', str(folder/engine), *flags]
        print('START', engine, flush=True)
        start = time.monotonic()
        record = dict(engine=engine, command=command, timeout_s=300,
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        with (folder/f'{engine}.log').open('x') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = proc.pid
            try:
                record['exit_code'] = proc.wait(timeout=300)
                record['termination_reason'] = 'exited'
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                record['exit_code'] = proc.wait()
                record['termination_reason'] = 'timeout'
        record['elapsed_s'] = time.monotonic()-start
        record['identities_unchanged'] = identities() == frozen
        report = folder/engine/'report.json'
        if report.exists():
            record['qualification_report'] = json.loads(report.read_text())
        (folder/f'{engine}-terminal.json').write_text(json.dumps(record, indent=2)+'\n')
        results.append(record)
        (folder/'progress.json').write_text(json.dumps(results, indent=2)+'\n')
        print('END', engine, record['exit_code'], record['termination_reason'], flush=True)
    (folder/'terminal.json').write_text(json.dumps(dict(supervisor_completed=True, results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
