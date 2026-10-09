"""Five bounded sequential matrix Q0 anchors, never a performance/full-R queue."""
import argparse
import json
import os
from pathlib import Path
import socket

from qualify_recurrent_matrix_v1 import ROOT, MANIFEST, dependencies, digest, write
from run_recurrent_queue import launch

VIEWS = [('atlas', 'cpu', 'atlas'), ('snntorch', 'cpu', 'snntorch_fp64'),
         ('spikingjelly', 'cpu', 'spikingjelly'), ('spyx', 'jax', 'spyx'), ('brainstate', 'jax', 'brainx_state')]


def no_overlap():
    import psutil
    own = psutil.Process()
    excluded = {own.pid, *[p.pid for p in own.parents()]}
    # Includes preparation and idle supervisors: the coordinator must establish
    # a fully serial phase. The current queue's own ancestors are allowed.
    names = {p.name for p in (ROOT/'tools').glob('*.py')
             if p.name.startswith(('run_', 'qualify_', 'benchmark_', 'generate_'))}
    names.update(['capability_migration.py', 'frontier_capability.py'])
    conflicts = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            argv = p.info['cmdline'] or []
            native = any(Path(arg).name == 'b2-train' for arg in argv) and any(str(ROOT) in arg for arg in argv)
            if p.pid not in excluded and (native or any(Path(arg).name in names for arg in argv)):
                conflicts.append(dict(pid=p.pid, command=p.info['cmdline']))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if conflicts:
        raise RuntimeError('Another evaluation task is present: '+json.dumps(conflicts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--storage', choices=['auto', 'dense'], default='auto')
    parser.add_argument('--torch-view', choices=['eager', 'fullgraph'], default='eager')
    args = parser.parse_args()
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('Matrix Q0 restricted to authorized 100.90.28.27')
    no_overlap()
    policy = json.loads(MANIFEST.read_text())
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    sources = [*dependencies(), Path(__file__), ROOT/'tools/run_recurrent_queue.py']
    frozen = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    write(folder/'freeze.json', dict(schema='recurrent-matrix-q0-queue-v1', policy=policy, order=VIEWS,
                                    storage=args.storage, torch_view=args.torch_view, identities=frozen,
                                    performance_run=False, total_slots=5, per_slot_s=300, maximum_slot_s=1500,
                                    scope='small [2,4,2] B2T32 anchors only; no R225 or other full graph execution',
                                    resource_disclosure='OMP settings do not imply one-core JAX; RSS includes complete diagnostics and compiler caches'))
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(ROOT/'cache/matplotlib'), JAX_PLATFORM_NAME='cpu',
               JAX_ENABLE_X64='true', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    records = []
    atlas_ok = False
    for name, environment, engine in VIEWS:
        no_overlap()
        if {str(path.relative_to(ROOT)): digest(path) for path in sources} != frozen:
            records.append(dict(view=name, status='not_executed', reason='Source/runtime/manifest changed'))
            continue
        if engine != 'atlas' and not atlas_ok:
            records.append(dict(view=name, status='not_executed', reason='Exact same-case Atlas Q0 gate failed'))
            continue
        output = folder/name
        jobenv = env.copy()
        jobenv.update(TORCHINDUCTOR_CACHE_DIR=str(folder/'compiler-cache'/name/'torch'),
                      JAX_COMPILATION_CACHE_DIR=str(folder/'compiler-cache'/name/'jax'), TMPDIR=str(folder/'tmp'/name))
        Path(jobenv['TMPDIR']).mkdir(parents=True, exist_ok=False)
        command = [str(ROOT/f'environment/{environment}/bin/python'), str(ROOT/'tools/qualify_recurrent_matrix_v1.py'),
                   '--engine', engine, '--storage', args.storage, '--output', str(output)]
        if engine != 'atlas':
            command.extend(['--atlas-reference', str(folder/'atlas')])
        if args.torch_view == 'fullgraph' and engine in ('snntorch_fp64', 'spikingjelly'):
            command.append('--compile')
        print('START', name, args.storage, args.torch_view, flush=True)
        row = launch(name, command, 300, 64*1024**3, folder, output, jobenv, 'interpreter_imports')
        row['view'] = name
        path = output/'result.json'
        if row['termination_reason'] in ('timeout', 'resource_limit'):
            row['status'] = row['termination_reason']
        elif path.exists():
            result = json.loads(path.read_text())
            row.update(status=result['status'], result_sha256=digest(path), result_path=str(path))
            if row['exit_code'] != 0 and row['status'] == 'qualified':
                row.update(status='unqualified', reason='Qualified JSON but nonzero process exit')
        else:
            row.update(status='software_rejected', reason='No terminal result; exit code alone is not OOM')
        if engine == 'atlas':
            atlas_ok = row['status'] == 'qualified' and row['exit_code'] == 0
        records.append(row)
        print('END', name, row['status'], row.get('termination_phase', {}), flush=True)
    write(folder/'terminal.json', dict(schema='recurrent-matrix-q0-queue-v1', supervisor_completed=True,
                                      records=records, all_qualified=all(row['status']=='qualified' for row in records),
                                      elapsed_slot_s=sum(row.get('elapsed_s', 0) for row in records), performance_run=False))
    return 0 if all(row['status']=='qualified' for row in records) else 2


if __name__ == '__main__':
    raise SystemExit(main())
