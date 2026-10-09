"""Five-seed Atlas/Torch MNIST queue after the remote qualification queue.

Training uses the already reviewed A1 v4 contract. Every fresh worker has one
1800s total budget, including qualification and final test. This queue never
retries a formal slot or changes the once-only held-out test ledger.
"""
import fcntl
import platform
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import time

from generate_dense_large import ROOT, digest
from run_recurrent_queue import launch
from run_extended_followup_r2 import active_evaluation_processes
from arm64_artifact_gate_r1 import gate_artifact

HOST = 'rock-mac-studio-1.local'
OUT = ROOT/'evidence/a1-queue-arm64-r1'
CONTRACT = ROOT/'protocol/A1-execution-addendum-arm64-r1.json'
PREREQUISITE = ROOT/'evidence/extended-followup-arm64-r5/terminal.json'
Q0 = ROOT/'evidence/remote-qualify-v1'
SEEDS = (11, 23, 37, 51, 71)
VIEWS = [('atlas', 'atlas', False), ('sj-layerwise', 'spikingjelly_frontier', False),
         ('snn-layerwise', 'snntorch_fp64', False), ('sj-compile', 'spikingjelly_frontier', True),
         ('snn-compile', 'snntorch_fp64', True)]
FILES = ['tools/run_a1_queue_arm64_r1.py', 'tools/run_a1_mnist_arm64_r1.py', 'tools/prepare_data.py',
         'tools/run_recurrent_queue.py', 'tools/run_qualification_followup_queue.py',
         'tools/generate_dense_large.py', 'tools/generate_recurrent_cases.py',
         'adapters/torch_adapter.py', 'adapters/atlas_adapter.py', 'adapters/oracle.py',
         'protocol/A1-execution-addendum-arm64-r1.json', 'protocol/recurrent-fixture-r1.json',
         'protocol/finite-engine-cases.json', 'runtime/arm64-r2/b2-train',
         'environment/cpu-lock.txt', 'environment/hardware.json']


def save(name, value):
    target = OUT/name
    temporary = target.with_suffix(target.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(target)


def a1_conflicts():
    return active_evaluation_processes()


def q0_report(view):
    return ROOT/'evidence/arm64-r2/q0/atlas/report.json' if view == 'atlas' else Q0/f'q0-{view}'/'report.json'


def main():
    if socket.gethostname() != HOST:
        raise RuntimeError('A1 runs only on the user-authorized remote Mac Studio.')
    if platform.machine() != 'arm64':raise RuntimeError('ARM64 Python required')
    family_lock=(ROOT/'evidence/extended-followup-r1.lock').open('a+')
    fcntl.flock(family_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runner=gate_artifact(ROOT/'runtime/arm64-r2/b2-train','runner','1dc6fac642e1829a942105ec6288fe24e5632f66ca54cf12a671258f6014e5a4')
    OUT.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    spec = json.loads(CONTRACT.read_text())
    if spec.get('status') != 'frozen_pretraining' or spec.get('implementation_sha256') != digest(ROOT/'tools/run_a1_mnist_arm64_r1.py'):
        raise RuntimeError('A1 contract is not frozen against this implementation.')
    files = [ROOT/rel for rel in FILES]
    files += [ROOT/p for p in ['tools/validate_a1_arm64_r1.py','tools/arm64_artifact_gate_r1.py','tools/run_extended_followup_r2.py','tools/build_arm64_runtime_r2.py','evidence/runtime-arm64-build-r2/terminal.json','evidence/runtime-arm64-preparation-r2/build-contract.json','evidence/h0r-metal-coregraphics-supervision-r1/terminal.json']]
    producer=json.loads((ROOT/'evidence/runtime-arm64-build-r2/terminal.json').read_text())
    if producer.get('source_integrity_after_exit') is not True or producer.get('runtime_sha256') != runner['sha256']:raise RuntimeError('Native producer identity/integrity mismatch')
    if producer.get('sources_final') != producer.get('sources_before'):raise RuntimeError('Native producer sources changed')
    for rel,h in producer['sources_final'].items():
        if digest(ROOT/rel) != h:raise RuntimeError('Native source changed: '+rel)
        files.append(ROOT/rel)
    files += [p for p in (ROOT/'snapshot/brian2-rust/python/brian2_rust').glob('training*') if p.is_file()]
    files += [q0_report(name) for name, _, _ in VIEWS]
    for name, _, _ in VIEWS:
        files += list(q0_report(name).parent.glob('*.json'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files))}
    plan = []
    for offset, seed in enumerate(SEEDS):
        for name, engine, compiled in VIEWS[offset:]+VIEWS[:offset]:
            plan.append(dict(view=name, engine=engine, compiled=compiled, seed=seed))
    save('freeze.json', dict(schema='a1-queue-arm64-r1', files=hashes, finite_slots=25, order=plan,
        runner=runner, python_architecture=platform.machine(), remote_budget_deadline_utc='2026-10-07T07:11:48.976559+00:00', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), host=HOST,
        wait_cap_s=72000, stage_execution_cap_s=48600, per_worker_total_wall_s=1800,
        per_slot_outer_supervision_s=1830, preparation_cap_s=300,
        held_out_policy='New ARM64 profile exclusive durable ledger with unchanged once-only policy, complete 10000-sample test once or no test score',
        resource_note='Requested one compute thread; no macOS core affinity. All evaluation jobs serialized.',
        exclusions=['JAX A1 adapter pending', 'HPO', 'SHD', 'CUDA']))
    start = time.monotonic()
    while True:
        if PREREQUISITE.exists() and not a1_conflicts():
            prior = json.loads(PREREQUISITE.read_text())
            if prior.get('status') != 'finite_extension_queue_exited':
                save('terminal.json', dict(status='prerequisite_incomplete', records=[], prior_status=prior.get('status')))
                return 2
            save('barrier.json', dict(status='released', waited_s=time.monotonic()-start,
                                     prerequisite_sha256=digest(PREREQUISITE)))
            break
        if datetime.datetime.now(datetime.timezone.utc).timestamp() >= datetime.datetime.fromisoformat('2026-10-07T07:11:48.976559+00:00').timestamp():raise RuntimeError('Remote reservation exhausted')
        if time.monotonic()-start >= 72000:
            save('terminal.json', dict(status='prerequisite_pending', records=[], conflicts=a1_conflicts()))
            return 2
        time.sleep(15)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT/'snapshot')+':'+str(ROOT/'snapshot/brian2-rust/python'),
        MPLCONFIGDIR=str(ROOT/'cache/matplotlib'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    cpu = str(ROOT/'environment/cpu/bin/python')
    base_command = [cpu, 'tools/run_a1_mnist_arm64_r1.py', '--root', str(ROOT), '--contract', str(CONTRACT)]
    execution_start = time.monotonic()
    records = []

    def guard():
        deadline=datetime.datetime.fromisoformat('2026-10-07T07:11:48.976559+00:00').timestamp()
        if time.time()+1830 > deadline:raise RuntimeError('Insufficient remaining remote reservation')
        gate_artifact(ROOT/'runtime/arm64-r2/b2-train','runner',runner['sha256'])
        if any(digest(ROOT/rel) != value for rel, value in hashes.items()):
            raise RuntimeError('Frozen A1 source/runtime/Q0 identities changed.')
        if a1_conflicts():
            raise RuntimeError('Another evaluation compute job is active.')
        if shutil.disk_usage(ROOT).free < 50*1024**3:
            raise RuntimeError('50 GiB free disk floor reached.')
        if time.monotonic()-execution_start > 48600-1830:
            raise RuntimeError('Stage execution cap has insufficient budget for another full slot.')

    try:
        guard()
        print('START shared-arrays', flush=True)
        prepared = launch('shared-arrays', [*base_command, '--prepare-shared'], 300, 64*1024**3,
                          OUT, ROOT/'fixtures/a1-arm64-r1', env, 'shared_array_preparation')
        save('preparation.json', prepared)
        if prepared.get('remaining_owned_processes'):raise RuntimeError('Preparation cleanup not verified')
        if prepared['exit_code'] != 0 or prepared['termination_reason'] != 'exited':
            raise RuntimeError('Common arrays were not completely prepared; no formal run launched.')
        for slot in plan:
            guard()
            job = f"{slot['view']}-seed-{slot['seed']}"
            output = OUT/job
            command = [*base_command, '--allow-run', '--engine', slot['engine'], '--seed', str(slot['seed']),
                       '--output', str(output), '--q0-report', str(q0_report(slot['view']))]
            if slot['compiled']:
                command.append('--compile')
            print('START', job, flush=True)
            row = launch(job, command, 1830, 64*1024**3, OUT, output, env, 'a1_supervisor')
            row.update(slot)
            if row.get('remaining_owned_processes'):raise RuntimeError('Slot cleanup not verified')
            terminal = output/'terminal.json'
            if terminal.exists():
                child = json.loads(terminal.read_text())
                row.update(child_status=child['status'], child_terminal_sha256=digest(terminal),
                    completed_epochs=child.get('completed_epochs'), test_status=child.get('test_status'))
                if child['status'] == 'cleanup_not_verified_do_not_launch_next_case':
                    records.append(row)
                    raise RuntimeError('Child cleanup was not verified; no next case launched.')
            else:
                row.update(child_status='no_terminal_report', scientific_success=False)
            records.append(row)
            save('progress.json', records)
            print('END', job, row['exit_code'], row['termination_reason'], row['child_status'], flush=True)
    except Exception as error:
        save('terminal.json', dict(status='coordinator_stopped', error_type=type(error).__name__, error=str(error),
            records=records, finite_slots=25, unexecuted_slots=plan[len(records):], complete_evaluation=False))
        raise
    save('terminal.json', dict(status='finite_a1_queue_exited', records=records, finite_slots=25,
        complete_evaluation=False, note='Independent evidence validation and per-seed quality/selection/test interpretation still required. JAX A1 and other obligations remain.'))
    if a1_conflicts():raise RuntimeError('Other evaluation active before audit')
    if time.time()+3600 > datetime.datetime.fromisoformat('2026-10-07T07:11:48.976559+00:00').timestamp():raise RuntimeError('Insufficient reservation for full audit')
    audit_command=[cpu,'tools/validate_a1_arm64_r1.py','--root',str(ROOT),'--mode','full','--output',str(OUT/'validation-full-r1.json')]
    audited=launch('full-audit',audit_command,3600,64*1024**3,OUT,OUT,env,'read_only_full_audit')
    save('audit-supervision.json',audited)
    return 0 if audited['exit_code']==0 and not audited.get('remaining_owned_processes') else 1


if __name__ == '__main__':
    raise SystemExit(main())
