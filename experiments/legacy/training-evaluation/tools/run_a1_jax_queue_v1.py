"""Prepare-only by default; explicit serial A1 JAX phase, ten fixed slots.

The outer coordinator finishes v4, its closed full audit, the fixture seal and
any inserted E5/Rmatrix/cold stages before --serial-phase. Missing prerequisites
close all ten slots as not_launched; this queue never reruns those stages.
Every model worker enforces1800s internally; outer1830 is cleanup only.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
HOST='rock-mac-studio-1.local'
SEEDS=(11,23,37,51,71)
VIEWS=(('spyx','spyx'),('brainstate','brainx_state'))
RSS_GUARD=64*1024**3
DISK_FLOOR=50*1024**3
INNER_CAP=1800
OUTER_CAP=1830
STAGE_CAP=18600
QCASES=('base_negative_count_input','batch_duplicate','initial_threshold_boundary','single_sample_no_carry')
DEPENDENCIES=[
 'tools/run_a1_jax_queue_v1.py','tools/run_a1_mnist_jax_v1.py','adapters/a1_jax_adapter_v1.py',
 'adapters/jax_adapter.py','adapters/oracle.py','tools/prepare_data.py',
 'tools/run_recurrent_queue.py','tools/generate_recurrent_cases.py',
 'tools/run_a1_mnist_v4.py','protocol/A1-execution-addendum-v4.json',
 'protocol/recurrent-fixture-r1.json','protocol/finite-engine-cases.json',
 'tools/seal_a1_v4_fixture_receipt.py','tools/validate_a1_v4.py',
 'runtime/b2-train','environment/cpu-lock.txt','environment/jax-lock.txt','environment/hardware.json',
]
class GateError(RuntimeError):pass

def read(path):return json.loads(Path(path).read_text())

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def write(path,value,replace=False):
    path=Path(path);body=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if replace:
        temporary=path.with_name(path.name+'.partial')
        temporary.write_text(body);os.replace(temporary,path)
    else:
        with path.open('x') as stream:stream.write(body)

def inside(root,path):
    p=Path(path);p=(p if p.is_absolute() else root/p).resolve()
    try:p.relative_to(root.resolve())
    except ValueError:raise GateError('Identity path escapes evaluation root: '+str(path))
    return p

def build_plan():
    plan=[]
    for offset,seed in enumerate(SEEDS):
        views=VIEWS[offset%2:]+VIEWS[:offset%2]
        for view,engine in views:
            plan.append(dict(view=view,engine=engine,seed=seed,compiled=True))
    return plan

def registry(root,slot,test=False):
    directory='a1-jax-v1-test-once' if test else 'a1-jax-v1-run-once'
    return root/'evidence'/directory/f"{slot['engine']}-jit-seed-{slot['seed']}.json"

def true_compute_conflicts(root):
    import psutil
    own=psutil.Process();excluded={own.pid,*[p.pid for p in own.parents()]}
    explicit={'run_a1_mnist_v4.py','run_a1_mnist_jax_v1.py','prepare_data.py',
              'capability_migration.py','frontier_capability.py','run_graph_phase.py','b2-train'}
    found=[]
    for process in psutil.process_iter(['pid','cmdline']):
        if process.pid in excluded:continue
        try:
            argv=process.info['cmdline'] or []
            names=[Path(arg).name for arg in argv]
            heavy=any(name in explicit or name.startswith(('benchmark_','qualify_','generate_')) for name in names)
            if heavy and any(str(root) in arg for arg in argv):
                found.append(dict(pid=process.pid,argv=argv))
        except psutil.Error:continue
    return found

def file_map(root,paths):
    result={}
    for path in paths:
        p=inside(root,path)
        if not p.is_file():raise GateError('Required identity file absent: '+str(p))
        result[str(p.relative_to(root))]=digest(p)
    return result

def verify_map(root,values):
    for rel,expected in values.items():
        p=inside(root,rel)
        if not p.is_file() or digest(p)!=expected:raise GateError('Frozen identity changed: '+rel)

def barriers(root,a1_terminal,audit_terminal,after):
    prior=read(a1_terminal);audit=read(audit_terminal)
    if prior.get('status')!='finite_a1_queue_exited' or prior.get('finite_slots')!=25 or len(prior.get('records',[]))!=25:
        raise GateError('A1 v4 queue has not closed its declared25 slots')
    if audit.get('exit_code')!=0 or audit.get('termination_reason','exited')!='exited':
        raise GateError('A1 v4 independent full-audit job did not exit0 normally')
    values=file_map(root,[a1_terminal,audit_terminal])
    for path in after:
        value=read(path)
        if not value or str(value.get('status','')).lower() in ('started','running','pending','waiting'):
            raise GateError('Additional outer phase is not terminal: '+str(path))
        values.update(file_map(root,[path]))
    conflicts=true_compute_conflicts(root)
    if conflicts:raise GateError('Another real evaluation compute job is active: '+json.dumps(conflicts))
    return values

def consume_receipt(root,path,audit_terminal):
    receipt=read(path)
    if receipt.get('schema')!='a1-v4-fixture-receipt-v1' or receipt.get('status')!='sealed_after_closed_full_audit':
        raise GateError('Independent fixture receipt is absent/unsealed/unsupported')
    audit=receipt.get('audit',{})
    if audit.get('mode')!='full' or audit.get('closed_slots')!=25 or audit.get('exit_code')!=0:
        raise GateError('Fixture receipt does not prove a closed25-slot full audit')
    terminal=inside(root,audit.get('job_terminal_path',''))
    if terminal!=audit_terminal or digest(terminal)!=audit.get('job_terminal_sha256'):
        raise GateError('Fixture receipt belongs to another audit job terminal')
    report=inside(root,audit.get('report_path',''))
    validator=inside(root,audit.get('validator_path',''))
    if digest(report)!=audit.get('sha256') or digest(validator)!=audit.get('validator_sha256'):
        raise GateError('Audit report/validator identity differs from sealed receipt')
    identities=receipt.get('identities',{})
    if not isinstance(identities,dict) or not identities:raise GateError('Receipt identity map missing')
    verify_map(root,identities)
    fixtures=receipt.get('fixtures',{})
    manifest=inside(root,fixtures.get('manifest_path','fixtures/a1-v4/manifest.json'))
    expected_manifest=fixtures.get('manifest_sha256')
    if digest(manifest)!=expected_manifest or identities.get(str(manifest.relative_to(root)))!=expected_manifest:
        raise GateError('Actual manifest differs from independent audit identity')
    data=read(manifest)
    files=fixtures.get('seed_files',[])
    if len(files)!=5 or {r.get('seed') for r in files}!=set(SEEDS):
        raise GateError('Receipt must preserve exactly five actual shared seed files')
    bound={}
    for item in files:
        seed=item['seed'];rel=f'fixtures/a1-v4/seed-{seed}.npz';p=root/rel
        if item.get('path')!=rel or identities.get(rel)!=item.get('sha256') or digest(p)!=item.get('sha256'):
            raise GateError('Audited actual seed-file identity differs')
        if data.get('files',{}).get(p.name,{}).get('sha256')!=item['sha256']:
            raise GateError('Manifest and independent seed-file receipt disagree')
        if item.get('bytes') is not None and p.stat().st_size!=item['bytes']:
            raise GateError('Actual seed-file byte count differs')
        bound[p.name]=dict(seed=seed,path=rel,sha256=item['sha256'],bytes=p.stat().st_size)
    splits=fixtures.get('split_files',[])
    if len(splits)!=3 or len({v.get('path') for v in splits})!=3:
        raise GateError('Exactly three audited split/preprocessing identities are required')
    for item in splits:
        if identities.get(item['path'])!=item['sha256'] or digest(inside(root,item['path']))!=item['sha256']:
            raise GateError('Audited split/preprocessing identity differs')
    linked=file_map(root,[path,report,validator,audit_terminal]);linked.update(identities)
    return receipt,dict(manifest_sha256=expected_manifest,seed_files=bound),linked

def q0_gate(root,qroot,view,engine):
    path=qroot/f'q0-{view}'/'report.json';linked={}
    try:
        report=read(path);linked.update(file_map(root,[path]))
        if report.get('engine')!=engine or report.get('dense_qualification_status')!='passed':
            raise GateError('Prior Q0 engine/status differs')
        if set(report.get('cases',{}))!=set(QCASES) or not report.get('identities'):
            raise GateError('Prior Q0 four-case/source denominator differs')
        verify_map(root,report['identities']);linked.update(report['identities'])
        raw_files={}
        for name in QCASES:
            case=report['cases'][name]
            if case.get('passed') is not True or not case.get('checks') or not all(v.get('passed') is True for v in case['checks'].values()):
                raise GateError('Prior Q0 case is not fully qualified: '+name)
            raw_path=path.parent/(name+'.json');raw=read(raw_path).get('raw',{})
            linked.update(file_map(root,[raw_path]));raw_files[name]=dict(path=str(raw_path.relative_to(root)),sha256=digest(raw_path))
            if raw.get('engine')!=engine or raw.get('jax_enable_x64') is not True or raw.get('parameter_dtype')!='float64' or raw.get('input_dtype')!='float64':
                raise GateError('Prior Q0 actual engine/FP64 profile differs')
        return dict(qualified=True,report_path=str(path.relative_to(root)),report_sha256=digest(path),raw_cases=raw_files),linked
    except (OSError,KeyError,ValueError,GateError) as error:
        return dict(qualified=False,reason=f'{type(error).__name__}: {error}',report_path=str(path.relative_to(root))),linked

def outcome(external,child):
    reason=external.get('termination_reason')
    if reason=='timeout':return 'outer_supervision_timeout'
    if reason=='resource_limit':return 'rss_guard_rejected'
    if external.get('remaining_owned_processes'):return 'cleanup_not_verified'
    status=child.get('status','worker_terminal_missing')
    if status=='completed' and external.get('exit_code')!=0:return 'inconsistent_completed_nonzero_exit'
    return status

def frozen_contract(root,out,draft,receipt_path,receipt,shared,bindings,qrefs):
    spec=copy.deepcopy(draft)
    if spec.get('status')!='draft_pretraining' or spec.get('schema')!='atlas-a1-jax-execution-addendum-v1':
        raise GateError('Only the reviewed unexecuted JAX draft may create this new frozen contract')
    worker=root/'tools/run_a1_mnist_jax_v1.py'
    if spec.get('implementation_sha256')!=digest(worker):raise GateError('Draft worker identity differs')
    verify_map(root,spec['source_bindings'])
    spec.update(status='frozen_pretraining',frozen_by='tools/run_a1_jax_queue_v1.py',
                frozen_queue_output=str(out.relative_to(root)),freeze_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    spec['shared_reference'].update(shared)
    spec['shared_reference'].pop('pending_reason',None)
    spec['fixture_receipt']=dict(path=str(receipt_path.relative_to(root)),sha256=digest(receipt_path),
                                schema=receipt['schema'],status=receipt['status'],audit=receipt['audit'])
    spec['q0_reference']=qrefs
    spec['source_bindings'].update(bindings)
    spec['queue_authority']=dict(run_once_directory='evidence/a1-jax-v1-run-once',
                                queue_output=str(out.relative_to(root)),
                                policy='Each exact new engine/seed has one durable run claim and one independent durable test claim; no retry or HPO')
    target=out/'A1-jax-execution-addendum-v1.frozen.json';write(target,spec)
    return target

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--draft-contract',type=Path,default=Path('protocol/A1-jax-execution-addendum-v1.draft.json'))
    parser.add_argument('--a1-terminal',type=Path,default=Path('evidence/a1-queue-v4/terminal.json'))
    parser.add_argument('--audit-terminal',type=Path,required=True)
    parser.add_argument('--q0-root',type=Path,default=Path('evidence/remote-qualify-v1'))
    parser.add_argument('--after',type=Path,action='append',default=[])
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-only',action='store_true')
    modes.add_argument('--serial-phase',action='store_true')
    parser.add_argument('--allow-run',action='store_true')
    args=parser.parse_args()
    if socket.gethostname()!=HOST:raise GateError('This phase is restricted to the authorized remote Mac Studio')
    root=args.root.resolve()
    if root!=ROOT.resolve():raise GateError('Queue root differs from its immutable script location')
    out=inside(root,args.output);out.mkdir(parents=True,exist_ok=False)
    plan=build_plan()
    rows=[dict(**slot,launch_status='not_launched',scientific_status='not_launched',
               resource_qualification=False,test_status='sealed_not_launched') for slot in plan]
    spec=dict(schema='a1-jax-queue-v1',host=HOST,finite_slots=10,order=plan,
              inner_total_wall_s=INNER_CAP,outer_supervision_s=OUTER_CAP,stage_cap_s=STAGE_CAP,
              rss_guard_bytes=RSS_GUARD,minimum_free_disk_bytes=DISK_FLOOR,
              resource_qualification=False,timing_class='JAX host-config diagnostics',
              outer_cap_policy='1830s allows supervision/cleanup only; the model worker enforces its own1800s total cap.',
              retry_policy='No run/test replay; exclusive engine/seed registries survive fresh output directories.',
              prerequisite_policy='v4 closed25 slots + independent audit normal exit0 + sealed fixture receipt + no real compute; outer handles inserted stages')
    write(out/'plan.json',spec)
    if args.prepare_only:
        write(out/'prepared-only.json',dict(status='plan_prepared_no_execution',finite_slots=10,records=rows,
             dependencies=DEPENDENCIES,receipt_required=str(args.receipt),note='No contract frozen, registries claimed, framework imported or worker launched.'))
        print(json.dumps(dict(status='plan_prepared_no_execution',output=str(out),finite_slots=10)));return 0
    begin=time.monotonic();coordinator_status='blocked_before_execution';failure=None;frozen=None
    try:
        if not args.allow_run:raise GateError('Serial execution requires explicit --allow-run accidental-execution guard')
        a1_terminal=inside(root,args.a1_terminal);audit_terminal=inside(root,args.audit_terminal)
        after=[inside(root,p) for p in args.after]
        barrier_files=barriers(root,a1_terminal,audit_terminal,after)
        if shutil.disk_usage(root).free<DISK_FLOOR:raise GateError('50 GiB remote free-disk floor reached')
        receipt_path=inside(root,args.receipt)
        receipt,shared,receipt_files=consume_receipt(root,receipt_path,audit_terminal)
        draft_path=inside(root,args.draft_contract);draft=read(draft_path)
        paths=[*DEPENDENCIES,str(draft_path.relative_to(root))]
        bindings=file_map(root,paths);bindings.update(barrier_files);bindings.update(receipt_files)
        verify_map(root,draft['source_bindings']);bindings.update(draft['source_bindings'])
        gates={};qrefs={};qroot=inside(root,args.q0_root)
        for view,engine in VIEWS:
            gate,files=q0_gate(root,qroot,view,engine);gates[engine]=gate;bindings.update(files);qrefs[engine]=gate
        phase_claim=root/'evidence/a1-jax-v1-run-once/phase-activated.json'
        worker_claims=list(phase_claim.parent.glob('*.worker-started.json'))
        if phase_claim.exists() or worker_claims or any(registry(root,s,test=t).exists() for s in plan for t in (False,True)):
            raise GateError('This formal JAX phase/view/seed already has a run/test claim; no automatic retry')
        frozen=frozen_contract(root,out,draft,receipt_path,receipt,shared,bindings,qrefs)
        spec.update(files=bindings,contract_path=str(frozen.relative_to(root)),contract_sha256=digest(frozen),
                    receipt_path=str(receipt_path.relative_to(root)),receipt_sha256=digest(receipt_path),
                    q0_gates=gates,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out/'freeze.json',spec)
        phase_claim.parent.mkdir(parents=True,exist_ok=True)
        with phase_claim.open('x') as stream:
            json.dump(dict(queue_output=str(out),freeze_sha256=digest(out/'freeze.json'),contract_sha256=digest(frozen)),stream)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        from run_recurrent_queue import launch
        env=os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORM_NAME='cpu',JAX_ENABLE_X64='true',
                   OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        def guard():
            verify_map(root,bindings)
            if digest(frozen)!=spec['contract_sha256']:raise GateError('Frozen JAX contract changed')
            if true_compute_conflicts(root):raise GateError('Another real compute job became active')
            if shutil.disk_usage(root).free<DISK_FLOOR:raise GateError('50 GiB remote disk floor reached')
            if time.monotonic()-begin>STAGE_CAP-OUTER_CAP:raise GateError('Stage cap cannot fit another complete original slot')
        coordinator_status='finite_queue_exited'
        for index,slot in enumerate(plan):
            guard();row=rows[index];gate=gates[slot['engine']]
            if not gate['qualified']:
                row.update(scientific_status='unqualified',reason=gate['reason'],test_status='sealed_q0_gate_failed')
                write(out/'progress.json',dict(records=rows,finite_slots=10),replace=True)
                continue
            job=f"{slot['view']}-seed-{slot['seed']}";output=out/job
            claim=registry(root,slot)
            with claim.open('x') as stream:
                json.dump(dict(**slot,output=str(output),queue_output=str(out),contract_sha256=digest(frozen),
                               freeze_sha256=digest(out/'freeze.json'),fixture_manifest_sha256=shared['manifest_sha256']),stream)
                stream.write('\n');stream.flush();os.fsync(stream.fileno())
            command=[str(root/'environment/jax/bin/python'),str(root/'tools/run_a1_mnist_jax_v1.py'),
                     '--root',str(root),'--contract',str(frozen),'--allow-run','--engine',slot['engine'],'--seed',str(slot['seed']),
                     '--output',str(output),'--q0-report',str(root/gate['report_path'])]
            row.update(launch_status='launch_claimed',run_claim_sha256=digest(claim),output=str(output.relative_to(root)))
            write(out/'progress.json',dict(records=rows,finite_slots=10),replace=True)
            print('START',job,flush=True)
            try:
                outer=launch(job,command,OUTER_CAP,RSS_GUARD,out,output,env,'a1_jax_supervisor')
            except BaseException:
                saved=out/f'{job}-terminal.json'
                if saved.is_file():row['outer']=read(saved)
                row.update(launch_status='launch_attempted',scientific_status='supervision_failed')
                raise
            child_path=output/'terminal.json';child=read(child_path) if child_path.is_file() else {}
            status=outcome(outer,child)
            row.update(launch_status='launched',scientific_status=status,outer=outer,child_status=child.get('status'),
                       child_terminal_sha256=digest(child_path) if child_path.is_file() else None,
                       completed_epochs=child.get('completed_epochs'),selected_checkpoint=child.get('selected_checkpoint'),
                       test_status=child.get('test_status','no_child_terminal'),test_attempted=registry(root,slot,test=True).exists())
            if status=='completed' and child.get('test_status')=='completed_once':
                row['test']=child.get('test')
            elif outer.get('termination_reason') in ('timeout','resource_limit'):
                row['test_status']='not_accepted_outer_'+outer['termination_reason']
            write(out/'progress.json',dict(records=rows,finite_slots=10),replace=True)
            print('END',job,status,flush=True)
            if status in ('cleanup_not_verified','cleanup_not_verified_do_not_launch_next_case'):
                raise GateError('Child cleanup not verified; no next formal slot')
    except BaseException as error:
        coordinator_status='coordinator_stopped' if any(r['launch_status']!='not_launched' for r in rows) else 'blocked_before_execution'
        failure=dict(error_type=type(error).__name__,error=str(error))
        for row in rows:
            if row['scientific_status']=='not_launched':row['reason']=failure
    terminal=dict(schema='a1-jax-queue-terminal-v1',status='finite_a1_jax_queue_closed',
                  coordinator_status=coordinator_status,finite_slots=10,records=rows,failure=failure,
                  elapsed_s=time.monotonic()-begin,resource_qualification=False,complete_evaluation=False,
                  note='Ten outcomes preserved; model/quality/test evidence still requires independent validation. JAX is not strict1thread qualified.')
    write(out/'terminal.json',terminal)
    print(json.dumps(dict(status=terminal['status'],coordinator_status=coordinator_status,finite_slots=10,
                          launched=sum(r['launch_status']=='launched' for r in rows),output=str(out))))
    return 2 if failure else 0

if __name__=='__main__':raise SystemExit(main())
