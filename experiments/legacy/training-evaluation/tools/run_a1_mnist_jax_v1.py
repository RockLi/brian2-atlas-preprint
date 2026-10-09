#!/usr/bin/env python3
"""A1 JAX phase v1: bounded genuine Spyx and Brainstate MNIST runs.

No task is launched by importing this file. Execution is guarded to the user-
authorized Mac Studio and requires a separately frozen A1 contract. Official
test labels are decoded only after a complete-epoch selected checkpoint exists.
The supervisor gives cold setup, qualification, training, validation and the
single final test one shared 1800-second wall budget. It kills its own PID tree.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

SEEDS=(11,23,37,51,71)
ENGINES=('spyx','brainx_state')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def dump(path,value,replace=False):
    path=Path(path)
    content=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if replace:
        partial=path.with_name(path.name+'.new');partial.write_text(content);os.replace(partial,path)
    else:
        with path.open('x') as stream:stream.write(content)


def read(path):return json.loads(Path(path).read_text())


def host_guard():
    if socket.gethostname().split('.')[0]!='rock-mac-studio-1':
        raise RuntimeError('A1 data/training runs only on authorized100.90.28.27; no MacBook execution')


class CapReached(Exception):pass
class Unqualified(Exception):pass
class BudgetRejected(Exception):pass


def check_cap(deadline):
    if time.monotonic()>=deadline:raise CapReached('strict total wall cap reached')


def encode(images,ids):
    import numpy as np
    # Actual construction/copy is inside the caller's wall-clock interval.
    current=np.asarray(images[ids],dtype=np.float64).reshape(len(ids),784)/255.0
    return np.repeat(current[:,None,:],100,axis=1)


def compare(actual,expected):
    import numpy as np
    a=np.asarray(actual);b=np.asarray(expected)
    return dict(passed=bool(a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.abs(a-b)<=1e-10+1e-8*np.abs(b))),
                actual_shape=list(a.shape),expected_shape=list(b.shape))


def test_ledger_path(args):
    # One immutable claim per new framework view/seed, even if contract changes.
    return args.root/'evidence/a1-jax-v1-test-once'/f'{args.engine}-jit-seed-{args.seed}.json'


def validate_contract(args):
    spec=read(args.contract)
    if spec.get('status')!='frozen_pretraining':raise Unqualified('A1 JAX phase needs a coordinator-frozen contract')
    if spec.get('schema')!='atlas-a1-jax-execution-addendum-v1':raise Unqualified('Wrong A1 phase contract')
    if spec.get('implementation_sha256')!=sha(__file__):raise Unqualified('A1 JAX worker differs from contract')
    if spec.get('total_wall_cap_seconds_per_seed')!=1800 or spec.get('formal_seeds')!=list(SEEDS):
        raise Unqualified('A1 cap/seed denominator changed')
    if spec.get('sizes')!=[784,128,10] or spec.get('T')!=100 or spec.get('global_batch')!=32:
        raise Unqualified('A1 model/batch/time denominator changed')
    for rel,h in spec['source_bindings'].items():
        if sha(args.root/rel)!=h:raise Unqualified('Frozen source/reference changed: '+rel)
    ref=spec.get('shared_reference',{})
    if not isinstance(ref.get('manifest_sha256'),str) or len(ref['manifest_sha256'])!=64:
        raise Unqualified('Actual v4 manifest identity is not independently sealed')
    if set(ref.get('seed_files',{}))!={f'seed-{seed}.npz' for seed in SEEDS}:
        raise Unqualified('All five audited actual v4 arrays must be frozen before A1 JAX execution')
    receipt=spec.get('fixture_receipt',{})
    if receipt.get('status')!='sealed_after_closed_full_audit' or not receipt.get('sha256'):
        raise Unqualified('Independent v4 fixture receipt is required')
    expected_q0=spec.get('q0_reference',{}).get(args.engine,{})
    if not expected_q0.get('report_path') or args.q0_report.resolve()!=(args.root/expected_q0['report_path']).resolve():
        raise Unqualified('Q0 report is not the queue-frozen report for this exact framework view')
    if sha(args.q0_report)!=expected_q0.get('report_sha256'):raise Unqualified('Frozen prior Q0 report identity changed')
    if expected_q0.get('qualified') is not True:raise Unqualified('Queue-frozen prior Q0 gate is not qualified')
    authority=spec.get('queue_authority',{})
    if authority.get('run_once_directory')!='evidence/a1-jax-v1-run-once':
        raise Unqualified('Exact JAX run-once registry is required')
    queue_output=(args.root/authority.get('queue_output','')).resolve()
    try:queue_output.relative_to(args.root)
    except ValueError:raise Unqualified('Queue output escapes evaluation root')
    claim_path=args.root/authority['run_once_directory']/f'{args.engine}-jit-seed-{args.seed}.json'
    claim=read(claim_path)
    if (claim.get('engine')!=args.engine or claim.get('seed')!=args.seed or claim.get('compiled') is not True
            or Path(claim.get('output','')).resolve()!=args.output
            or Path(claim.get('queue_output','')).resolve()!=queue_output
            or claim.get('contract_sha256')!=sha(args.contract)
            or claim.get('fixture_manifest_sha256')!=ref['manifest_sha256']
            or claim.get('freeze_sha256')!=sha(queue_output/'freeze.json')):
        raise Unqualified('Queue run-once claim does not bind this exact engine/seed/output/contract/freeze')
    freeze=read(queue_output/'freeze.json')
    phase=read(claim_path.parent/'phase-activated.json')
    if (freeze.get('contract_sha256')!=sha(args.contract)
            or phase.get('contract_sha256')!=sha(args.contract)
            or phase.get('freeze_sha256')!=claim['freeze_sha256']
            or Path(phase.get('queue_output','')).resolve()!=queue_output):
        raise Unqualified('Immutable phase activation differs from the queue run claim')
    return spec


def claim_worker_once(args):
    path=args.root/'evidence/a1-jax-v1-run-once'/f'{args.engine}-jit-seed-{args.seed}.worker-started.json'
    with path.open('x') as stream:
        json.dump(dict(engine=args.engine,seed=args.seed,output=str(args.output),
                       contract_sha256=sha(args.contract),started_monotonic=args.started),stream)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def qualify(model,weights,images,labels,order,dest,deadline):
    import numpy as np
    from oracle import forward_vjp,adam
    report=dict(scope='actual A1 full shape and tail-shape train/evaluate paths; prior Q0 required',cases={})
    for name,ids,steps in [('B32',order[:32],3),('tail_B24',order[-24:],1)]:
        model.reset();w=[a.copy() for a in weights];m=[np.zeros_like(a) for a in w];v=copy.deepcopy(m)
        x=encode(images,ids);y=np.asarray(labels[ids],dtype=np.int64)
        case=dict(sizes=[784,128,10],inputs=x,labels=y,weights=w,beta=.95,theta=1.)
        checks={}
        for index in range(1,steps+1):
            check_cap(deadline);reference=forward_vjp(case,w);actual=model.batch(x,y,True,diagnostic=True)
            w,m,v=adam(w,reference['gradients'],m,v,index)
            for field,expected in [('loss',reference['loss']),('logits',reference['logits'])]:checks[f'step{index}_{field}']=compare(actual[field],expected)
            for field,expected in [('gradients',reference['gradients']),('weights',w),('first',m),('second',v)]:
                for bank,(a,b) in enumerate(zip(actual[field],expected)):checks[f'step{index}_{field}_{bank}']=compare(a,b)
            checks[f'step{index}_counter']={'passed':model.counter()==index}
            checks[f'step{index}_initial_vjp']=compare(actual['initial_vjp'],reference['initial_vjp'])
            check_cap(deadline)
        if name=='B32':
            checkpoint=dest/'qualification-checkpoint.npz'
            before_restore=model.state_arrays();model.save(checkpoint);model.reset();model.restore(checkpoint)
            after_restore=model.state_arrays()
            for key in ['weights','first','second']:
                for bank,(a,b) in enumerate(zip(after_restore[key],before_restore[key])):
                    checks[f'checkpoint_restore_{key}_{bank}']=compare(a,b)
            checks['checkpoint_restore_counter']={'passed':model.counter()==steps}
            check_cap(deadline)
        report['cases'][name]=dict(passed=all(c['passed'] for c in checks.values()),checks=checks)
    for count in [32,8,16]:
        check_cap(deadline);model.reset();ids=order[:count];x=encode(images,ids);y=np.asarray(labels[ids],dtype=np.int64)
        ref=forward_vjp(dict(sizes=[784,128,10],inputs=x,labels=y,weights=weights,beta=.95,theta=1.))
        before_state=model.state_arrays();before_count=model.counter()
        actual=model.batch(x,y,False);after_state=model.state_arrays()
        checks={k:compare(actual[k],ref[k]) for k in ['loss','logits']}
        checks['pure_forward_counter_unchanged']={'passed':model.counter()==before_count}
        for field in ['weights','first','second']:
            for bank,(a,b) in enumerate(zip(after_state[field],before_state[field])):
                checks[f'pure_forward_{field}_{bank}_unchanged']=compare(a,b)
        report['cases'][f'evaluate_B{count}']=dict(passed=all(c['passed'] for c in checks.values()),checks=checks)
    # Explicit no-gradient evaluation qualification at and around threshold.
    # Find a representable voltage whose canonical FP64 beta*v-theta is zero.
    center=np.float64(1./.95);candidates=[center]
    lower=upper=center
    for _ in range(8):
        lower=np.nextafter(lower,-np.inf);upper=np.nextafter(upper,np.inf);candidates.extend([lower,upper])
    zeros=[v for v in candidates if np.float64(.95)*v-np.float64(1.)==0.]
    if not zeros:raise Unqualified('cannot construct exact FP64 threshold fixture')
    center=zeros[0];lower=np.nextafter(center,-np.inf);upper=np.nextafter(center,np.inf)
    while .95*lower-1.>=0:lower=np.nextafter(lower,-np.inf)
    while .95*upper-1.<=0:upper=np.nextafter(upper,np.inf)
    voltages=[float(lower),float(center),float(upper)]
    initial=np.zeros((2,138),dtype=np.float64)
    initial[:,0:3]=voltages;initial[:,128:131]=voltages
    check_cap(deadline);model.reset();ids=order[:2];x=encode(images,ids);y=np.asarray(labels[ids],dtype=np.int64)
    case=dict(sizes=[784,128,10],inputs=x,labels=y,weights=weights,beta=.95,theta=1.,initial=initial)
    ref=forward_vjp(case);actual=model.batch(x,y,False,explicit_initial=initial)
    checks={k:compare(actual[k],ref[k]) for k in ['loss','logits']}
    margin=(.95*initial-1.)[:,[0,1,2,128,129,130]]
    checks['explicit_zero_and_neighbors']={'passed':bool(np.any(margin==0.) and np.any(margin<0.) and np.any(margin>0.))}
    detail=dict(initial_nonzero_columns=[0,1,2,128,129,130],initial_voltage=initial[:,[0,1,2,128,129,130]].tolist(),
                canonical_actual_FP64_margin=margin.tolist(),grad_enabled=False,checks=checks)
    if 'first_tick_spikes' in actual:
        checks['actual_compiled_first_tick_spikes']=compare(actual['first_tick_spikes'],ref['spikes'][:,0,:])
        detail['actual_compiled_first_tick_boundary_spikes']=actual['first_tick_spikes'][:,[0,1,2,128,129,130]].tolist()
    if 'first_tick_margins' not in actual:raise Unqualified('Actual compiled evaluation did not return threshold margins')
    observed=actual['first_tick_margins'];selected=observed[:,[0,1,2,128,129,130]]
    checks['actual_compiled_zero_and_neighbors']={'passed':bool(np.all(selected[:,[1,4]]==0.) and
        np.all(selected[:,[0,3]]<0.) and np.all(selected[:,[2,5]]>0.))}
    checks['actual_compiled_margin_vs_oracle']=compare(observed,.95*initial-1.)
    checks['actual_compiled_strict_spikes']={'passed':bool(np.array_equal(actual['first_tick_spikes'],(observed>0.).astype(np.float64)))}
    detail['actual_compiled_first_tick_boundary_margins']=selected.tolist()
    detail['evaluation_graph']='same pure-forward framework JIT used for validation/test, full B2/T100'
    detail['passed']=all(c['passed'] for c in checks.values())
    report['cases']['no_gradient_evaluate_B2_exact_threshold_and_neighbors']=detail
    report['passed']=all(row['passed'] for row in report['cases'].values())
    dump(dest/'a1-shape-qualification.json',report)
    if not report['passed']:raise Unqualified('actual A1 shape qualification failed')
    model.reset();check_cap(deadline)
    return report


def worker(args):
    root=args.root;dest=args.output;deadline=args.deadline
    sys.path.insert(0,str(root/'adapters'));sys.path.insert(0,str(root/'tools'))
    report=dict(schema='a1-mnist-jax-run-v1',engine=args.engine,compile=True,seed=args.seed,status='started',
        total_wall_cap_s=1800,epochs=[],selection=None,test_status='sealed_no_selected_checkpoint',test_labels_decoded=False,
        primary_timing='coordinator total wall includes cold/input construction/full train/validation/checkpoint/finaltest',
        source_hashes={str(p.relative_to(root)):sha(p) for p in [Path(__file__),root/'adapters/a1_jax_adapter_v1.py',root/'adapters/jax_adapter.py',root/'adapters/oracle.py',root/'tools/run_a1_mnist_v4.py']})
    log=None
    def progress(phase):
        report['phase']=phase;report['elapsed_total_s']=time.monotonic()-args.started
        dump(dest/'progress.json',report,replace=True)
    try:
        spec=validate_contract(args)
        claim_worker_once(args)
        import numpy as np
        from prepare_data import read_idx
        q=read(args.q0_report)
        if q.get('dense_qualification_status')!='passed':raise Unqualified('matching prior dense Q0 is required')
        if q.get('engine')!=args.engine:raise Unqualified('prior Q0 engine does not match requested engine')
        for rel,h in q['identities'].items():
            if sha(root/rel)!=h:raise Unqualified(f'prior Q0 source identity changed:{rel}')
        expected_cases={'base_negative_count_input','batch_duplicate','initial_threshold_boundary','single_sample_no_carry'}
        if set(q.get('cases',{}))!=expected_cases:raise Unqualified('Prior JAX Q0 case denominator differs')
        for name,case in q['cases'].items():
            if case.get('passed') is not True or not case.get('checks') or not all(c.get('passed') is True for c in case['checks'].values()):
                raise Unqualified('Prior JAX Q0 has an incomplete/failed case')
            previous=read(args.q0_report.parent/(name+'.json'))['raw']
            if previous.get('engine')!=args.engine or previous.get('jax_enable_x64') is not True:
                raise Unqualified('Prior JAX Q0 engine/precision differs')
            if previous.get('parameter_dtype')!='float64' or previous.get('input_dtype')!='float64':
                raise Unqualified('Prior JAX Q0 is not actual FP64')
        report['resource_qualification']=False
        report['timing_class']='JAX host-config diagnostics; no strict1thread ranking'
        if test_ledger_path(args).exists():raise Unqualified('This formal view/seed already claimed its once-only test; no retry allowed')
        shared=root/'fixtures/a1-v4'
        if sha(shared/'manifest.json')!=spec['shared_reference']['manifest_sha256']:
            raise Unqualified('Actual shared manifest differs from independently audited v4 identity')
        manifest=read(shared/'manifest.json')
        if manifest['contract_sha256']!=spec['shared_reference']['contract_sha256']:raise Unqualified('shared arrays came from a different base A1 contract')
        if manifest.get('generator_sha256')!=spec['shared_reference']['generator_sha256']:raise Unqualified('shared arrays came from another v4 generator')
        if manifest.get('seeds')!=list(SEEDS):raise Unqualified('shared array seed denominator differs')
        artifact=shared/f'seed-{args.seed}.npz'
        expected=spec['shared_reference']['seed_files'][artifact.name]
        if expected.get('seed')!=args.seed or expected.get('path')!=str(artifact.relative_to(root)):
            raise Unqualified('Frozen shared seed path/identity differs')
        if sha(artifact)!=expected.get('sha256') or manifest['files'][artifact.name]['sha256']!=expected['sha256']:
            raise Unqualified('Actual v4 shared array differs from independent receipt')
        arrays=np.load(artifact);weights=[arrays['bank_0'].copy(),arrays['bank_1'].copy()];order=arrays['epoch_indices']
        p=root/'data/mnist/processed'
        for name,h in manifest['split_sha256'].items():
            if sha(p/name)!=h:raise Unqualified('MNIST preprocessing identity changed')
        metadata=read(p/'preprocess.json')
        for name in ['train-images-idx3-ubyte','train-labels-idx1-ubyte']:
            row=metadata['raw_sources'][name]
            if sha(root/'data'/row['path'])!=row['sha256']:raise Unqualified('MNIST train original identity changed')
        images=read_idx(root/'data/mnist/raw/train-images-idx3-ubyte',(60000,28,28))
        labels=read_idx(root/'data/mnist/raw/train-labels-idx1-ubyte',(60000,))
        train=np.load(p/'train_indices.npy');validation=np.load(p/'validation_indices.npy')
        if order.shape!=(10,55000) or any(not np.array_equal(np.sort(row),train) for row in order):raise Unqualified('epoch denominator changed')
        report.update(contract_sha256=sha(args.contract),common_arrays_sha256=sha(artifact),shared_manifest_sha256=sha(shared/'manifest.json'),
            q0_report_sha256=sha(args.q0_report),numpy_version=np.__version__,thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']})
        report['environment_lock_sha256']={p.name:sha(p) for p in (root/'environment').glob('*lock.txt')}
        progress('cold_setup');check_cap(deadline)
        from a1_jax_adapter_v1 import A1Jax
        model=A1Jax(root,weights,args.seed,args.engine)
        report['evaluation_policy']='pure forward actual framework JIT; no derivative or optimizer update; same strict canonical neuron equations'
        report['versions']={name:importlib.metadata.version(name) for name in ['jax','jaxlib','optax','spyx' if args.engine=='spyx' else 'brainstate']}
        framework='spyx' if args.engine=='spyx' else 'brainstate'
        for name in q['cases']:
            old=read(args.q0_report.parent/(name+'.json'))['raw']
            if old.get('version')!=report['versions'][framework] or old.get('jax')!=report['versions']['jax']:
                raise Unqualified('Actual framework/JAX versions differ from prior Q0')

        progress('A1_shape_qualification');qualify(model,weights,images,labels,order[0],dest,deadline)
        report['A1_shape_qualification']='passed';report['qualification_elapsed_s']=time.monotonic()-args.started
        report['implementation']=model.implementation()
        log=(dest/'batches.jsonl').open('x')
        def batches(ids,phase,epoch,training,image_source=images,label_source=labels):
            samples=correct=0;total_loss=0.;start=time.monotonic()
            for offset in range(0,len(ids),32):
                check_cap(deadline);before=time.monotonic();take=ids[offset:offset+32]
                x=encode(image_source,take);y=np.asarray(label_source[take],dtype=np.int64)
                value=model.batch(x,y,training);check_cap(deadline)
                n=len(take);hits=int((np.argmax(value['logits'],axis=1)==y).sum())
                samples+=n;correct+=hits;total_loss+=float(value['loss'])*n
                row=dict(phase=phase,epoch=epoch,offset=offset,samples=n,elapsed_s=time.monotonic()-before,
                         elapsed_total_s=time.monotonic()-args.started,public_api_s=value.get('public_api_s'))
                # Never publish partial test score, even in a raw batch log.
                if phase!='test':row.update(loss=value['loss'],correct=hits)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            return dict(samples=samples,correct=correct,loss=total_loss/samples,accuracy=correct/samples,elapsed_s=time.monotonic()-start)
        for epoch in range(1,11):
            epoch_start=time.monotonic();progress('training')
            trained=batches(order[epoch-1],'train',epoch,True)
            if trained['samples']!=55000 or model.counter()!=1719*epoch:raise Unqualified('incomplete epoch or incorrect global Adam count')
            progress('validation');validated=batches(validation,'validation',epoch,False)
            if validated['samples']!=5000:raise Unqualified('partial validation cannot select a checkpoint')
            check_cap(deadline)
            row=dict(epoch=epoch,train=trained,validation=validated,elapsed_total_s=time.monotonic()-args.started)
            if report['selection'] is None or validated['correct']>report['selection']['validation_correct']:
                checkpoint=dest/(f'best-epoch-{epoch}'+model.checkpoint_suffix);model.save(checkpoint);check_cap(deadline)
                report['selection']=dict(epoch=epoch,validation_correct=validated['correct'],validation_accuracy=validated['accuracy'],
                    checkpoint=checkpoint.name,sha256=sha(checkpoint))
            row['epoch_wall_s']=time.monotonic()-epoch_start;report['epochs'].append(row)
            report['test_status']='not_started_selected_checkpoint_available'
            reserve=1.25*2*validated['elapsed_s']+2.;remaining=deadline-time.monotonic()
            row.update(remaining_s=remaining,test_reserve_estimate_s=reserve,
                       continue_threshold_s=1.10*row['epoch_wall_s']+reserve)
            progress('epoch_complete')
            if epoch==10 or remaining<=row['continue_threshold_s']:break
        check_cap(deadline)
        if report['selection'] is None:raise CapReached('no complete epoch selected')
        progress('restore_selected_checkpoint');chosen=dest/report['selection']['checkpoint']
        if sha(chosen)!=report['selection']['sha256']:raise Unqualified('selected checkpoint identity changed')
        model.restore(chosen);check_cap(deadline)
        # Durable once-only marker precedes decoding; a killed run cannot retry.
        marker=dict(selected=report['selection'],elapsed_total_s=time.monotonic()-args.started,attempt=1,
                    run_directory=str(dest),seed=args.seed,engine=args.engine,compile=True,contract_sha256=sha(args.contract))
        ledger=test_ledger_path(args);ledger.parent.mkdir(parents=True,exist_ok=True)
        with ledger.open('x') as locked:
            locked.write(json.dumps(marker,sort_keys=True)+'\n');locked.flush();os.fsync(locked.fileno())
        dump(dest/'test-started.json',marker)
        report['test_status']='in_progress_once_only';progress('test')
        for name in ['t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']:
            row=metadata['raw_sources'][name]
            if sha(root/'data'/row['path'])!=row['sha256']:raise Unqualified('sealed test original identity changed')
        test_images=read_idx(root/'data/mnist/raw/t10k-images-idx3-ubyte',(10000,28,28))
        test_labels=read_idx(root/'data/mnist/raw/t10k-labels-idx1-ubyte',(10000,))
        report['test_labels_decoded']=True
        tested=batches(np.arange(10000),'test',report['selection']['epoch'],False,test_images,test_labels)
        check_cap(deadline)
        if tested['samples']!=10000:raise Unqualified('partial test result cannot be reported')
        report.update(status='completed',test_status='completed_once',test=tested)
    except CapReached as error:
        report.update(status='censored_at_cap',reason=str(error))
        report['test_status']='censored_at_cap' if report.get('selection') is not None else 'sealed_no_complete_epoch'
    except BaseException as error:
        status=('unqualified' if isinstance(error,Unqualified) else 'budget_rejected' if isinstance(error,BudgetRejected)
                else 'numerical_divergence' if isinstance(error,FloatingPointError) else 'dependency_failed' if isinstance(error,ImportError)
                else 'oom' if isinstance(error,MemoryError) else 'software_rejected')
        report.update(status=status,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        if log is not None:log.close()
        report['elapsed_total_s']=time.monotonic()-args.started
        dump(dest/'result.json',report)
    return 0 if report['status']=='completed' else 1


def stop_owned_tree(process,known=None):
    import psutil
    known={} if known is None else dict(known)
    parent=None;inspection_error=None
    try:
        if process.poll() is None:
            parent=psutil.Process(process.pid)
            for child in parent.children(recursive=True):known[(child.pid,child.create_time())]=child
    except psutil.NoSuchProcess:pass
    except (psutil.Error,OSError) as error:inspection_error=f'{type(error).__name__}: {error}'
    children=list(known.values());captured=[]
    for child in children:
        try:captured.append(dict(pid=child.pid,created=child.create_time()))
        except psutil.NoSuchProcess:pass
    # Native Atlas children start independent sessions; worker killpg alone is insufficient.
    for child in reversed(children):
        try:child.kill()
        except psutil.NoSuchProcess:pass
        except psutil.Error as error:inspection_error=f'{type(error).__name__}: {error}'
    if parent is not None:
        try:parent.kill()
        except psutil.NoSuchProcess:pass
    # Popen.wait must exclusively reap the direct worker and preserve -SIGKILL.
    # psutil waits only for descendants, never for that Popen-owned parent.
    _,alive=psutil.wait_procs(children,timeout=3)
    for child in alive:
        try:child.kill()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(alive,timeout=1)
    remaining=[]
    for child in alive:
        try:
            if child.status()!=psutil.STATUS_ZOMBIE:remaining.append(child.pid)
        except psutil.NoSuchProcess:pass
    return {'captured':captured,'remaining_non_zombie':remaining,'inspection_error':inspection_error}


def supervise(args):
    import psutil
    start=time.monotonic();deadline=start+1800
    dest=args.output;dest.mkdir(parents=True,exist_ok=False)
    contract=validate_contract(args)
    if contract.get('status')!='frozen_pretraining':raise RuntimeError('Draft A1 contract cannot launch training; freeze the reviewed contract first')
    if contract.get('total_wall_cap_seconds_per_seed')!=1800:raise RuntimeError('A1 strict total cap must remain1800s')
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR=str(dest/'cache/matplotlib'),
        TMPDIR=str(dest/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(dest/'cache/torchinductor'),TRITON_CACHE_DIR=str(dest/'cache/triton'))
    env.update(JAX_PLATFORM_NAME='cpu',JAX_ENABLE_X64='true',JAX_COMPILATION_CACHE_DIR=str(dest/'cache/jax'),
               BRAINEVENT_CACHE_DIR=str(dest/'cache/brainevent'),XDG_CACHE_HOME=str(dest/'cache'))
    Path(env['TMPDIR']).mkdir()
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--allow-run','--root',str(args.root),
        '--contract',str(args.contract),'--engine',args.engine,'--seed',str(args.seed),'--output',str(dest),
        '--q0-report',str(args.q0_report),'--started',str(start),'--deadline',str(deadline)]
    dump(dest/'launch.json',dict(command=command,contract_sha256=sha(args.contract),script_sha256=sha(__file__),
        started_monotonic=start,total_wall_cap_s=1800,thread_environment={k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
        cache_policy='new run directory; per-run initially absent compilation cache'))
    timed_out=False;descendants={'captured':[],'remaining_non_zombie':[]};known={};inspection_errors=[]
    with (dest/'worker.log').open('x') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        while process.poll() is None:
            try:
                for child in psutil.Process(process.pid).children(recursive=True):known[(child.pid,child.create_time())]=child
            except psutil.NoSuchProcess:pass
            except (psutil.Error,OSError) as error:
                inspection_errors.append(f'{type(error).__name__}: {error}')
                descendants=stop_owned_tree(process,known);break
            if time.monotonic()>=deadline:
                timed_out=True;descendants=stop_owned_tree(process,known);break
            time.sleep(.1)
        code=process.wait()
    if not timed_out:
        live={}
        for identity,child in known.items():
            try:
                if child.is_running() and child.status()!=psutil.STATUS_ZOMBIE:live[identity]=child
            except psutil.NoSuchProcess:pass
        if live:descendants=stop_owned_tree(process,live)
    last=read(dest/'result.json') if (dest/'result.json').exists() else read(dest/'progress.json') if (dest/'progress.json').exists() else {}
    effective='censored_at_cap' if timed_out else last.get('status','worker_result_missing')
    if descendants['remaining_non_zombie'] or descendants.get('inspection_error') or inspection_errors:
        effective='cleanup_not_verified_do_not_launch_next_case'
    summary=dict(resource_qualification=False,timing_class='JAX host-config diagnostics',status=effective,exit_code=code,hard_cap_triggered=timed_out,elapsed_total_s=time.monotonic()-start,
        worker_result_present=(dest/'result.json').exists(),selected_checkpoint=last.get('selection'),completed_epochs=len(last.get('epochs',[])),
        test_attempted=(dest/'test-started.json').exists() or test_ledger_path(args).exists(),test_status=last.get('test_status','sealed_no_complete_epoch'),owned_descendants_terminated=descendants,
        descendant_inspection_errors=inspection_errors)
    if timed_out:
        summary['test_status']='censored_at_cap' if last.get('selection') else 'sealed_no_complete_epoch'
    elif last.get('test_status')=='completed_once':summary['test']=last['test']
    dump(dest/'terminal.json',summary);print(json.dumps(summary,allow_nan=False))
    return 0 if effective=='completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--allow-run',action='store_true')
    parser.add_argument('--engine',choices=ENGINES);parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--output',type=Path);parser.add_argument('--q0-report',type=Path)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--started',type=float);parser.add_argument('--deadline',type=float)
    args=parser.parse_args();host_guard();args.root=args.root.resolve();args.contract=args.contract.resolve()
    if Path(sys.prefix).resolve()!=(args.root/'environment/jax').resolve():
        parser.error('A1 JAX phase must use its bound environment/jax Python interpreter')

    if not args.allow_run:parser.error('Training requires explicit --allow-run and frozen contract')
    if any(value is None for value in [args.engine,args.seed,args.output,args.q0_report]):parser.error('Run requires engine, seed, output and q0-report')
    args.output=args.output.resolve();args.q0_report=args.q0_report.resolve()
    if args.worker:
        if args.started is None or args.deadline is None:parser.error('Worker requires coordinator wall-clock bounds')
        if not math.isclose(args.deadline-args.started,1800,rel_tol=0,abs_tol=1e-6):parser.error('Worker cannot change the1800s cap')
        return worker(args)
    return supervise(args)


if __name__=='__main__':raise SystemExit(main())
