#!/usr/bin/env python3
"""A1 v4 pipeline: prepare shared arrays, then bounded Atlas/Torch runs.

No task is launched by importing this file. Execution is guarded to the user-
authorized Mac Studio and requires a separately frozen A1 contract. Official
test labels are decoded only after a complete-epoch selected checkpoint exists.
The supervisor gives cold setup, qualification, training, validation and the
single final test the remaining shared reservation budget, with no 30min per-seed limit. It kills its own PID tree.
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
ENGINES=('atlas','snntorch_fp64','spikingjelly_frontier')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def dump(path,value,replace=False):
    path=Path(path)
    content=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if replace:
        partial=path.with_name(path.name+'.new')
        with partial.open('w') as stream:
            stream.write(content);stream.flush();os.fsync(stream.fileno())
        os.replace(partial,path)
        descriptor=os.open(str(path.parent),os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    else:
        with path.open('x') as stream:
            stream.write(content);stream.flush();os.fsync(stream.fileno())


def read(path):return json.loads(Path(path).read_text())


def host_guard():
    import platform
    if platform.machine() != 'arm64':raise RuntimeError('ARM64 Python required')
    if socket.gethostname().split('.')[0]!='rock-mac-studio-1':
        raise RuntimeError('A1 data/training runs only on authorized100.90.28.27; no MacBook execution')


class CapReached(Exception):pass
class Unqualified(Exception):pass
class BudgetRejected(Exception):pass


def check_cap(deadline):
    if time.monotonic()>=deadline:raise CapReached('strict total wall cap reached')


def prepare_shared(args):
    import numpy as np
    root=args.root;dest=root/'fixtures/a1-arm64-long-r1'
    spec=read(args.contract)
    if spec.get('status')!='frozen_pretraining':raise RuntimeError('Freeze reviewed contract before materializing authoritative shared arrays')
    dest.mkdir(parents=True,exist_ok=False)
    p=root/'data/mnist/processed'
    train=np.load(p/'train_indices.npy');val=np.load(p/'validation_indices.npy')
    if len(train)!=55000 or len(val)!=5000 or np.intersect1d(train,val).size:
        raise ValueError('MNIST frozen split differs from55000/5000 disjoint contract')
    if spec['formal_seeds']!=list(SEEDS) or spec['sizes']!=[784,128,10] or spec['T']!=100:
        raise ValueError('A1 contract shape/seeds changed')
    manifest=dict(schema='a1-shared-arrays-arm64-long-r1',contract_sha256=sha(args.contract),
        generator_sha256=sha(__file__),numpy_version=np.__version__,seeds=list(SEEDS),files={},
        split_sha256={name:sha(p/name) for name in ['train_indices.npy','validation_indices.npy','preprocess.json']},
        original_train_samples=55000,epochs=10,train_batches_per_epoch=1719,tail_batch=24)
    for seed in SEEDS:
        weights_rng=np.random.Generator(np.random.PCG64(seed))
        w=[weights_rng.normal(0.,math.sqrt(2/a),size=(a,b)).astype('<f8').reshape(-1) for a,b in [(784,128),(128,10)]]
        shuffle=np.random.Generator(np.random.PCG64(seed+1000003))
        order=np.stack([shuffle.permutation(train) for _ in range(10)]).astype('<i8')
        if any(not np.array_equal(np.sort(row),train) for row in order):raise ValueError('Incomplete epoch permutation')
        path=dest/f'seed-{seed}.npz'
        np.savez_compressed(path,bank_0=w[0],bank_1=w[1],epoch_indices=order)
        manifest['files'][path.name]={'bytes':path.stat().st_size,'sha256':sha(path),
            'weights_sha256':[hashlib.sha256(a.tobytes()).hexdigest() for a in w],
            'epoch_order_sha256':[hashlib.sha256(row.tobytes()).hexdigest() for row in order]}
    dump(dest/'manifest.json',manifest)
    print(json.dumps({'status':'prepared','manifest':str(dest/'manifest.json'),'sha256':sha(dest/'manifest.json')}))


def encode(images,ids):
    import numpy as np
    # Actual construction/copy is inside the caller's wall-clock interval.
    current=np.asarray(images[ids],dtype=np.float64).reshape(len(ids),784)/255.0
    return np.repeat(current[:,None,:],100,axis=1)


class Atlas:
    def __init__(self,root,weights,seed):
        from atlas_adapter import load_api
        self.root=root;self.weights=[w.tolist() for w in weights];self.api=load_api(root/'snapshot/brian2-rust')
        self.plan=self.api.lif_training_plan([784,128,10],backend='cpu',beta=.95,threshold=1.,
            reset='subtract',detach_reset=True,surrogate_slope=5.,surrogate_scale=1.,optimizer='adam',
            learning_rate=.001,seed=seed,logit_scale=5.,max_tape_bytes=1024**3)
        self.reset()
    def reset(self):
        self.trainer=self.api.NativeLIFTrainer(self.plan,runner=self.root/'runtime/arm64-r2/b2-train',weights=self.weights)
    def batch(self,x,y,training,diagnostic=False,explicit_initial=None):
        import numpy as np
        from atlas_adapter import admission
        initial=np.zeros((len(y),138),dtype=np.float64) if explicit_initial is None else np.asarray(explicit_initial,dtype=np.float64)
        if initial.shape!=(len(y),138):raise ValueError('initial shape differs from A1 model')
        operation='train' if training else 'evaluate'
        check=admission(self.plan,self.trainer.state,x,y,operation=operation,initial=initial)
        if check['status']!='admitted':raise BudgetRejected(json.dumps(check))
        start=time.perf_counter();raw=self.trainer.execute(x,y,operation=operation,initial=initial)
        elapsed=time.perf_counter()-start
        out=dict(loss=float(raw['loss']),logits=np.asarray(raw['logits']),public_api_s=elapsed,admission=check)
        if explicit_initial is not None:out['first_tick_spikes']=np.asarray(raw['spikes'])[:,0,:]
        if not math.isfinite(out['loss']) or not np.all(np.isfinite(out['logits'])):raise FloatingPointError('nonfinite Atlas batch')
        if diagnostic:
            out.update(gradients=[np.asarray(w) for w in raw['gradients']],weights=[np.asarray(w) for w in raw['state']['weights']],
                       first=[np.asarray(w) for w in raw['state']['first_moment']],second=[np.asarray(w) for w in raw['state']['second_moment']])
        self.trainer.last_result=None
        return out
    def counter(self):return int(self.trainer.state['step'])
    def save(self,path):self.trainer.store(path)
    def restore(self,path):self.trainer.restore(path)
    checkpoint_suffix='.json'


class Torch:
    def __init__(self,root,weights,seed,engine,compiled):
        import torch
        from torch_adapter import Model
        self.torch=torch;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64)
        self.weights=weights
        self.model=Model(dict(sizes=[784,128,10],weights=[w.tolist() for w in weights],beta=.95,theta=1.),engine,layerwise=True)
        self.model.record_states=False
        self.fn=torch.compile(self.model,fullgraph=True) if compiled else self.model
        self.opt=torch.optim.Adam(self.model.weights,lr=.001,betas=(.9,.999),eps=1e-8,foreach=False)
    def reset(self):
        torch=self.torch
        with torch.no_grad():
            for parameter,w in zip(self.model.weights,self.weights):parameter.copy_(torch.as_tensor(w).reshape_as(parameter))
        self.opt.state.clear();self.opt.zero_grad(set_to_none=True)
    def batch(self,x,y,training,diagnostic=False,explicit_initial=None):
        import numpy as np
        torch=self.torch
        tx=torch.as_tensor(x,dtype=torch.float64);ty=torch.as_tensor(y,dtype=torch.long)
        initial=(torch.zeros((len(y),138),dtype=torch.float64) if explicit_initial is None
                 else torch.as_tensor(explicit_initial,dtype=torch.float64))
        if tuple(initial.shape)!=(len(y),138):raise ValueError('initial shape differs from A1 model')
        initial=initial.detach().requires_grad_(training)
        # No dropout/BN exists in this model. Keep the qualified strict-spike
        # neuron path for evaluation; SpikingJelly eval mode substitutes >=0.
        self.model.train(True)
        with torch.set_grad_enabled(training):
            if training:self.opt.zero_grad(set_to_none=True)
            logits,_,_=self.fn(tx,initial);loss=torch.nn.functional.cross_entropy(logits,ty)
            if training:loss.backward();self.opt.step()
        array=lambda a:a.detach().cpu().numpy().reshape(-1).copy()
        out=dict(loss=float(loss.detach()),logits=logits.detach().cpu().numpy().copy())
        if not math.isfinite(out['loss']) or not np.all(np.isfinite(out['logits'])):raise FloatingPointError('nonfinite Torch batch')
        if diagnostic:
            out.update(gradients=[array(w.grad) for w in self.model.weights],weights=[array(w) for w in self.model.weights],
                first=[array(self.opt.state[w]['exp_avg']) for w in self.model.weights],second=[array(self.opt.state[w]['exp_avg_sq']) for w in self.model.weights])
        return out
    def counter(self):
        values=[int(self.opt.state[w]['step']) for w in self.model.weights]
        if len(set(values))!=1:raise Unqualified('Adam bank step counters disagree')
        return values[0]
    def cell_margin_probe(self,voltages):
        torch=self.torch;self.model.train(True);records=[]
        with torch.no_grad():
            exact=torch.tensor([-2.**-20,0.,2.**-20],dtype=torch.float64)
            for i,cell in enumerate(self.model.cells):
                voltage=torch.tensor([voltages],dtype=torch.float64)
                if self.model.engine.startswith('snntorch'):
                    direct=cell.spike_grad(exact);spikes,charged=cell(torch.zeros_like(voltage),voltage)
                else:
                    direct=cell.surrogate_function(exact);cell.v=voltage;spikes=cell(torch.zeros_like(voltage))
                    charged=cell.v+self.model.theta*spikes
                margin=charged-self.model.theta
                records.append(dict(layer=i+1,training_mode=cell.training,grad_enabled=torch.is_grad_enabled(),
                    initial_voltage=voltage.tolist(),actual_charged_voltage=charged.tolist(),actual_margin=margin.tolist(),
                    actual_spikes=spikes.tolist(),actual_exact_margin_input=exact.tolist(),actual_exact_margin_spikes=direct.tolist(),
                    passed=bool((margin==0).any() and torch.equal(spikes,(margin>0).to(spikes.dtype)) and
                                torch.equal(direct,(exact>0).to(direct.dtype)))))
        return records
    def save(self,path):self.torch.save({'weights':[w.detach().clone() for w in self.model.weights],'optimizer':self.opt.state_dict()},path)
    def restore(self,path):
        state=self.torch.load(path,weights_only=True,map_location='cpu')
        with self.torch.no_grad():
            for parameter,w in zip(self.model.weights,state['weights']):parameter.copy_(w)
        self.opt.load_state_dict(state['optimizer'])
    checkpoint_suffix='.pt'


def compare(actual,expected):
    import numpy as np
    a=np.asarray(actual);b=np.asarray(expected)
    return dict(passed=bool(a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.abs(a-b)<=1e-10+1e-8*np.abs(b))),
                actual_shape=list(a.shape),expected_shape=list(b.shape))


def test_ledger_path(args):
    view=args.engine+('-compile' if args.compile else '-eager')
    return args.root/'evidence/a1-arm64-long-r1-test-once'/f'{view}-seed-{args.seed}-{read(args.contract)['original_contract_sha256']}.json'


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
            check_cap(deadline)
        report['cases'][name]=dict(passed=all(c['passed'] for c in checks.values()),checks=checks)
    for count in [8,16]:
        check_cap(deadline);model.reset();ids=order[:count];x=encode(images,ids);y=np.asarray(labels[ids],dtype=np.int64)
        ref=forward_vjp(dict(sizes=[784,128,10],inputs=x,labels=y,weights=weights,beta=.95,theta=1.))
        actual=model.batch(x,y,False)
        checks={k:compare(actual[k],ref[k]) for k in ['loss','logits']}
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
        checks['actual_native_first_tick_spikes']=compare(actual['first_tick_spikes'],ref['spikes'][:,0,:])
        detail['actual_native_first_tick_boundary_spikes']=actual['first_tick_spikes'][:,[0,1,2,128,129,130]].tolist()
    if hasattr(model,'cell_margin_probe'):
        detail['actual_cell_margin_probes']=model.cell_margin_probe(voltages)
        checks['actual_cell_exact_zero_margin']={'passed':all(row['passed'] for row in detail['actual_cell_margin_probes'])}
        # A direct cell probe is eager even when the actual callable is compiled.
        # Also observe the first output threshold through that actual callable.
        # This auxiliary T1 fixture does not replace the full B2/T100 comparison.
        check_cap(deadline);model.reset();boundary_x=x[:,:1,:].copy()
        boundary_case=dict(case,inputs=boundary_x)
        boundary_ref=forward_vjp(boundary_case)
        boundary_actual=model.batch(boundary_x,y,False,explicit_initial=initial)
        expected_first_logits=5.*((.95*initial[:,128:]-1.)>0.).astype(np.float64)
        for field in ['loss','logits']:
            checks['actual_callable_T1_'+field]=compare(boundary_actual[field],boundary_ref[field])
        checks['actual_callable_T1_exact_threshold']={'passed':bool(np.array_equal(boundary_actual['logits'],expected_first_logits))}
        detail['actual_callable_first_tick_probe']=dict(sizes=[784,128,10],B=2,T=1,
            scope='auxiliary boundary qualification through actual eager/compiled no-gradient callable; formal A1 remains T100',
            actual_logits=boundary_actual['logits'].tolist(),oracle_logits=boundary_ref['logits'].tolist(),
            expected_strict_threshold_logits=expected_first_logits.tolist())
    detail['passed']=all(c['passed'] for c in checks.values())
    report['cases']['no_gradient_evaluate_B2_exact_threshold_and_neighbors']=detail
    report['passed']=all(row['passed'] for row in report['cases'].values())
    dump(dest/'a1-shape-qualification.json',report)
    if not report['passed']:raise Unqualified('actual A1 shape qualification failed')
    model.reset();check_cap(deadline)
    return report


def worker(args):
    import numpy as np
    root=args.root;dest=args.output;deadline=args.deadline
    sys.path.insert(0,str(root/'adapters'));sys.path.insert(0,str(root/'tools'))
    from prepare_data import read_idx
    report=dict(schema='a1-mnist-run-arm64-remaining-r1',engine=args.engine,compile=args.compile,seed=args.seed,status='started',
        total_wall_cap_s=args.wall_cap,epochs=[],selection=None,test_status='sealed_no_selected_checkpoint',test_labels_decoded=False,
        primary_timing='Current session wall; any recovery segments recorded separately; no spliced end-to-end timing',
        source_hashes={str(p.relative_to(root)):sha(p) for p in [Path(__file__),root/'adapters/torch_adapter.py',root/'adapters/atlas_adapter.py',root/'adapters/oracle.py']})
    log=None
    def progress(phase):
        report['phase']=phase;report['elapsed_total_s']=time.monotonic()-args.started
        dump(dest/'progress.json',report,replace=True)
    try:
        spec=read(args.contract)
        if spec.get('status')!='frozen_pretraining':raise Unqualified('A1 contract must be reviewed and frozen before execution')
        q=read(args.q0_report)
        if q.get('dense_qualification_status')!='passed':raise Unqualified('matching prior dense Q0 is required')
        if q.get('engine')!=args.engine:raise Unqualified('prior Q0 engine does not match requested engine')
        for rel,h in q['identities'].items():
            if sha(root/rel)!=h:raise Unqualified(f'prior Q0 source identity changed:{rel}')
        if args.engine=='atlas':
            from arm64_artifact_gate_r1 import gate_artifact
            gate_artifact(root/'runtime/arm64-r2/b2-train','runner')
            runtime_hash=sha(root/'runtime/arm64-r2/b2-train');api_hash=sha(root/'snapshot/brian2-rust/python/brian2_rust/training.py')
            for name in q['cases']:
                previous=read(args.q0_report.parent/(name+'.json'))['raw']
                if previous['runner_sha256']!=runtime_hash or previous['training_api_sha256']!=api_hash:
                    raise Unqualified('prior Atlas Q0 belongs to another runtime or training API')
            report['runtime_sha256']=runtime_hash;report['training_api_sha256']=api_hash
        else:
            for name in q['cases']:
                previous=read(args.q0_report.parent/(name+'.json'))['raw']
                if previous.get('engine')!=args.engine or previous.get('compile') is not args.compile:
                    raise Unqualified('prior Torch Q0 engine/compile mode does not match requested view')
        if test_ledger_path(args).exists():raise Unqualified('This formal view/seed already claimed its once-only test; no retry allowed')
        shared=root/'fixtures/a1-arm64-long-r1';manifest=read(shared/'manifest.json')
        if manifest['contract_sha256']!=spec['original_contract_sha256']:raise Unqualified('shared arrays came from a different contract')
        artifact=shared/f'seed-{args.seed}.npz'
        if sha(artifact)!=manifest['files'][artifact.name]['sha256']:raise Unqualified('shared array identity changed')
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
        model=Atlas(root,weights,args.seed) if args.engine=='atlas' else Torch(root,weights,args.seed,args.engine,args.compile)
        report['evaluation_policy']='same training-mode neuron forward path, no_grad only; this model has no dropout or batch normalization'
        if args.engine!='atlas':
            report['versions']={name:importlib.metadata.version(name) for name in ['torch','snntorch' if args.engine.startswith('snntorch') else 'spikingjelly']}
        progress('A1_shape_qualification');qualify(model,weights,images,labels,order[0],dest,deadline)
        report['A1_shape_qualification']='passed';report['qualification_elapsed_s']=time.monotonic()-args.started
        resume=spec.get('resume')
        cursor=None
        if resume:
            if sha(root/resume['cursor_path'])!=resume['cursor_sha256']:
                raise Unqualified('Frozen recovery cursor changed')
            cursor=read(root/resume['cursor_path'])
            checkpoint=root/cursor['checkpoint']
            if sha(checkpoint)!=cursor['checkpoint_sha256']:
                raise Unqualified('Frozen recovery checkpoint changed')
            completed=cursor['completed_epochs']
            if [e['epoch'] for e in completed]!=list(range(1,len(completed)+1)):
                raise Unqualified('Recovery complete epoch sequence differs')
            if any(e['train']['samples']!=55000 or e['validation']['samples']!=5000 for e in completed):
                raise Unqualified('Recovery epoch denominators differ')
            if cursor['phase']=='train':
                if cursor['epoch']!=len(completed)+1 or not 0<cursor['next_offset']<=55000:
                    raise Unqualified('Recovery train cursor differs')
                if cursor['next_offset']!=55000 and cursor['next_offset']%32:
                    raise Unqualified('Recovery batch boundary differs')
                expected=1719*len(completed)+math.ceil(cursor['next_offset']/32)
            elif cursor['phase']=='epoch_complete':
                expected=1719*len(completed)
            else:raise Unqualified('Unknown recovery phase')
            if cursor['adam_step']!=expected:raise Unqualified('Recovery Adam counter differs')
            model.restore(checkpoint)
            if model.counter()!=expected:raise Unqualified('Restored Adam counter differs')
            import shutil
            for name,identity in resume['retained_checkpoints'].items():
                source=root/identity['path']
                if sha(source)!=identity['sha256']:raise Unqualified('Prior best checkpoint changed')
                shutil.copyfile(source,dest/name)
            report['epochs']=copy.deepcopy(completed)
            report['selection']=copy.deepcopy(cursor['selection'])
            report['resume']=dict(resume,restored_step=expected,phase=cursor['phase'],next_offset=cursor['next_offset'])
            dump(dest/'resume-receipt.json',report['resume'])
        log=(dest/'batches.jsonl').open('x')
        def recovery(epoch,samples,correct,total_loss,phase='train'):
            log.flush();os.fsync(log.fileno())
            path=dest/f'recovery-epoch{epoch}-samples{samples}.json'
            if not path.exists():model.save(path)
            with path.open('rb') as durable:os.fsync(durable.fileno())
            state=dict(schema='a1-remaining-recovery-r1',seed=args.seed,phase=phase,epoch=epoch,
                next_offset=samples,training_samples=samples,correct=correct,total_loss=total_loss,
                adam_step=model.counter(),checkpoint=str(path.relative_to(root)),checkpoint_sha256=sha(path),
                completed_epochs=copy.deepcopy(report['epochs']),selection=copy.deepcopy(report['selection']),
                contract_sha256=sha(args.contract),source_directory=str(dest.relative_to(root)),
                batch_log_bytes=log.tell(),scope='Recovery state; complete validation alone permits selection')
            dump(dest/'recovery-cursor.json',state,replace=True)
        def batches(ids,phase,epoch,training,image_source=images,label_source=labels):
            prefix=cursor if cursor and cursor['phase']=='train' and training and epoch==cursor['epoch'] else None
            samples=prefix['training_samples'] if prefix else 0
            correct=prefix['correct'] if prefix else 0
            total_loss=prefix['total_loss'] if prefix else 0.
            start=time.monotonic()
            for offset in range(prefix['next_offset'] if prefix else 0,len(ids),32):
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
                if training and ((offset//32+1)%100==0 or samples==len(ids)):
                    recovery(epoch,samples,correct,total_loss)
            return dict(samples=samples,correct=correct,loss=total_loss/samples,accuracy=correct/samples,elapsed_s=time.monotonic()-start)
        for epoch in range(len(report['epochs'])+1,11):
            epoch_start=time.monotonic();progress('training')
            trained=batches(order[epoch-1],'train',epoch,True)
            if trained['samples']!=55000 or model.counter()!=1719*epoch:raise Unqualified('incomplete epoch or incorrect global Adam count')
            progress('validation');validated=batches(validation,'validation',epoch,False)
            if validated['samples']!=5000:raise Unqualified('partial validation cannot select a checkpoint')
            check_cap(deadline)
            row=dict(epoch=epoch,train=trained,validation=validated,elapsed_total_s=time.monotonic()-args.started)
            if report['selection'] is None or validated['correct']>report['selection']['validation_correct']:
                checkpoint=dest/(f'best-epoch-{epoch}'+model.checkpoint_suffix);model.save(checkpoint);check_cap(deadline)
                with checkpoint.open('rb') as durable:os.fsync(durable.fileno())
                report['selection']=dict(epoch=epoch,validation_correct=validated['correct'],validation_accuracy=validated['accuracy'],
                    checkpoint=checkpoint.name,sha256=sha(checkpoint))
            row['epoch_wall_s']=time.monotonic()-epoch_start;report['epochs'].append(row)
            report['test_status']='not_started_selected_checkpoint_available'
            reserve=1.25*2*validated['elapsed_s']+2.;remaining=deadline-time.monotonic()
            row.update(remaining_s=remaining,test_reserve_estimate_s=reserve,
                       continue_threshold_s=1.10*row['epoch_wall_s']+reserve)
            progress('epoch_complete')
            recovery(epoch,trained['samples'],trained['correct'],trained['loss']*trained['samples'],'epoch_complete')
            if epoch==10:break
        check_cap(deadline)
        if len(report['epochs'])!=10 or report['selection'] is None:raise CapReached('All10epochs required before final test')
        progress('restore_selected_checkpoint');chosen=dest/report['selection']['checkpoint']
        if sha(chosen)!=report['selection']['sha256']:raise Unqualified('selected checkpoint identity changed')
        model.restore(chosen);check_cap(deadline)
        # Durable once-only marker precedes decoding; a killed run cannot retry.
        marker=dict(selected=report['selection'],elapsed_total_s=time.monotonic()-args.started,attempt=1,
                    run_directory=str(dest),seed=args.seed,engine=args.engine,compile=args.compile,contract_sha256=sha(args.contract))
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
    start=time.monotonic();deadline=start+args.wall_cap
    dest=args.output;dest.mkdir(parents=True,exist_ok=False)
    contract=read(args.contract)
    if contract.get('status')!='frozen_pretraining':raise RuntimeError('Draft A1 contract cannot launch training; freeze the reviewed contract first')
    if contract.get('total_wall_cap_seconds_per_seed') is not None or contract.get('profile') != 'arm64-remaining-r1':raise RuntimeError('Independent uncapped contract required')
    import datetime
    end=datetime.datetime.fromisoformat(contract['remote_budget_deadline_utc']).timestamp()
    if time.time()+args.wall_cap > end:raise RuntimeError('Cannot exceed frozen remote reservation')
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR=str(dest/'cache/matplotlib'),
        TMPDIR=str(dest/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(dest/'cache/torchinductor'),TRITON_CACHE_DIR=str(dest/'cache/triton'))
    Path(env['TMPDIR']).mkdir()
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--allow-run','--root',str(args.root),
        '--contract',str(args.contract),'--engine',args.engine,'--seed',str(args.seed),'--output',str(dest),
        '--q0-report',str(args.q0_report),'--wall-cap',str(args.wall_cap),'--started',str(start),'--deadline',str(deadline)]
    if args.compile:command.append('--compile')
    dump(dest/'launch.json',dict(command=command,contract_sha256=sha(args.contract),script_sha256=sha(__file__),
        started_monotonic=start,total_wall_cap_s=args.wall_cap,thread_environment={k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
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
    summary=dict(status=effective,exit_code=code,hard_cap_triggered=timed_out,elapsed_total_s=time.monotonic()-start,
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
    parser.add_argument('--prepare-shared',action='store_true');parser.add_argument('--allow-run',action='store_true')
    parser.add_argument('--engine',choices=ENGINES);parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--output',type=Path);parser.add_argument('--q0-report',type=Path);parser.add_argument('--compile',action='store_true')
    parser.add_argument('--wall-cap',type=float,required=True,help='Remaining shared reservation budget, not a per-seed 30min cap')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--started',type=float);parser.add_argument('--deadline',type=float)
    args=parser.parse_args();host_guard();args.root=args.root.resolve();args.contract=args.contract.resolve()
    if not math.isfinite(args.wall_cap) or args.wall_cap<=0:parser.error("Positive finite reservation budget required")
    if args.prepare_shared:prepare_shared(args);return 0
    if not args.allow_run:parser.error('Training requires explicit --allow-run and frozen contract')
    if any(value is None for value in [args.engine,args.seed,args.output,args.q0_report]):parser.error('Run requires engine, seed, output and q0-report')
    if args.compile and args.engine=='atlas':parser.error('--compile is a separate Torch view only')
    args.output=args.output.resolve();args.q0_report=args.q0_report.resolve()
    if args.worker:
        if args.started is None or args.deadline is None:parser.error('Worker requires coordinator wall-clock bounds')
        if not math.isclose(args.deadline-args.started,args.wall_cap,rel_tol=0,abs_tol=1e-6):parser.error('Worker must retain the shared reservation deadline')
        return worker(args)
    return supervise(args)


if __name__=='__main__':raise SystemExit(main())
