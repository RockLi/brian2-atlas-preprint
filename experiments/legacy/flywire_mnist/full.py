"""Resumable, sharded full-MNIST experiment with a locked official test.

Stages: init -> extract train -> fit -> extract test -> evaluate.
One coordinator owns a run; independent CPU workers own disjoint shards.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import numpy as np
from . import dataset
from .config import Config,digest
from .encoding import encode,projection
from .protocol import atomic_json,file_sha,source_identity,training_data,test_report


def read(path):return json.loads(Path(path).read_text())


def init(args):
    from .model import build
    from .graph import load_graph
    from .backends import CPU
    config=Config(**(read(args.config) if args.config else {}))
    graph=load_graph(args.graph)
    _,_,_,_,_,split=training_data(args.data,config.seed)
    model=build(graph,config,recurrent=not args.no_recurrence)
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output/'split.json',split)
    atomic_json(args.output/'model.json',model)
    readouts=graph.readouts(config.kc_count,config.seed)
    np.savez(args.output/'cells.npz',inputs=graph.inputs,readouts=readouts)
    CPU(model,args.output/'execution',threads=1)
    protocol={'schema':'flywire-mnist-full-v1','backend':'cpu-f64','config':asdict(config),
              'scope':graph.scope,'graph':graph.identity,'neurons':len(graph.root_ids),'edges':graph.edge_count,
              'recurrent':not args.no_recurrence,'split_sha256':split['sha256'],
              'model_sha256':file_sha(args.output/'model.json'),'cells_sha256':file_sha(args.output/'cells.npz'),
              'native_binary_sha256':file_sha(args.output/'execution/native/b2-native'),
              'native_source_sha256':file_sha(args.output/'execution/native/main.rs'),
              'experiment_sources':source_identity(),'shard_size':args.shard_size,
              'feature_dimension':len(readouts)*config.bins,
              'projected_dimension':len(graph.inputs)*config.bins,
              'readout_alpha_budget':[.1,1.,10.,100.],
              'final_fit':'select alpha on 50000/10000, then fit all 60000; never select on official test'}
    protocol['sha256']=digest(protocol)
    atomic_json(args.output/'protocol.json',protocol)
    (args.output/'shards').mkdir()
    atomic_json(args.output/'progress.json',{'phase':'initialized','training_images':60000,'test_images':10000})


def validate(root):
    p=read(root/'protocol.json')
    if p['sha256']!=digest({k:v for k,v in p.items() if k!='sha256'}):raise ValueError('protocol hash mismatch')
    for name,key in [('model.json','model_sha256'),('cells.npz','cells_sha256'),
                     ('execution/native/b2-native','native_binary_sha256'),('execution/native/main.rs','native_source_sha256')]:
        if file_sha(root/name)!=p[key]:raise ValueError(f'frozen artifact changed: {name}')
    if source_identity()!=p['experiment_sources']:raise ValueError('experiment source changed; create a new run')
    split=read(root/'split.json')
    if split['sha256']!=p['split_sha256'] or digest({k:v for k,v in split.items() if k!='sha256'})!=split['sha256']:
        raise ValueError('split changed')
    return p,split


def phase_ids(split,phase):
    return np.r_[split['fit_ids'],split['validation_ids']].astype(np.int64) if phase=='train' else np.arange(60000,70000)


def shard_digest(ids,x,projected):
    h=hashlib.sha256()
    for a in (ids,x,projected):h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def load_shard(path,ids,protocol):
    with np.load(path,allow_pickle=False) as f:
        actual=f['sample_ids'];x=f['features'];projected=f['projected']
        if str(f['protocol'])!=protocol['sha256'] or not np.array_equal(actual,ids):raise ValueError('shard identity mismatch')
        if x.shape!=(len(ids),protocol['feature_dimension']) or projected.shape!=(len(ids),protocol['projected_dimension']):
            raise ValueError('shard shape mismatch')
        if x.dtype!=np.uint16 or projected.dtype!=np.uint16:raise ValueError('shard dtype mismatch')
        if str(f['payload_sha256'])!=shard_digest(actual,x,projected):raise ValueError('shard checksum mismatch')
        return x.copy(),projected.copy()


_worker=None


def worker_init(root,data,phase):
    global _worker
    from .backends.frozen_cpu import FrozenCPU
    root=Path(root);p,split=validate(root);config=Config(**p['config'])
    model=read(root/'model.json')
    images,_,_=dataset.load(data,test=phase=='test')
    if phase=='train':
        for name,expected in split['training_files'].items():
            if file_sha(Path(data)/name)!=expected:raise ValueError('training data changed')
    work=root/'workers'/f'{phase}-{os.getpid()}';work.parent.mkdir(exist_ok=True)
    if work.exists():shutil.rmtree(work)
    runner=FrozenCPU(model,root/'execution/native',work)
    with np.load(root/'cells.npz',allow_pickle=False) as f:inputs=f['inputs'].copy();readouts=f['readouts'].copy()
    src,dst=projection(inputs,config.fanout,config.seed)
    slots=np.searchsorted(np.sort(inputs),dst)
    _worker=(root,p,config,model,images,readouts,runner,src,slots,phase)


def worker_shard(task):
    from .model import features
    offset,ids=task;root,p,c,model,images,readouts,runner,src,slots,phase=_worker
    rows=[];controls=[];seconds=[]
    started=time.perf_counter()
    for sample_id in ids:
        sample_id=int(sample_id);image=images[sample_id-(60000 if phase=='test' else 0)]
        indices,ticks=encode(image,sample_id,c)
        result=runner.run(indices,ticks,f'sample-{sample_id}')
        row=features(model,result,readouts,c.bins)
        control=[]
        for left,right in zip(c.edges,c.edges[1:]):
            counts=np.bincount(indices[(ticks>=left)&(ticks<right)],minlength=784)
            control.append(np.bincount(slots,weights=counts[src],minlength=p['projected_dimension']//c.bins))
        control=np.concatenate(control)
        if max(row.max(initial=0),control.max(initial=0))>65535:raise OverflowError('spike counter archive overflow')
        rows.append(row.astype(np.uint16));controls.append(control.astype(np.uint16));seconds.append(result['wall_seconds'])
        del result
        shutil.rmtree(runner.directory/f'sample-{sample_id}')
        (runner.directory/f'sample-{sample_id}.spikes').unlink()
    x=np.asarray(rows);projected=np.asarray(controls)
    path=root/'shards'/f'{phase}-{offset:06d}.npz';temp=path.with_suffix('.partial')
    with temp.open('wb') as stream:
        np.savez_compressed(stream,sample_ids=ids,features=x,projected=projected,protocol=p['sha256'],
                            payload_sha256=shard_digest(ids,x,projected),sample_seconds=np.array(seconds))
        stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
    return {'offset':offset,'count':len(ids),'wall_seconds':time.perf_counter()-started,
            'mean_sample_seconds':float(np.mean(seconds))}


def require_test_lock(root,p):
    lock=read(root/'test-lock.json')
    if lock['protocol_sha256']!=p['sha256']:raise ValueError('test lock does not match protocol')
    expected={'readout-flywire.npz','readout-projected_input.npz','readout-pixels.npz'}
    if (lock.get('fit_count')!=60000 or set(lock.get('readout_sha256',{}))!=expected
            or len(lock.get('training_shards_sha256',{}))!=(60000+p['shard_size']-1)//p['shard_size']):
        raise ValueError('incomplete full-dataset test lock')
    for name,expected in lock['readout_sha256'].items():
        if file_sha(root/name)!=expected:raise ValueError('readout changed after test lock')
    return lock


def extract(args):
    p,split=validate(args.output)
    if args.phase=='test':require_test_lock(args.output,p)
    ids=phase_ids(split,args.phase);tasks=[];done=0
    for offset in range(0,len(ids),p['shard_size']):
        chosen=ids[offset:offset+p['shard_size']];path=args.output/'shards'/f'{args.phase}-{offset:06d}.npz'
        if path.exists():load_shard(path,chosen,p);done+=len(chosen)
        else:tasks.append((offset,chosen))
    if args.max_shards is not None:tasks=tasks[:args.max_shards]
    started=time.perf_counter();initial=done
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init,initargs=(str(args.output),str(args.data),args.phase)) as pool:
        futures=[pool.submit(worker_shard,task) for task in tasks]
        for future in as_completed(futures):
            result=future.result();done+=result['count'];elapsed=time.perf_counter()-started
            progress={'phase':'extract-'+args.phase,'complete_images':done,'total_images':len(ids),
                      'workers':args.workers,'elapsed_seconds_this_launch':elapsed,
                      'images_per_second_this_launch':(done-initial)/elapsed,'last_shard':result}
            atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    # Worker snapshots are recreatable and must not accumulate across resumes.
    if (args.output/'workers').exists():shutil.rmtree(args.output/'workers')


def all_features(root,p,split,phase):
    ids=phase_ids(split,phase)
    x=np.lib.format.open_memmap(root/f'{phase}-features.npy',mode='w+',dtype=np.uint16,
                               shape=(len(ids),p['feature_dimension']))
    projected=np.lib.format.open_memmap(root/f'{phase}-projected.npy',mode='w+',dtype=np.uint16,
                                       shape=(len(ids),p['projected_dimension']))
    hashes={}
    for offset in range(0,len(ids),p['shard_size']):
        path=root/'shards'/f'{phase}-{offset:06d}.npz';chosen=ids[offset:offset+p['shard_size']]
        row,control=load_shard(path,chosen,p);x[offset:offset+len(chosen)]=row;projected[offset:offset+len(chosen)]=control
        hashes[path.name]=file_sha(path)
    x.flush();projected.flush()
    return ids,x,projected,hashes


def fit_readouts(args):
    from threadpoolctl import threadpool_limits
    from .readout import ridge_path,predict,metrics
    p,split=validate(args.output)
    if (args.output/'test-lock.json').exists():raise ValueError('test already locked; use a new experiment to retrain')
    images,labels,_,_,_,current=training_data(args.data,Config(**p['config']).seed)
    if current['sha256']!=split['sha256']:raise ValueError('training split mismatch')
    ids,x,projected,hashes=all_features(args.output,p,split,'train');y=labels[ids]
    controls={'flywire':x,'projected_input':projected,'pixels':images[ids].reshape(60000,784)}
    selection={};readout_hashes={}
    with threadpool_limits(limits=args.threads):
        for name,values in controls.items():
            best=(-1,None);trials=[]
            for classifier in ridge_path(values[:50000],y[:50000],p['readout_alpha_budget']):
                score=metrics(y[50000:],predict(classifier,values[50000:]))
                alpha=float(classifier['alpha']);trials.append({'alpha':alpha,**score})
                if score['accuracy']>best[0]:best=(score['accuracy'],alpha)
            classifier=next(ridge_path(values,y,[best[1]]))
            path=args.output/f'readout-{name}.npz';np.savez_compressed(path,**classifier)
            readout_hashes[path.name]=file_sha(path)
            selection[name]={'selected_alpha':best[1],'validation_accuracy':best[0],'trials':trials}
            print(json.dumps({'readout':name,'alpha':best[1],'validation_accuracy':best[0]}),flush=True)
    atomic_json(args.output/'selection.json',selection)
    atomic_json(args.output/'test-lock.json',{'protocol_sha256':p['sha256'],'readout_sha256':readout_hashes,
                'training_shards_sha256':hashes,'selection_sha256':file_sha(args.output/'selection.json'),
                'fit_count':60000,'test_used_before_lock':False})


def evaluate(args):
    from .readout import predict
    p,split=validate(args.output);lock=require_test_lock(args.output,p)
    if (args.output/'report.json').exists():raise ValueError('official test report already exists')
    if file_sha(args.output/'selection.json')!=lock['selection_sha256']:raise ValueError('selection changed')
    for name,expected in lock['training_shards_sha256'].items():
        if file_sha(args.output/'shards'/name)!=expected:raise ValueError('training shard changed after lock')
    ids,x,projected,hashes=all_features(args.output,p,split,'test')
    images,labels,official_ids=dataset.load(args.data,test=True)
    if not np.array_equal(ids,official_ids):raise ValueError('test sample order mismatch')
    report={'protocol_sha256':p['sha256'],'scope':p['scope'],'fit_count':60000,'test_count':10000,
            'results':{},'test_shards_sha256':hashes,'note':'Fixed internal network; only linear readout trained.'}
    for name,values in [('flywire',x),('projected_input',projected),('pixels',images.reshape(10000,784))]:
        with np.load(args.output/f'readout-{name}.npz',allow_pickle=False) as classifier:prediction=predict(classifier,values)
        report['results'][name]=test_report(labels,prediction)
        np.savez_compressed(args.output/f'test-predictions-{name}.npz',sample_ids=ids,labels=labels,predicted=prediction)
    atomic_json(args.output/'report.json',report);atomic_json(args.output/'progress.json',{'phase':'complete'})
    print(json.dumps({k:v['accuracy'] for k,v in report['results'].items()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['init','extract','fit','evaluate'])
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--graph',type=Path);parser.add_argument('--config',type=Path)
    parser.add_argument('--no-recurrence',action='store_true');parser.add_argument('--shard-size',type=int,default=128)
    parser.add_argument('--phase',choices=['train','test'],default='train');parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--threads',type=int,default=4);parser.add_argument('--max-shards',type=int)
    args=parser.parse_args();args.output=args.output.resolve();args.data=args.data.resolve()
    if args.workers<1 or args.threads<1 or not 1<=args.shard_size<=4096:parser.error('invalid resource/shard settings')
    if args.max_shards is not None and args.max_shards<1:parser.error('max-shards must be positive')
    if args.stage=='init':
        if not args.graph:parser.error('init requires --graph')
        init(args);return
    # OS advisory lock is released on crash, unlike stale sentinel lock files.
    import fcntl
    with (args.output/'coordinator.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'extract':extract,'fit':fit_readouts,'evaluate':evaluate}[args.stage](args)


if __name__=='__main__':main()
