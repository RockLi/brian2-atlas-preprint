"""Bounded sequential supervisor for the explicitly authorized Mac Studio.

All logs/results survive timeout; no cleanups of other jobs or shared caches.
"""
import argparse,datetime,hashlib,json,os,signal,socket,subprocess,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def launch(name,command,timeout,folder,env):
    import psutil
    record=dict(name=name,command=command,timeout_s=timeout,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    start=time.monotonic();samples=[]
    with (folder/(name+'.log')).open('x') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        while proc.poll() is None:
            try:
                process=psutil.Process(proc.pid);members=[process]+process.children(recursive=True)
                rss={str(p.pid):p.memory_info().rss for p in members if p.is_running()}
                samples.append(dict(elapsed_s=time.monotonic()-start,aggregate_rss=sum(rss.values()),rss_by_pid=rss,root_threads=process.num_threads()))
            except psutil.Error:pass
            if time.monotonic()-start>=timeout:
                os.killpg(proc.pid,signal.SIGKILL);record['termination_reason']='timeout';break
            time.sleep(.05)
        record['exit_code']=proc.wait()
    record['elapsed_s']=time.monotonic()-start
    record['termination_reason']=record.get('termination_reason','exited')
    record['peak_job_rss_same_sample']=max((s['aggregate_rss'] for s in samples),default=None)
    (folder/(name+'-resource.json')).write_text(json.dumps(dict(sampling_interval_s=.05,samples=samples),separators=(',',':'))+'\n')
    (folder/(name+'-terminal.json')).write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['qualify','benchmark'],required=True);args=parser.parse_args()
    assert socket.gethostname()=='rock-mac-studio-1.local', 'Complex runs are restricted to the user-selected host.'
    os.chdir(ROOT)
    folder=ROOT/'evidence'/('remote-'+args.phase+'-v1');folder.mkdir(exist_ok=False,parents=True)
    cpu=str(ROOT/'environment/cpu/bin/python');jax=str(ROOT/'environment/jax/bin/python')
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),TMPDIR=str(ROOT/'cache/tmp'),TORCHINDUCTOR_CACHE_DIR=str(ROOT/'cache/torchinductor'),TRITON_CACHE_DIR=str(ROOT/'cache/triton'),JAX_COMPILATION_CACHE_DIR=str(ROOT/'cache/jax'),JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
    views=[('atlas',cpu,'atlas',[]),('sj-layerwise',cpu,'spikingjelly_frontier',['--layerwise']),('snn-layerwise',cpu,'snntorch_fp64',['--layerwise']),('spyx',jax,'spyx',[]),('brainstate',jax,'brainx_state',[]),('sj-compile',cpu,'spikingjelly_frontier',['--layerwise','--compile']),('snn-compile',cpu,'snntorch_fp64',['--layerwise','--compile'])]
    identity=dict(host=socket.gethostname(),phase=args.phase,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_manifest=digest(ROOT/'sources/snapshot-manifest.json'),runtime_sha256=digest(ROOT/'runtime/b2-train'),scripts={str(p.relative_to(ROOT)):digest(p) for parent in ['adapters','tools'] for p in sorted((ROOT/parent).glob('*.py'))},locks={p.name:digest(p) for p in (ROOT/'environment').glob('*lock.txt')},hardware=digest(ROOT/'environment/hardware.json'),views=views)
    (folder/'freeze.json').write_text(json.dumps(identity,indent=2)+'\n')
    records=[]
    def run(name,command,cap):
        print('START',name,flush=True)
        jobenv=env.copy()
        if args.phase=='benchmark':
            jobenv['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'cache/benchmark'/name/'torchinductor')
            jobenv['JAX_COMPILATION_CACHE_DIR']=str(ROOT/'cache/benchmark'/name/'jax')
        row=launch(name,command,cap,folder,jobenv);records.append(row)
        (folder/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('END',name,row['exit_code'],row['termination_reason'],flush=True)
    if args.phase=='qualify':
        for name,python,engine,opts in views:
            script='qualify_jax.py' if python==jax else 'qualify_dense.py'
            run('q0-'+name,[python,str(ROOT/'tools'/script),'--engine',engine,'--output',str(folder/('q0-'+name)),*opts],600 if '--compile' in opts else 300)
        run('atlas-graph',[cpu,str(ROOT/'adapters/atlas_graph_qualification.py'),'--base',str(ROOT),'--output',str(folder/'atlas-graph.json')],300)
    else:
        qfolder=ROOT/'evidence/remote-qualify-v1'
        # Round-robin rotations fixed before seeing a performance sample.
        plan=[]
        for i,seed in enumerate([11,23,37,51,71]):
            ordered=views[i:]+views[:i]
            for view in ordered:plan.append((seed,view))
        (folder/'order.json').write_text(json.dumps(plan,indent=2)+'\n')
        spent={name:0. for name,_,_,_ in views}
        for seed,(name,python,engine,opts) in plan:
            qpath=qfolder/('q0-'+name)/'report.json'
            if not qpath.exists() or json.loads(qpath.read_text()).get('dense_qualification_status')!='passed':
                records.append(dict(name=name,seed=seed,execution_status='unqualified',reason='matching remote dense Q0 has not passed'));continue
            q=json.loads(qpath.read_text())
            for rel,h in q['identities'].items():
                assert digest(ROOT/rel)==h, f'Qualification implementation changed: {rel}'
            cap=min(360,1800-spent[name])
            if cap<=0:
                records.append(dict(name=name,seed=seed,execution_status='timeout',reason='per-view 1800s budget exhausted'));continue
            output=folder/f'{name}-seed-{seed}'
            if name=='atlas':command=[cpu,str(ROOT/'tools/benchmark_e1.py'),'--worker','--base',str(ROOT),'--source-root',str(ROOT/'snapshot/brian2-rust'),'--runner',str(ROOT/'runtime/b2-train'),'--output',str(output),'--seed',str(seed)]
            else:command=[python,str(ROOT/'tools/benchmark_competitor.py'),'--engine',engine,'--seed',str(seed),'--output',str(output),*opts]
            run(f'{name}-seed-{seed}',command,cap);spent[name]+=records[-1]['elapsed_s']
    (folder/'terminal.json').write_text(json.dumps(dict(supervisor_completed=True,records=records),indent=2)+'\n')


if __name__=='__main__':main()
