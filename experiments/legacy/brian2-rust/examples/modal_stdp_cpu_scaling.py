"""Measure actual CPU parallelism and fully gated CPU/GPU STDP replays."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def compare(neurons,steps,degree,repeats,native_backend='cuda'):
    import os,random,select,signal,subprocess,sys,tempfile,statistics
    import numpy as np
    sys.path.insert(0,'/workspace/brian2-rust/examples')
    from gpu_stdp_precompiled import BACKENDS as DEFAULT_BACKENDS
    worker_script=Path(sys.modules['gpu_stdp_precompiled'].__file__).resolve()
    from gpu_stdp_compare import configuration,oracle,checks as stdp_checks
    BACKENDS=('rust-f64','cpp-f64-t1','cpp-f64-t2','cpp-f64-t4','cpu-f32',native_backend)
    PRECISE_BASIC_FLAGS='--fmad=false --ftz=false --prec-div=true --prec-sqrt=true'
    config=configuration(neurons,degree,steps)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    report=dict(schema='b2-stdp-cpu-scaling-v1',configuration=config,declared_input_sha256=fingerprint,
        repeats=repeats,order_seed=1729,backends=list(BACKENDS),rust_requested_thread_probes=[],workers={},samples=[],payloads={},status='running',
        scope='precompiled reset/initialization through completed host result arrays; includes process launch/file IO where required and GeNN unload; excludes compilation, frontend/model construction, protocol, hashes and NPZ serialization',
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True) if __import__('shutil').which('nvidia-smi') else None,
        host_cpu=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else __import__('platform').platform(),allocated_cpu_cores=4 if Path('/proc/cpuinfo').exists() else None,
        locks={p.name:p.read_text() for p in Path('/opt/baseline-locks').glob('*.txt')})
    def resource_state():
        return dict(allowed_cpus=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            cgroup={str(p):p.read_text() if p.exists() else None for p in map(Path,('/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu.stat','/sys/fs/cgroup/cpuset.cpus.effective'))})
    report['resources_before']=resource_state()
    interpreters={b:sys.executable for b in BACKENDS}
    workers={};logs={};rng=random.Random(1729)
    def receive(backend,timeout=300):
        process=workers[backend]
        if not select.select([process.stdout],[],[],timeout)[0]:raise TimeoutError(backend+' worker response timeout')
        line=process.stdout.readline()
        if not line:raise RuntimeError(backend+' worker exited before response')
        value=json.loads(line)
        if value['event']=='error':raise RuntimeError(backend+': '+value['error_type']+': '+value['error'])
        return value
    def request(backend,message):
        p=workers[backend];p.stdin.write(json.dumps(message)+'\n');p.stdin.flush();return receive(backend)
    def capture(artifact,name):
        payload=Path(artifact['path']).read_bytes();assert hashlib.sha256(payload).hexdigest()==artifact['sha256']
        result=dict(np.load(artifact['path']));report['payloads'][name]=payload
        artifact['path']=name
        return result
    reference=None;f64_reference=oracle(neurons,degree,steps)
    def checks(backend,actual):
        control=f64_reference if backend=='rust-f64' else reference
        result=stdp_checks(actual,control)
        result['control']='independent-f64' if backend=='rust-f64' else 'compiled-cpu-f32'
        result['original_f64_gate']={k:v['passed'] for k,v in stdp_checks(actual,f64_reference)['fields'].items()}
        return result
    with tempfile.TemporaryDirectory(prefix='b2-stdp-precompiled-') as temporary:
        try:
            # Prepare CPU f32 first to establish the numerical gate. All six
            # workers stay alive; no GPU simulation runs concurrently.
            preparation_order=['cpu-f32']+[b for b in BACKENDS if b!='cpu-f32']
            for backend in preparation_order:
                output=Path(temporary)/backend;log=Path(temporary)/(backend+'.log');logs[backend]=log
                env={**os.environ,'CUDA_PATH':'/usr/local/cuda','NVCC_APPEND_FLAGS':PRECISE_BASIC_FLAGS,
                    'OMP_DYNAMIC':'FALSE','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
                env.pop('NVCC_PREPEND_FLAGS',None)
                if backend in {'genn','brian2cuda','brian2genn','brian2genn-corrected'}:env['PYTHONPATH']=''
                if backend in {'brian2genn','brian2genn-corrected'}:env['PATH']='/opt/genn4/bin:'+env['PATH']
                with log.open('w') as stderr:
                    workers[backend]=subprocess.Popen([interpreters[backend],str(worker_script),
                        '--backend',backend,'--neurons',str(neurons),'--steps',str(steps),'--degree',str(degree),'--output',str(output)],
                        cwd=temporary,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,start_new_session=True)
                ready=receive(backend);assert ready['event']=='ready' and ready['configuration']==config
                actual=capture(ready['bootstrap'],backend+'-bootstrap.npz')
                if backend=='cpu-f32':reference=actual
                ready['checks']=checks(backend,actual);report['workers'][backend]=ready
                if not ready['checks']['passed'] or not all(ready['checks']['original_f64_gate'].values()):raise RuntimeError(backend+' bootstrap violates declared numerical gate')
                print('prepared',backend,flush=True)
            for requested in (2,4):
                sample=request('rust-f64',dict(command='run',sample=-requested,rust_threads=requested))
                actual=capture(sample['result'],f'rust-requested-{requested}.npz')
                sample['checks']=checks('rust-f64',actual)
                with np.load(Path(temporary)/'rust-f64/bootstrap.npz') as z:bootstrap=dict(z)
                sample['bitwise_bootstrap']=all(np.array_equal(actual[k],bootstrap[k]) for k in bootstrap)
                sample['requested_threads']=requested;report['rust_requested_thread_probes'].append(sample)
                if not sample['checks']['passed'] or not sample['bitwise_bootstrap']:raise RuntimeError('Rust thread probe changed results')
            # One explicit unmeasured-for-summary warmup per backend, then
            # randomized paired rounds on the same physical GPU allocation.
            for round_index in range(-1,repeats):
                order=list(BACKENDS);rng.shuffle(order)
                for position,backend in enumerate(order):
                    sample=request(backend,dict(command='run',sample=round_index))
                    assert sample['event']=='result' and sample['sample']==round_index
                    actual=capture(sample['result'],f'{backend}-{round_index}.npz')
                    sample.update(backend=backend,round=round_index,order=position,checks=checks(backend,actual))
                    report['samples'].append(sample)
                    if not sample['checks']['passed'] or not all(sample['checks']['original_f64_gate'].values()):raise RuntimeError(backend+' replay violates declared numerical gate')
                print('round',round_index,'complete',flush=True)
            report['status']='passed-matched-numeric-gates'
        except BaseException as error:
            report.update(status='failed',error_type=type(error).__name__,error=str(error))
        finally:
            for backend,p in workers.items():
                try:
                    if p.poll() is None:
                        p.stdin.write('{"command":"close"}\n');p.stdin.flush()
                        reply=receive(backend,30);assert reply['event']=='closed'
                    code=p.wait(timeout=10)
                    report.setdefault('worker_exit_codes',{})[backend]=code
                except BaseException as error:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
                    p.wait(timeout=10);report.setdefault('shutdown_errors',{})[backend]=str(error)
            for backend,log in logs.items():report.setdefault('worker_logs',{})[backend]=log.read_text()
    if report.get('shutdown_errors') or any(code!=0 for code in report.get('worker_exit_codes',{}).values()):report['status']='failed'
    report['original_f64_gate_passed_all_samples']=bool(report['samples']) and all(all(r['checks']['original_f64_gate'].values()) for r in report['samples'])
    report['resources_after']=resource_state()
    report['summary']=[]
    for backend in BACKENDS:
        rows=[r for r in report['samples'] if r['backend']==backend and r['round']>=0]
        values=[r['wall_seconds'] for r in rows]
        report['summary'].append(dict(backend=backend,samples_seconds=values,
            complete=len(rows)==repeats,matched_gate_passed=len(rows)==repeats and all(r['checks']['passed'] for r in rows),
            median_seconds=statistics.median(values) if values else None,
            min_seconds=min(values) if values else None,max_seconds=max(values) if values else None))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4')
    parser.add_argument('--neurons',type=int,default=4096);parser.add_argument('--steps',type=int,default=512)
    parser.add_argument('--degree',type=int,default=32);parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from gpu_stdp_compare import topology
    if not 2<=args.neurons<=4096 or not 1<=args.steps<=512 or not 1<=args.repeats<=10:parser.error('bounded workload and 1..10 repeats required')
    try:topology(args.neurons,args.degree)
    except ValueError as error:parser.error(str(error))
    args.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    names=('gpu_baseline.py','gpu_stdp_compare.py','gpu_stdp_precompiled.py','modal_stdp_cpu_scaling.py','modal_cuda_tests.py','threaded_benchmark.py')
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT/'src',ROOT.parent/'brian2') for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for p in [*(ROOT/'examples'/n for n in names),ROOT/'Cargo.toml',ROOT/'Cargo.lock',
              *(ROOT.parent/n for n in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        hashes[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output/'source-hashes.json').write_text(json.dumps(hashes,sort_keys=True,indent=2)+'\n')
    image=native_image().add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=['**/__pycache__/**','**/*.pyc','**/*.so','**/*.dylib'])
    for n in names:
        image=image.add_local_file(ROOT/'examples'/n,'/workspace/brian2-rust/examples/'+n)
    image=image.add_local_file(Path(__file__),'/root/modal_stdp_cpu_scaling.py')
    app=modal.App('brian2-stdp-cpu-scaling',include_source=False)
    run=app.function(image=image,gpu=args.gpu,cpu=4,memory=8192,max_containers=1,timeout=1200,retries=0)(compare)
    start=time.perf_counter()
    with modal.enable_output(),app.run():report=run.remote(args.neurons,args.steps,args.degree,args.repeats)
    report.update(requested_gpu=args.gpu,remote_wall_seconds=time.perf_counter()-start,
        source_manifest_sha256=hashlib.sha256((args.output/'source-hashes.json').read_bytes()).hexdigest())
    for name,payload in report.pop('payloads').items():(args.output/name).write_bytes(payload)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],report['summary'])
    if report['status']!='passed-matched-numeric-gates':raise SystemExit(1)


if __name__=='__main__':main()
