"""Interleave precompiled reset-to-result replays across six backend workers."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def compare(neurons,steps,degree,repeats):
    import os,random,select,signal,subprocess,sys,tempfile,statistics
    import numpy as np
    sys.path.insert(0,'/workspace/brian2-rust/examples')
    from gpu_precompiled import BACKENDS
    from gpu_recurrent import configuration,arrays,oracle
    from modal_gpu_compare import paired_checks,PRECISE_BASIC_FLAGS
    initial,drive,projections=arrays(neurons,degree);config=configuration(neurons,steps,degree)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()+initial.tobytes()+drive.tobytes()+
        b''.join(s.tobytes()+t.tobytes()+w.tobytes() for s,t,w in projections)).hexdigest()
    report=dict(schema='b2-precompiled-comparison-v0',configuration=config,declared_input_sha256=fingerprint,
        repeats=repeats,order_seed=1729,workers={},samples=[],payloads={},status='running',
        scope='precompiled reset/initialization through completed host result arrays; includes process launch/file IO where required and GeNN unload; excludes compilation, frontend/model construction, protocol, hashes and NPZ serialization',
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),
        host_cpu=Path('/proc/cpuinfo').read_text(),allocated_cpu_cores=4,
        locks={p.name:p.read_text() for p in Path('/opt/baseline-locks').glob('*.txt')})
    interpreters={b:sys.executable for b in BACKENDS}
    interpreters.update(brian2cuda='/opt/brian2cuda-env/bin/python',brian2genn='/opt/brian2genn-env/bin/python',genn='/opt/genn5-env/bin/python')
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
    reference=None;f64_reference=oracle(neurons,steps,degree)
    def checks(backend,actual):
        control=f64_reference if backend=='rust-f64' else reference
        result=paired_checks(actual,control,'recurrent-cuba-v0',neurons,steps)
        result['control']='independent-f64' if backend=='rust-f64' else 'compiled-cpu-f32'
        result['original_f64_gate']=paired_checks(actual,f64_reference,'recurrent-cuba-v0',neurons,steps)['checks']
        return result
    with tempfile.TemporaryDirectory(prefix='b2-precompiled-') as temporary:
        try:
            # Prepare CPU f32 first to establish the numerical gate. All six
            # workers stay alive; no GPU simulation runs concurrently.
            preparation_order=['cpu-f32']+[b for b in BACKENDS if b!='cpu-f32']
            for backend in preparation_order:
                output=Path(temporary)/backend;log=Path(temporary)/(backend+'.log');logs[backend]=log
                env={**os.environ,'CUDA_PATH':'/usr/local/cuda','NVCC_APPEND_FLAGS':PRECISE_BASIC_FLAGS}
                env.pop('NVCC_PREPEND_FLAGS',None)
                if backend in {'genn','brian2cuda','brian2genn'}:env['PYTHONPATH']=''
                if backend=='brian2genn':env['PATH']='/opt/genn4/bin:'+env['PATH']
                with log.open('w') as stderr:
                    workers[backend]=subprocess.Popen([interpreters[backend],'/workspace/brian2-rust/examples/gpu_precompiled.py',
                        '--backend',backend,'--neurons',str(neurons),'--steps',str(steps),'--degree',str(degree),'--output',str(output)],
                        cwd=temporary,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,start_new_session=True)
                ready=receive(backend);assert ready['event']=='ready' and ready['configuration']==config
                actual=capture(ready['bootstrap'],backend+'-bootstrap.npz')
                if backend=='cpu-f32':reference=actual
                ready['checks']=checks(backend,actual);report['workers'][backend]=ready
                if not ready['checks']['passed']:raise RuntimeError(backend+' bootstrap violates declared numerical gate')
                print('prepared',backend,flush=True)
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
                    if not sample['checks']['passed']:raise RuntimeError(backend+' replay violates declared numerical gate')
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
    parser.add_argument('--neurons',type=int,default=4096);parser.add_argument('--steps',type=int,default=2048)
    parser.add_argument('--degree',type=int,default=32);parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from gpu_recurrent import validate_size
    if not 1<=args.neurons<=65536 or not 1<=args.steps<=4096 or not 1<=args.repeats<=10:parser.error('bounded workload and 1..10 repeats required')
    try:validate_size(args.neurons,args.degree)
    except ValueError as error:parser.error(str(error))
    args.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    names=('gpu_baseline.py','gpu_recurrent.py','gpu_hh.py','gpu_precompiled.py','modal_precompiled_compare.py','modal_gpu_compare.py','modal_cuda_tests.py','install_gpu_baselines.sh')
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT/'src',ROOT.parent/'brian2') for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for p in [*(ROOT/'examples'/n for n in names),ROOT/'Cargo.toml',ROOT/'Cargo.lock',
              *(ROOT.parent/n for n in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        hashes[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output/'source-hashes.json').write_text(json.dumps(hashes,sort_keys=True,indent=2)+'\n')
    image=(native_image().apt_install('libffi-dev','pkg-config').pip_install('uv==0.8.22')
        .add_local_file(ROOT/'examples/install_gpu_baselines.sh','/opt/install_gpu_baselines.sh',copy=True)
        .run_commands('sh /opt/install_gpu_baselines.sh')
        .add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=['**/__pycache__/**','**/*.pyc']))
    for n in names:
        if n!='install_gpu_baselines.sh':image=image.add_local_file(ROOT/'examples'/n,'/workspace/brian2-rust/examples/'+n)
    image=image.add_local_file(Path(__file__),'/root/modal_precompiled_compare.py')
    app=modal.App('brian2-precompiled-comparison',include_source=False)
    run=app.function(image=image,gpu=args.gpu,cpu=4,memory=8192,max_containers=1,timeout=1800,retries=0)(compare)
    start=time.perf_counter()
    with modal.enable_output(),app.run():report=run.remote(args.neurons,args.steps,args.degree,args.repeats)
    report.update(requested_gpu=args.gpu,remote_wall_seconds=time.perf_counter()-start,
        source_manifest_sha256=hashlib.sha256((args.output/'source-hashes.json').read_bytes()).hexdigest())
    for name,payload in report.pop('payloads').items():(args.output/name).write_bytes(payload)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],report['summary'])
    if report['status']!='passed-matched-numeric-gates':raise SystemExit(1)


if __name__=='__main__':main()
