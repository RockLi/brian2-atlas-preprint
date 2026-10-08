"""Bounded same-GPU delayed-STDP adapter diagnostics, retaining every failure."""
import argparse,hashlib,io,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES=('modal_stdp_compare.py','gpu_stdp_compare.py','gpu_baseline.py','modal_cuda_tests.py','install_gpu_baselines.sh')


def compare(neurons,degree,steps):
    import os,signal,subprocess,sys,tempfile
    import numpy as np
    sys.path.insert(0,'/workspace/brian2-rust/examples')
    from gpu_stdp_compare import checks
    report=dict(schema='b2-modal-stdp-diagnostic-v0',configuration=dict(neurons=neurons,degree=degree,steps=steps),
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),
        locks={p.name:p.read_text() for p in Path('/opt/baseline-locks').glob('*.txt')},
        scope='conformance diagnosis, fresh project wall; no warm throughput ranking',trials=[])
    interpreters={k:sys.executable for k in ('rust','cpu-f32','cuda')}
    interpreters.update(brian2cuda='/opt/brian2cuda-env/bin/python',brian2genn='/opt/brian2genn-env/bin/python',genn='/opt/genn5-env/bin/python')
    deadline=time.monotonic()+1100
    with tempfile.TemporaryDirectory(prefix='stdp-adapters-') as temporary:
        for backend,split in [('rust',False),('cpu-f32',False),('cuda',False),('brian2cuda',False),('brian2genn',False),('brian2genn',True),('genn',True)]:
            label=backend+('-split' if split and backend!='genn' else '')
            remaining=deadline-time.monotonic()
            if remaining<30:
                report['trials'].append(dict(label=label,backend=backend,error='container time budget exhausted'));continue
            out=Path(temporary)/label;env={**os.environ,'CUDA_PATH':'/usr/local/cuda','NVCC_APPEND_FLAGS':'--fmad=false --ftz=false --prec-div=true --prec-sqrt=true'}
            env.pop('NVCC_PREPEND_FLAGS',None)
            if backend not in {'rust','cuda','cpu-f32'}:env['PYTHONPATH']=''
            if backend=='brian2genn':env['PATH']='/opt/genn4/bin:'+env['PATH']
            command=[interpreters[backend],'/workspace/brian2-rust/examples/gpu_stdp_compare.py','--backend',backend,
                '--neurons',str(neurons),'--degree',str(degree),'--steps',str(steps),'--output',str(out)]
            if split:command.append('--split-delays')
            start=time.perf_counter();proc=subprocess.Popen(command,env=env,cwd=temporary,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
            try:stdout,stderr=proc.communicate(timeout=min(210,remaining-15));code=proc.returncode
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate();code=None
            row=dict(label=label,backend=backend,split=split,exit_code=code,stdout=stdout,stderr=stderr,
                process_wall_seconds=time.perf_counter()-start,
                report=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else None)
            if (out/'result.npz').exists():row['result_npz']=(out/'result.npz').read_bytes()
            row['generated_sources']={str(p.relative_to(out)):p.read_text() for p in out.rglob('*') if p.is_file()
                and p.suffix in {'.cc','.cu','.cpp','.h'} and p.stat().st_size<200000}
            report['trials'].append(row);print(label,code,row['report']['status'] if row['report'] else 'no report',flush=True)
    report['pairwise_against_cpu_f32']=[]
    control=next((r for r in report['trials'] if r['label']=='cpu-f32' and 'result_npz' in r),None)
    if control:
        with np.load(io.BytesIO(control['result_npz'])) as a:expected=dict(a)
        for row in report['trials']:
            if 'result_npz' not in row:continue
            with np.load(io.BytesIO(row['result_npz'])) as a:actual=dict(a)
            report['pairwise_against_cpu_f32'].append(dict(label=row['label'],**checks(actual,expected)))
    reports=[r['report'] for r in report['trials'] if r.get('report')]
    report['same_model_configuration']=len(reports)==7 and len({r['model_sha256'] for r in reports})==1
    report['status']='diagnostic_complete' if len(reports)==7 else 'incomplete_diagnostic'
    report['all_adapters_passed']=len(reports)==7 and all(r['status']=='passed' for r in reports)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--neurons',type=int,default=256);p.add_argument('--degree',type=int,default=128);p.add_argument('--steps',type=int,default=128);a=p.parse_args()
    from gpu_stdp_compare import topology
    topology(a.neurons,a.degree)
    if not 1<=a.steps<=512:p.error('steps must be 1..512')
    a.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    ignore=['**/__pycache__/**','**/*.pyc','**/*.so','**/*.dylib']
    sources={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT.parent/'brian2',ROOT/'src') for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for p in [*(ROOT/'examples'/name for name in NAMES),ROOT/'Cargo.toml',ROOT/'Cargo.lock',
              *(ROOT.parent/name for name in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        sources[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (a.output/'source-hashes.json').write_text(json.dumps(sources,sort_keys=True,indent=2)+'\n')
    image=(native_image().apt_install('libffi-dev','pkg-config').pip_install('uv==0.8.22')
        .add_local_file(ROOT/'examples/install_gpu_baselines.sh','/opt/install_gpu_baselines.sh',copy=True)
        .run_commands('sh /opt/install_gpu_baselines.sh').add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=ignore))
    for name in NAMES:
        if name!='install_gpu_baselines.sh':image=image.add_local_file(ROOT/'examples'/name,'/workspace/brian2-rust/examples/'+name)
    image=image.add_local_file(Path(__file__),'/root/modal_stdp_compare.py')
    app=modal.App('brian2-stdp-comparison',include_source=False)
    run=app.function(image=image,gpu=a.gpu,cpu=4,memory=8192,max_containers=1,timeout=1200,retries=0)(compare)
    started=time.perf_counter()
    with modal.enable_output(),app.run():report=run.remote(a.neurons,a.degree,a.steps)
    report.update(requested_gpu=a.gpu,remote_wall_seconds=time.perf_counter()-started)
    for row in report['trials']:
        payload=row.pop('result_npz',None)
        if payload is not None:
            name=row['label']+'.npz';(a.output/name).write_bytes(payload)
            row['result_artifact']=dict(path=name,sha256=hashlib.sha256(payload).hexdigest())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],[(r['label'],r.get('exit_code')) for r in report['trials']])
    if report['status']!='diagnostic_complete':raise SystemExit(1)


if __name__=='__main__':main()
