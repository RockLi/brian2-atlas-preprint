"""One bounded GPU job: stock Brian2GeNN plus explicit STDP correction gates."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES=('modal_brian2genn_stdp_adapter.py','gpu_brian2genn_stdp_adapter.py','gpu_stdp_compare.py','gpu_baseline.py','modal_cuda_tests.py','install_gpu_baselines.sh')


def compare():
    import os,signal,subprocess,sys,tempfile,io
    import numpy as np
    sys.path.insert(0,'/workspace/brian2-rust/examples')
    from gpu_stdp_compare import checks
    cases=[(17,7,48),(256,128,128),(17,1,1),(17,7,4)]
    report=dict(schema='b2-modal-genn-stdp-adapter-v1',trials=[],
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),
        locks={p.name:p.read_text() for p in Path('/opt/baseline-locks').glob('*.txt')})
    deadline=time.monotonic()+1100
    with tempfile.TemporaryDirectory(prefix='genn-stdp-correction-') as temporary:
        for n,k,t in cases:
            control=None
            for backend in ['cpu-f32','brian2genn-stock','brian2genn-corrected']:
                # One retained stock comparison suffices; other cases test boundaries.
                if backend=='brian2genn-stock' and (n,k,t)!=(256,128,128):continue
                remaining=deadline-time.monotonic()
                if remaining<30:raise TimeoutError('Container diagnostic time budget exhausted')
                label=f'n{n}-k{k}-t{t}-'+backend;out=Path(temporary)/label
                env={**os.environ,'CUDA_PATH':'/usr/local/cuda','NVCC_APPEND_FLAGS':'--fmad=false --ftz=false --prec-div=true --prec-sqrt=true'}
                env.pop('NVCC_PREPEND_FLAGS',None)
                if backend!='cpu-f32':env['PYTHONPATH']='';env['PATH']='/opt/genn4/bin:'+env['PATH']
                script='gpu_brian2genn_stdp_adapter.py' if backend.endswith('corrected') else 'gpu_stdp_compare.py'
                command=[sys.executable if backend=='cpu-f32' else '/opt/brian2genn-env/bin/python','/workspace/brian2-rust/examples/'+script,
                    '--neurons',str(n),'--degree',str(k),'--steps',str(t),'--output',str(out)]
                if backend!='brian2genn-corrected':command+=['--backend','cpu-f32' if backend=='cpu-f32' else 'brian2genn','--split-delays']
                start=time.perf_counter();proc=subprocess.Popen(command,env=env,cwd=temporary,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
                try:stdout,stderr=proc.communicate(timeout=min(210,remaining-15));code=proc.returncode
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate();code=None
                row=dict(label=label,backend=backend,case=[n,k,t],exit_code=code,stdout=stdout,stderr=stderr,wall_seconds=time.perf_counter()-start,
                    report=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else None)
                if (out/'result.npz').exists():
                    row['result_npz']=(out/'result.npz').read_bytes()
                    with np.load(io.BytesIO(row['result_npz'])) as z:actual=dict(z)
                    if backend=='cpu-f32':control=actual
                    if control is not None:row['against_cpu_f32']=checks(actual,control)
                row['sources']={str(p.relative_to(out)):p.read_text() for p in out.rglob('*') if p.is_file() and
                    (p.suffix in {'.cpp','.cc','.cu','.h','.original','.patch'} or p.name=='transformations.json') and p.stat().st_size<200000}
                report['trials'].append(row);print(label,code,row['report']['status'] if row['report'] else 'no report', (row['report'] or {}).get('error',''),flush=True)
    report['status']='passed' if all(r['exit_code']==0 and r.get('against_cpu_f32',{}).get('passed') for r in report['trials'] if r['backend']!='brian2genn-stock') else 'diagnostic_failed'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    ignore=['**/__pycache__/**','**/*.pyc','**/*.so','**/*.dylib']
    sources={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT.parent/'brian2',ROOT/'src') for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for p in [*(ROOT/'examples'/n for n in NAMES),ROOT/'Cargo.toml',ROOT/'Cargo.lock',*(ROOT.parent/n for n in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        sources[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (a.output/'source-hashes.json').write_text(json.dumps(sources,sort_keys=True,indent=2)+'\n')
    image=(native_image().apt_install('libffi-dev','pkg-config').pip_install('uv==0.8.22')
        .add_local_file(ROOT/'examples/install_gpu_baselines.sh','/opt/install_gpu_baselines.sh',copy=True)
        .run_commands('sh /opt/install_gpu_baselines.sh').add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=ignore))
    for n in NAMES:
        if n!='install_gpu_baselines.sh':image=image.add_local_file(ROOT/'examples'/n,'/workspace/brian2-rust/examples/'+n)
    image=image.add_local_file(Path(__file__),'/root/modal_brian2genn_stdp_adapter.py')
    app=modal.App('brian2genn-stdp-correction',include_source=False)
    run=app.function(image=image,gpu=a.gpu,cpu=4,memory=8192,max_containers=1,timeout=1200,retries=0)(compare)
    started=time.perf_counter()
    with modal.enable_output(),app.run():report=run.remote()
    report.update(requested_gpu=a.gpu,remote_wall_seconds=time.perf_counter()-started)
    for row in report['trials']:
        payload=row.pop('result_npz',None)
        if payload is not None:
            name=row['label']+'.npz';(a.output/name).write_bytes(payload);row['result_artifact']=dict(path=name,sha256=hashlib.sha256(payload).hexdigest())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'])
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
