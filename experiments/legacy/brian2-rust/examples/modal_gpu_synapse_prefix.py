"""Bounded opt-in edge-prefix conformance and degree-32/128 ablations."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def verify(benchmark_only=False):
    import os,sys
    os.chdir('/workspace');sys.path.insert(0,'/workspace/brian2-rust/examples')
    from modal_cuda_tests import run_tests
    if benchmark_only:
        import subprocess,platform
        report=dict(passed=True,stdout='',stderr='',selected_tests=[],tests_run=False,
            host=platform.platform(),python=sys.version,
            nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
            packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
            rustc=subprocess.check_output(['rustc','-vV'],text=True))
    else:
        report=run_tests(['brian2-rust/tests/test_gpu_synapse_prefix.py',
            'brian2-rust/tests/test_gpu_target_pathway.py'])
        report['tests_run']=True
    report['payloads']={}
    for pattern in ('prefix-results.npz','prefix-device.npz','model.json','prefix-plan.json','previous-plan.json'):
        for p in Path('/tmp/pytest-of-root/pytest-0').rglob(pattern):
            report['payloads']['tests/'+str(p.relative_to('/tmp/pytest-of-root/pytest-0'))]=p.read_bytes()
    if report['passed']:
        import subprocess
        report['benchmarks']={}
        for degree in (32,128):
            case='stdp-degree-'+str(degree)
            directory=Path('/tmp/prefix-benchmark')/case
            completed=subprocess.run([sys.executable,'/workspace/brian2-rust/examples/gpu_synapse_prefix_benchmark.py',
                '--backend','cuda','--case','stdp-random','--degree',str(degree),'--neurons',str(4096 if degree==32 else 2048),'--repeats','7','--output',str(directory)],capture_output=True,text=True)
            report['benchmarks'][case]=dict(exit_code=completed.returncode,stdout=completed.stdout,stderr=completed.stderr)
            for p in directory.rglob('*'):
                if p.is_file() and (p.suffix=='.npz' or p.name.endswith('plan.json') or p.name in {'model.json','report.json'}):
                    report['payloads']['benchmarks/'+case+'/'+str(p.relative_to(directory))]=p.read_bytes()
            report['passed']=report['passed'] and completed.returncode==0

    return report


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def save_result(report,output,metadata):
    # Save the small verdict first. A failed large payload write must never
    # erase the test counts, hardware inventory or the retained call identity.
    result={k:v for k,v in report.items() if k!='payloads'}
    result.update(metadata)
    result['artifact_status']='pending'
    result['result_artifacts']={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in report.get('payloads',{}).items()}
    write_json(output/'report.json',result)
    for name,data in report.get('payloads',{}).items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid result path')
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.partial')
        temporary.write_bytes(data);temporary.replace(path)
    result['artifact_status']='complete'
    write_json(output/'report.json',result)
    print(result['stdout'])
    if not result['passed']:raise SystemExit(1)


def resume_result(output):
    import modal
    checkpoint=json.loads((output/'call.json').read_text())
    manifest=(output/'source-hashes.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest()!=checkpoint['source_manifest_sha256']:
        raise ValueError('Saved call manifest changed')
    # Reading the retained call is the only cloud operation in this path.
    # Never invoke verify, create an App or retry an expired/missing call.
    report=modal.FunctionCall.from_id(checkpoint['function_call_id']).get(timeout=0)
    save_result(report,output,dict(checkpoint,remote_wall_seconds=None,recovered_existing_call=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4');p.add_argument('--output',type=Path,required=True);p.add_argument('--benchmark-only',action='store_true',help='Run bounded paired workloads without repeating conformance tests');p.add_argument('--resume',action='store_true',help='Fetch the saved call again without launching GPU work');a=p.parse_args()
    if a.resume:return resume_result(a.output)
    a.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    names=('cuda_dag_benchmark.py','gpu_recurrent.py','modal_gpu_synapse_prefix.py','modal_cuda_tests.py','gpu_baseline.py','gpu_stdp_compare.py','gpu_stdp_precompiled.py','gpu_synapse_prefix_benchmark.py','gpu_decode_benchmark.py')
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT/'tests',ROOT/'src',ROOT.parent/'brian2') for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for path in [*(ROOT/'examples'/n for n in names),ROOT/'Cargo.toml',ROOT/'Cargo.lock',
        *(ROOT.parent/n for n in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        hashes[str(path.relative_to(ROOT.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=json.dumps(hashes,sort_keys=True,indent=2)+'\n';(a.output/'source-hashes.json').write_text(manifest)
    ignore=['**/__pycache__/**','**/*.pyc','**/*.so','**/*.dylib']
    image=native_image().add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=ignore).add_local_dir(ROOT/'tests','/workspace/brian2-rust/tests',ignore=ignore)
    for name in names:image=image.add_local_file(ROOT/'examples'/name,'/workspace/brian2-rust/examples/'+name)
    image=image.add_local_file(Path(__file__),'/root/modal_gpu_synapse_prefix.py')
    app=modal.App('brian2-gpu-synapse-prefix',include_source=False)
    run=app.function(image=image,gpu=a.gpu,cpu=4,memory=8192,max_containers=1,timeout=1200,retries=0)(verify)
    started=time.perf_counter()
    with modal.enable_output(),app.run():
        call=run.spawn(a.benchmark_only)
        checkpoint=dict(requested_gpu=a.gpu,function_call_id=call.object_id,
            source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())
        write_json(a.output/'call.json',checkpoint)
        report=call.get()
    save_result(report,a.output,dict(checkpoint,remote_wall_seconds=time.perf_counter()-started,
        recovered_existing_call=False))



if __name__=='__main__':main()
