"""Bounded native CUDA unbound projection storage conformance."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TEST_NAMES=tuple(sorted(p.name for p in (ROOT/'tests').glob('*.py')))
SUITES={'api':('test_gpu_bitset_api.py','test_gpu_compact_bitset.py','test_gpu_projection_storage.py','test_execution_plan.py'),
        'plan':('test_gpu_bitset_api.py::test_public_plan_matches_experiment_and_rejects_wrong_selection',)}


def verify(suite='api'):
    import os,subprocess,platform,sys
    started=time.perf_counter()
    directory=Path('/tmp/projection-storage-tests')
    selected=SUITES[suite]
    env={**os.environ,'PYTHONPATH':'/workspace/brian2-rust/examples:/workspace/brian2-rust/python:/workspace',
         'CUDA_PATH':'/usr/local/cuda','B2_TEST_CUDA':'1','PYTHONDONTWRITEBYTECODE':'1'}
    command=[sys.executable,'-m','pytest','-v',*('/workspace/brian2-rust/tests/'+n for n in selected),
             '--basetemp='+str(directory),'--junitxml=/tmp/projection-storage-junit.xml']
    try:
        done=subprocess.run(command,capture_output=True,text=True,env=env,timeout=650)
    except subprocess.TimeoutExpired as exc:
        decoded=lambda value:value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
        done=subprocess.CompletedProcess(command,124,decoded(exc.output),decoded(exc.stderr)+'\nTest deadline exceeded: 650 seconds\n')
    report=dict(passed=done.returncode==0,exit_code=done.returncode,stdout=done.stdout,stderr=done.stderr,
        payloads={'tests/cuda.log':(done.stdout+done.stderr).encode()},host=platform.platform(),
        nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),suite=suite,selected_tests=list(selected))
    junit=Path('/tmp/projection-storage-junit.xml')
    if junit.is_file():report['payloads']['tests/junit.xml']=junit.read_bytes()
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.name.startswith('._'):continue
        if path.suffix in {'.npz','.json','.cu','.cubin','.ptx','.sass'}:
            report['payloads']['tests/'+str(path.relative_to(directory))]=path.read_bytes()
    if report['passed'] and suite=='api':
        output=Path('/tmp/projection-storage-benchmark')
        command=[sys.executable,'/workspace/brian2-rust/examples/gpu_projection_storage_compare.py',
                 '--backend','cuda','--output',str(output)]
        try:
            benchmark=subprocess.run(command,capture_output=True,text=True,env=env,timeout=450)
        except subprocess.TimeoutExpired as exc:
            decoded=lambda value:value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
            benchmark=subprocess.CompletedProcess(command,124,decoded(exc.output),decoded(exc.stderr)+'\nBenchmark deadline exceeded\n')
        report['benchmark_exit_code']=benchmark.returncode
        report['passed']=benchmark.returncode==0
        report['payloads']['benchmark.log']=(benchmark.stdout+benchmark.stderr).encode()
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.name.startswith('._') and path.suffix in {'.json','.npz','.cu','.cubin','.ptx','.sass'}:
                report['payloads']['benchmark/'+str(path.relative_to(output))]=path.read_bytes()
    return report


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def save_result(report,output,metadata):
    # Save the small verdict first. A failed large payload write must never
    # erase the test counts, hardware inventory or the retained call identity.
    for name in report.get('payloads',{}):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or str(relative) in {'report.json','call.json','source-hashes.json'}:
            raise ValueError('Result payload collides with provenance or escapes output')
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4');p.add_argument('--output',type=Path,required=True);p.add_argument('--suite',choices=tuple(SUITES),default='api');p.add_argument('--resume',action='store_true',help='Fetch the saved call again without launching GPU work');a=p.parse_args()
    if a.resume:return resume_result(a.output)
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'declared-tests.json',dict(suite=a.suite,selected_tests=list(SUITES[a.suite]),timeout_seconds=1200,test_timeout_seconds=650,retries=0))
    import modal
    from modal_cuda_tests import native_image
    names=('modal_projection_storage_tests.py','modal_cuda_tests.py','cuda_cooperative_compare.py','gpu_sparse_saturation_compare.py','gpu_target_bitset_compare.py','gpu_projection_storage_compare.py','cuda_atomic_load_compare.py','gpu_prefix_locals_compare.py','cuda_dag_benchmark.py','gpu_recurrent.py','gpu_stdp_compare.py','gpu_stdp_precompiled.py','gpu_baseline.py')
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT/'src',ROOT.parent/'brian2') for p in sorted(folder.rglob('*'))
        if p.is_file() and not p.name.startswith('._') and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for path in [*(ROOT/'examples'/n for n in names),ROOT/'Cargo.toml',ROOT/'Cargo.lock',
        *(ROOT.parent/n for n in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        hashes[str(path.relative_to(ROOT.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for n in TEST_NAMES:hashes['brian2-rust/tests/'+n]=hashlib.sha256((ROOT/'tests'/n).read_bytes()).hexdigest()
    golden=ROOT/'tests/golden/b2ir-v1/minimal-v1.json'
    hashes[str(golden.relative_to(ROOT.parent))]=hashlib.sha256(golden.read_bytes()).hexdigest()
    manifest=json.dumps(hashes,sort_keys=True,indent=2)+'\n';(a.output/'source-hashes.json').write_text(manifest)
    ignore=['**/__pycache__/**','**/*.pyc','**/*.so','**/*.dylib','**/._*']
    image=native_image().add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=ignore)
    for name in names:
        if name!='install_gpu_baselines.sh':image=image.add_local_file(ROOT/'examples'/name,'/workspace/brian2-rust/examples/'+name)
    for n in TEST_NAMES:image=image.add_local_file(ROOT/'tests'/n,'/workspace/brian2-rust/tests/'+n)
    image=image.add_local_file(golden,'/workspace/brian2-rust/tests/golden/b2ir-v1/minimal-v1.json')
    image=image.add_local_file(Path(__file__),'/root/modal_projection_storage_tests.py')
    app=modal.App('brian2-projection-storage-conformance',include_source=False)
    run=app.function(image=image,gpu=a.gpu,cpu=4,memory=8192,max_containers=1,timeout=1200,retries=0)(verify)
    started=time.perf_counter()
    with modal.enable_output(),app.run():
        call=run.spawn(suite=a.suite)
        checkpoint=dict(requested_gpu=a.gpu,function_call_id=call.object_id,
            source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())
        write_json(a.output/'call.json',checkpoint)
        report=call.get()
    save_result(report,a.output,dict(checkpoint,remote_wall_seconds=time.perf_counter()-started,
        recovered_existing_call=False))



if __name__=='__main__':main()
