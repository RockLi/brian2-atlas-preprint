"""Bounded same-executor native GPU spike-readbackr comparison."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TEST_NAMES=('test_cuda.py', 'test_cuda_graphs.py', 'test_gpu_composed_models.py', 'test_gpu_custom_events.py', 'test_gpu_expression_contract.py', 'test_gpu_monitors.py', 'test_gpu_multiclock.py', 'test_gpu_refractory.py', 'test_gpu_spike_generator.py', 'test_gpu_spike_readback.py', 'test_gpu_summed_parallel.py', 'test_gpu_typed_storage.py', 'test_gpu_workgroup.py', 'test_metal_dag.py', 'test_metal_delays.py', 'test_metal_plasticity.py', 'test_population.py')


def verify():
    import os,subprocess,platform,sys
    directory=Path('/tmp/spike-readback')
    env={**os.environ,'PYTHONPATH':'/workspace/brian2-rust/examples:/workspace/brian2-rust/python:/workspace','CUDA_PATH':'/usr/local/cuda'}
    env['B2_TEST_CUDA']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    tested=subprocess.run([sys.executable,'-m','pytest','-q','/workspace/brian2-rust/tests/test_gpu_spike_readback.py','--basetemp=/tmp/spike-readback-tests'],capture_output=True,text=True,env=env,timeout=300)
    if tested.returncode:
        return dict(passed=False,exit_code=tested.returncode,stdout=tested.stdout,stderr=tested.stderr,payloads={'tests/cuda.log':(tested.stdout+tested.stderr).encode()},nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True))
    done=subprocess.run([sys.executable,'/workspace/brian2-rust/examples/gpu_spike_readback_compare.py',
        '--backend','cuda','--output',str(directory)],capture_output=True,text=True,env=env,timeout=1100)
    report=dict(passed=done.returncode==0,exit_code=done.returncode,stdout=done.stdout,stderr=done.stderr,payloads={},
        host=platform.platform(),nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True))
    report['payloads']['tests/cuda.log']=(tested.stdout+tested.stderr).encode()
    names=['report.json','declared-protocol.json']
    if (directory/'report.json').exists():names+=json.loads((directory/'report.json').read_text())['artifact_files']
    else:names+=[str(p.relative_to(directory)) for p in directory.glob('*.npz')]
    for name in sorted(set(names)):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid retained result path')
        if (directory/relative).is_file():report['payloads']['trial/'+name]=(directory/relative).read_bytes()
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4');p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true',help='Fetch the saved call again without launching GPU work');a=p.parse_args()
    if a.resume:return resume_result(a.output)
    a.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    names=('gpu_brian2genn_stdp_adapter.py','gpu_baseline.py','gpu_stdp_compare.py','gpu_stdp_precompiled.py','modal_stdp_precompiled.py','modal_stdp_prefix_compare.py','gpu_genn_barrier_adapter.py','gpu_genn_readback.py','gpu_spike_readback_compare.py','genn_postsynaptic_barrier.py','modal_spike_readback_compare.py','modal_cuda_tests.py','install_gpu_baselines.sh')
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT/'src',ROOT.parent/'brian2') for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for path in [*(ROOT/'examples'/n for n in names),ROOT/'Cargo.toml',ROOT/'Cargo.lock',
        *(ROOT.parent/n for n in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        hashes[str(path.relative_to(ROOT.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for n in TEST_NAMES:hashes['brian2-rust/tests/'+n]=hashlib.sha256((ROOT/'tests'/n).read_bytes()).hexdigest()
    manifest=json.dumps(hashes,sort_keys=True,indent=2)+'\n';(a.output/'source-hashes.json').write_text(manifest)
    ignore=['**/__pycache__/**','**/*.pyc','**/*.so','**/*.dylib']
    image=(native_image().apt_install('libffi-dev','pkg-config').pip_install('uv==0.8.22')
        .add_local_file(ROOT/'examples/install_gpu_baselines.sh','/opt/install_gpu_baselines.sh',copy=True)
        .run_commands('sh /opt/install_gpu_baselines.sh')
        .add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=ignore))
    for name in names:
        if name!='install_gpu_baselines.sh':image=image.add_local_file(ROOT/'examples'/name,'/workspace/brian2-rust/examples/'+name)
    for n in TEST_NAMES:image=image.add_local_file(ROOT/'tests'/n,'/workspace/brian2-rust/tests/'+n)
    image=image.add_local_file(Path(__file__),'/root/modal_spike_readback_compare.py')
    app=modal.App('brian2-spike-readback-comparison',include_source=False)
    run=app.function(image=image,gpu=a.gpu,cpu=4,memory=8192,max_containers=1,timeout=1200,retries=0)(verify)
    started=time.perf_counter()
    with modal.enable_output(),app.run():
        call=run.spawn()
        checkpoint=dict(requested_gpu=a.gpu,function_call_id=call.object_id,
            source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())
        write_json(a.output/'call.json',checkpoint)
        report=call.get()
    save_result(report,a.output,dict(checkpoint,remote_wall_seconds=time.perf_counter()-started,
        recovered_existing_call=False))



if __name__=='__main__':main()
