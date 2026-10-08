"""Bounded CUDA cross-activation buffer ownership and Graph validation."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def verify():
    import os,sys
    os.chdir('/workspace');sys.path.insert(0,'/workspace/brian2-rust/examples')
    from modal_cuda_tests import run_tests
    report=run_tests(['brian2-rust/tests/test_gpu_buffer_transfer.py',
        'brian2-rust/tests/test_cuda_graphs.py','brian2-rust/tests/test_cuda_chunks.py',
        'brian2-rust/tests/test_gpu_multiclock.py','brian2-rust/tests/test_gpu_pathway_order.py',
        'brian2-rust/tests/test_cuda.py'])
    report['payloads']={}
    for pattern in ('transfer-results.npz','transfer-runtime.json','activation-results.npz','activation-reports.json'):
        for p in Path('/tmp/pytest-of-root/pytest-0').rglob(pattern):
            report['payloads'][str(p.relative_to('/tmp/pytest-of-root/pytest-0'))]=p.read_bytes()
    if report['passed']:
        from gpu_activation_benchmark import benchmark
        output=Path('/tmp/activation-benchmark')
        try:report['activation_benchmark']=benchmark('cuda',output)
        except Exception as error:
            report.update(passed=False,benchmark_error_type=type(error).__name__,benchmark_error=str(error))
        for p in output.rglob('*'):
            if p.is_file() and p.suffix in {'.json','.npz'}:
                report['payloads']['benchmark/'+str(p.relative_to(output))]=p.read_bytes()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    names=('modal_gpu_buffer_transfer.py','modal_cuda_tests.py','gpu_baseline.py','gpu_activation_benchmark.py')
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
    image=image.add_local_file(Path(__file__),'/root/modal_gpu_buffer_transfer.py')
    app=modal.App('brian2-gpu-buffer-transfer',include_source=False)
    run=app.function(image=image,gpu=a.gpu,cpu=4,memory=8192,max_containers=1,timeout=1200,retries=0)(verify)
    started=time.perf_counter()
    with modal.enable_output(),app.run():report=run.remote()
    for name,data in report.pop('payloads').items():
        path=a.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    report.update(requested_gpu=a.gpu,remote_wall_seconds=time.perf_counter()-started,
        source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['stdout'])
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
