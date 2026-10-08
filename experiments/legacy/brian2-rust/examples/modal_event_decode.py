"""Bounded CUDA conformance and paired host event-decoder ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def verify():
    import os,sys
    os.chdir('/workspace');sys.path.insert(0,'/workspace/brian2-rust/examples')
    from modal_cuda_tests import run_tests
    from gpu_decode_benchmark import benchmark
    report=run_tests(['brian2-rust/tests/test_gpu_event_decode.py','brian2-rust/tests/test_gpu_monitors.py',
        'brian2-rust/tests/test_gpu_custom_events.py','brian2-rust/tests/test_gpu_ticks.py',
        'brian2-rust/tests/test_cuda_graphs.py','brian2-rust/tests/test_cuda_chunks.py'])
    report['ablations']={};report['payloads']={}
    if report['passed']:
        for n,steps in ((512,512),(4096,2048)):
            label=f'n{n}-t{steps}';output=Path('/tmp')/('event-decode-'+label)
            r=benchmark('cuda',n,steps,32,7,output)
            report['ablations'][label]=r
            for name in ('model.json','legacy.npz','boolean.npz'):
                report['payloads'][label+'/'+name]=(output/name).read_bytes()
            print(label,r['summary'],flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    ignore=['**/__pycache__/**','**/*.pyc','**/*.so','**/*.dylib']
    names=('modal_event_decode.py','gpu_decode_benchmark.py','cuda_dag_benchmark.py','modal_cuda_tests.py','gpu_baseline.py','gpu_recurrent.py')
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT/'tests',ROOT/'src',ROOT.parent/'brian2') for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for path in [*(ROOT/'examples'/n for n in names),ROOT/'Cargo.toml',ROOT/'Cargo.lock',
        *(ROOT.parent/n for n in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        hashes[str(path.relative_to(ROOT.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=json.dumps(hashes,sort_keys=True,indent=2)+'\n';(a.output/'source-hashes.json').write_text(manifest)
    image=native_image().add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=ignore).add_local_dir(ROOT/'tests','/workspace/brian2-rust/tests',ignore=ignore)
    for n in names:image=image.add_local_file(ROOT/'examples'/n,'/workspace/brian2-rust/examples/'+n)
    image=image.add_local_file(Path(__file__),'/root/modal_event_decode.py')
    app=modal.App('brian2-event-decode',include_source=False)
    run=app.function(image=image,gpu=a.gpu,cpu=4,memory=8192,max_containers=1,timeout=900,retries=0)(verify)
    started=time.perf_counter()
    with modal.enable_output(),app.run():report=run.remote()
    for name,data in report.pop('payloads').items():
        path=a.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    report.update(requested_gpu=a.gpu,remote_wall_seconds=time.perf_counter()-started,
        source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['stdout'])
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
