"""Two bounded, fully gated six-backend STDP cases on one GPU allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
CASES={
    'random-sparse':dict(neurons=1024,degree=8,steps=256,workload=dict(drive=.0625,delay_span=8,post_delay=3,topology_kind='random-fixed-outdegree',topology_seed=42)),
    'random-larger':dict(neurons=4096,degree=32,steps=256,workload=dict(drive=.0625,delay_span=8,post_delay=3,topology_kind='random-fixed-outdegree',topology_seed=42)),
    'lower-drive':dict(neurons=1024,degree=8,steps=256,workload=dict(drive=.0625,delay_span=8,post_delay=3)),
    'long-delays':dict(neurons=1024,degree=8,steps=256,workload=dict(drive=.0625,delay_span=16,post_delay=16)),
}


def compare_matrix(case_names, repeats=5, requested_backends=None):
    import sys
    sys.path.insert(0,'/workspace/brian2-rust/examples')
    from modal_stdp_precompiled import compare
    from gpu_stdp_precompiled import BACKENDS
    names=list(case_names)
    if not names or len(names)>2 or len(set(names))!=len(names) or any(n not in CASES for n in names):
        raise ValueError('Select one or two distinct declared STDP cases')
    if type(repeats) is not int or not 1<=repeats<=5:raise ValueError('Require 1..5 measured rounds')
    backends=list(BACKENDS if requested_backends is None else requested_backends)
    if ('cpu-f32' not in backends or len(set(backends))!=len(backends)
            or any(b not in (*BACKENDS,'metal') for b in backends)):
        raise ValueError('Unique supported backends including cpu-f32 are required')
    report=dict(schema='b2-stdp-matrix-v0',case_order=names,repeats=repeats,backends=backends,
        cases={},payloads={},status='running',scope='Sequential complete comparisons on one allocation; no cross-case timing aggregation')
    gpu=None
    for name in names:
        config=CASES[name]
        print('case',name,'starting',flush=True)
        row=compare(config['neurons'],config['steps'],config['degree'],repeats,
            requested_backends=backends,workload=dict(config['workload']),continue_on_gate_failure=True)
        observed=row['nvidia_smi']
        if name==names[0]:gpu=observed
        if observed!=gpu:raise RuntimeError('Physical GPU identity changed between cases')
        if row['status'] not in {'failed','completed-with-gate-failures','passed-matched-numeric-gates'}:
            raise RuntimeError('Unknown terminal comparison status')
        for key,value in row.pop('payloads').items():
            if Path(key).name!=key or key in {'.','..'}:raise ValueError('Unexpected artifact filename')
            report['payloads'][name+'/'+key]=value
        report['cases'][name]=row
        print('case',name,row['status'],flush=True)
    statuses={row['status'] for row in report['cases'].values()}
    report['status']=('failed' if 'failed' in statuses else 'completed-with-gate-failures'
        if 'completed-with-gate-failures' in statuses else 'passed-matched-numeric-gates')
    report['nvidia_smi']=gpu
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4')
    p.add_argument('--case',choices=tuple(CASES),action='append',dest='cases')
    p.add_argument('--repeats',type=int,choices=range(1,6),default=5)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();names=a.cases or ['lower-drive','long-delays']
    if len(names)>2:p.error('Select one or two declared STDP cases')
    if len(set(names))!=len(names):p.error('Duplicate case selection')
    a.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    scripts=('gpu_brian2genn_stdp_adapter.py','gpu_baseline.py','gpu_stdp_compare.py',
        'gpu_stdp_precompiled.py','modal_stdp_precompiled.py','modal_stdp_matrix.py','modal_cuda_tests.py','install_gpu_baselines.sh')
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT/'src',ROOT.parent/'brian2') for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for path in [*(ROOT/'examples'/n for n in scripts),ROOT/'Cargo.toml',ROOT/'Cargo.lock',
        *(ROOT.parent/n for n in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        hashes[str(path.relative_to(ROOT.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=json.dumps(hashes,sort_keys=True,indent=2)+'\n';(a.output/'source-hashes.json').write_text(manifest)
    image=(native_image().apt_install('libffi-dev','pkg-config').pip_install('uv==0.8.22')
        .add_local_file(ROOT/'examples/install_gpu_baselines.sh','/opt/install_gpu_baselines.sh',copy=True)
        .run_commands('sh /opt/install_gpu_baselines.sh')
        .add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=['**/__pycache__/**','**/*.pyc','**/*.so','**/*.dylib']))
    for n in scripts:
        if n!='install_gpu_baselines.sh':image=image.add_local_file(ROOT/'examples'/n,'/workspace/brian2-rust/examples/'+n)
    image=image.add_local_file(Path(__file__),'/root/modal_stdp_matrix.py')
    app=modal.App('brian2-stdp-matrix',include_source=False)
    run=app.function(image=image,gpu=a.gpu,cpu=4,memory=8192,max_containers=1,timeout=1200,retries=0)(compare_matrix)
    started=time.perf_counter()
    with modal.enable_output(),app.run():report=run.remote(names,a.repeats)
    for name,data in report.pop('payloads').items():
        target=a.output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    report.update(requested_gpu=a.gpu,remote_wall_seconds=time.perf_counter()-started,
        source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'])
    if report['status']!='passed-matched-numeric-gates':raise SystemExit(1)


if __name__=='__main__':main()
