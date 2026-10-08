"""Bounded 2x2 GeNN recurrent arithmetic/input-order ablation on one GPU."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]


def diagnose(neurons,steps,degree,repeats):
    import os,signal,subprocess,sys,tempfile
    import numpy as np
    sys.path.insert(0,'/workspace/brian2-rust/examples')
    from gpu_recurrent import GENN_POLICIES
    from modal_gpu_compare import paired_checks,completed_trial,PRECISE_BASIC_FLAGS
    report=dict(schema='b2-genn-arithmetic-ablation-v0',neurons=neurons,steps=steps,degree=degree,
        repeats=repeats,order_seed=1729,math_policy='precise-basic',
        timing_scope='diagnostic fresh-project adapter wall; compilation/initialization/readback included; not simulation throughput',
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),
        host_cpu=Path('/proc/cpuinfo').read_text(),allocated_cpu_cores=4,
        locks={p.name:p.read_text() for p in Path('/opt/baseline-locks').glob('*.txt')},trials=[])
    rng=random.Random(1729)
    with tempfile.TemporaryDirectory(prefix='b2-genn-arithmetic-') as temporary:
        for repeat in range(repeats):
            order=[('cpu-f32','legacy'),('cuda','legacy')]+[('genn',p) for p in GENN_POLICIES]
            rng.shuffle(order)
            for position,(backend,policy) in enumerate(order):
                label=backend if backend!='genn' else 'genn-'+policy
                output=Path(temporary)/f'{repeat}-{label}'
                env={**os.environ,'CUDA_PATH':'/usr/local/cuda','NVCC_APPEND_FLAGS':PRECISE_BASIC_FLAGS}
                env.pop('NVCC_PREPEND_FLAGS',None)
                if backend=='genn':env['PYTHONPATH']=''
                interpreter='/opt/genn5-env/bin/python' if backend=='genn' else sys.executable
                start=time.perf_counter()
                process=subprocess.Popen([interpreter,'/workspace/brian2-rust/examples/gpu_baseline.py',
                    '--backend',backend,'--case','recurrent-cuba-v0','--neurons',str(neurons),
                    '--steps',str(steps),'--degree',str(degree),'--output',str(output),
                    '--genn-recurrent-policy',policy],env=env,cwd=temporary,stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,text=True,start_new_session=True)
                try:
                    stdout,stderr=process.communicate(timeout=300);code=process.returncode
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);stdout,stderr=process.communicate();code=None
                row=dict(backend=backend,policy=policy,label=label,repeat=repeat,order=position,
                    exit_code=code,stdout=stdout,stderr=stderr,process_wall_seconds=time.perf_counter()-start,
                    report=json.loads((output/'report.json').read_text()) if (output/'report.json').exists() else None)
                if (output/'result.npz').exists():row['result_npz']=(output/'result.npz').read_bytes()
                row['generated_files']={str(p.relative_to(output)):p.read_text() for p in output.rglob('*')
                    if p.is_file() and (p.name=='neuronUpdate.cc' or p.name.lower()=='makefile') and p.stat().st_size<200000}
                report['trials'].append(row);print(label,repeat,code,flush=True)
            if any(not completed_trial(r) for r in report['trials']):break
    report['execution_completed']=len(report['trials'])==6*repeats and all(completed_trial(r) for r in report['trials'])
    reports=[r['report'] for r in report['trials'] if r['report']]
    report['identity_checks']=dict(all_reports_present=len(reports)==len(report['trials']),
        same_model=len({r['model_sha256'] for r in reports})==1,
        same_gpu={r['nvidia_smi'] for r in reports}=={report['nvidia_smi']})
    report['pairwise_against_cpu_f32']=[]
    for repeat in range(repeats):
        rows=[r for r in report['trials'] if r['repeat']==repeat and completed_trial(r)]
        control=next((r for r in rows if r['backend']=='cpu-f32'),None)
        if control is None:continue
        with np.load(io.BytesIO(control['result_npz'])) as data:expected=dict(data)
        for row in rows:
            with np.load(io.BytesIO(row['result_npz'])) as data:actual=dict(data)
            check=paired_checks(actual,expected,'recurrent-cuba-v0',neurons,steps)
            changed=[int(i) for i in range(neurons) if not np.array_equal(
                actual['ticks'][actual['indices']==i],expected['ticks'][expected['indices']==i])]
            report['pairwise_against_cpu_f32'].append(dict(label=row['label'],repeat=repeat,
                neurons_with_different_spike_trains=changed,**check))
    report['reference_gate_passed']=report['execution_completed'] and all(r['status']=='passed' for r in reports)
    report['status']=('completed' if report['execution_completed'] and all(report['identity_checks'].values()) else 'execution_or_identity_failure')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4')
    parser.add_argument('--neurons',type=int,default=4096);parser.add_argument('--steps',type=int,default=2048)
    parser.add_argument('--degree',type=int,default=32);parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from gpu_recurrent import validate_size
    if not 1<=args.neurons<=65536 or not 1<=args.steps<=4096 or not 1<=args.repeats<=3:parser.error('bounded positive sizes and 1..3 repeats required')
    try:validate_size(args.neurons,args.degree)
    except ValueError as error:parser.error(str(error))
    args.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    names=('gpu_baseline.py','gpu_recurrent.py','gpu_hh.py','modal_gpu_compare.py','modal_cuda_tests.py','modal_genn_arithmetic.py','install_gpu_baselines.sh')
    sources={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT.parent/'brian2',ROOT/'src') for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for p in [*(ROOT/'examples'/n for n in names),ROOT/'Cargo.toml',ROOT/'Cargo.lock',
              *(ROOT.parent/n for n in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        sources[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output/'source-hashes.json').write_text(json.dumps(sources,sort_keys=True,indent=2)+'\n')
    image=(native_image().apt_install('libffi-dev','pkg-config').pip_install('uv==0.8.22')
        .add_local_file(ROOT/'examples/install_gpu_baselines.sh','/opt/install_gpu_baselines.sh',copy=True)
        .run_commands('sh /opt/install_gpu_baselines.sh')
        .add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=['**/__pycache__/**','**/*.pyc']))
    for name in names:
        if name!='install_gpu_baselines.sh':image=image.add_local_file(ROOT/'examples'/name,'/workspace/brian2-rust/examples/'+name)
    image=image.add_local_file(Path(__file__),'/root/modal_genn_arithmetic.py')
    app=modal.App('brian2-genn-arithmetic-ablation',include_source=False)
    run=app.function(image=image,gpu=args.gpu,cpu=4,memory=8192,max_containers=1,timeout=1800,retries=0)(diagnose)
    start=time.perf_counter()
    with modal.enable_output(),app.run():report=run.remote(args.neurons,args.steps,args.degree,args.repeats)
    report.update(requested_gpu=args.gpu,remote_wall_seconds=time.perf_counter()-start,
        source_manifest_sha256=hashlib.sha256((args.output/'source-hashes.json').read_bytes()).hexdigest())
    for row in report['trials']:
        payload=row.pop('result_npz',None)
        if payload is not None:
            name=f"{row['repeat']}-{row['label']}.npz";(args.output/name).write_bytes(payload)
            row['result_artifact']=dict(path=name,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],'reference gate:',report['reference_gate_passed'])
    for row in report['pairwise_against_cpu_f32']:print(row['label'],row['repeat'],row['passed'],len(row['neurons_with_different_spike_trains']))
    if report['status']!='completed' or not report['reference_gate_passed']:raise SystemExit(1)


if __name__=='__main__':main()
