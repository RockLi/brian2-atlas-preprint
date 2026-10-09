"""Frozen E1 small Atlas public API benchmark, one fresh process per seed.

Prepare common arrays with --prepare-only (no Atlas import), then run on the
approved remote machine. This entry point does not optimize or compile Atlas.
Five processes/seeds, each 10 warmups + 50 measured full training steps; 360 s
per worker and 1800 s aggregate cap include qualification/preflight/I/O overhead.
"""
from __future__ import annotations
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import threading
import traceback
import numpy as np

SEEDS=(11,23,37,51,71)
RULE={'sizes':[128,128,10],'B':16,'T':128,'input_values':[0,1,2],
      'input_probabilities':[.9,.095,.005], 'weight_distribution':'independent Normal(0,sqrt(2/fan_in)) per bank',
      'labels':'arange(B)%C','beta':.95,'theta':1.,'seeds':list(SEEDS),
      'precision':'arrays cast to IEEE f64 in compute; integer count JSON is lossless',
      'numpy_rng':'default_rng PCG64; draw inputs then banks in network order'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def write_new(path,value):
    path=Path(path)
    with path.open('xb') as f:f.write(encode(value)+b'\n')


def e1_fixture(seed):
    rng=np.random.default_rng(seed);sizes=RULE['sizes']
    x=rng.choice([0,1,2],size=(16,128,128),p=[.9,.095,.005])
    w=[rng.normal(0.,np.sqrt(2/a),size=(a,b)) for a,b in zip(sizes[:-1],sizes[1:])]
    return dict(id=f'E1-small-seed-{seed}',seed=seed,sizes=sizes,inputs=x.tolist(),
                labels=(np.arange(16)%10).tolist(),weights=[a.ravel().tolist() for a in w],beta=.95,theta=1.)


def prepare(base):
    directory=base/'fixtures/e1-small';directory.mkdir(parents=True,exist_ok=True)
    existing=directory/'manifest.json'
    if existing.exists():
        frozen=json.loads(existing.read_text())
        if frozen['rule']!=RULE:raise ValueError('frozen E1 generation rule differs')
        if set(frozen['files'])!={f'seed-{seed}.json' for seed in SEEDS}:raise ValueError('frozen seed denominator differs')
        for filename,identity in frozen['files'].items():
            if digest(directory/filename)!=identity['sha256']:raise ValueError(f'common array hash mismatch: {filename}')
        return existing  # Common arrays are authoritative across NumPy versions.
    manifest=dict(schema='e1-common-arrays-v1',rule=RULE,numpy_version=np.__version__,files={})
    for seed in SEEDS:
        case=e1_fixture(seed);path=directory/f'seed-{seed}.json';data=encode(case)+b'\n'
        if path.exists():
            if path.read_bytes()!=data:raise ValueError(f'frozen common array differs: {path}')
        else:
            with path.open('xb') as f:f.write(data)
        manifest['files'][path.name]={'sha256':digest(path),'bytes':len(data)}
    path=directory/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:raise ValueError('frozen E1 manifest differs')
    else:write_new(path,manifest)
    return path


def compare(actual,reference,exact=False):
    a=np.asarray(actual);b=np.asarray(reference);d=np.abs(a-b)
    return dict(passed=bool(np.array_equal(a,b) if exact else np.all(d<=1e-10+1e-8*np.abs(b))),
                max_absolute_error=float(np.max(d,initial=0)),shape=list(b.shape))


def qualify(adapter,case,oracle):
    reference=oracle.forward_vjp(case);actual=adapter.call('train')
    report=dict(execution=actual,scope='dense synchronous LIF only; Q0 all-state qualification separately required')
    if actual['status']!='executed':report['status']=actual['status'];return report
    a=actual['result'];checks={k:compare(a[k],reference[k],exact=k=='spikes') for k in ('loss','logits','spikes')}
    checks['final_state']=compare(a['final_membrane'],reference['states'][:,-1])
    checks['initial_vjp']=compare(a['initial_gradients'],reference['initial_vjp'])
    w=[np.asarray(z,dtype=np.float64) for z in case['weights']];zero=[np.zeros_like(z) for z in w]
    expected,m,v=oracle.adam(w,reference['gradients'],zero,zero,1)
    for bank,g in enumerate(reference['gradients']):
        checks[f'gradient_{bank}']=compare(a['gradients'][bank],g)
        for name,values in [('weights',expected),('first_moment',m),('second_moment',v)]:
            checks[f'adam_{name}_{bank}']=compare(a['state'][name][bank],values[bank])
    checks['adam_step']={'passed':a['state']['step']==1}
    checks['native_admission_count']={'passed':actual['tape_count_matches_runtime']}
    report['checks']=checks;report['status']='qualified' if all(v['passed'] for v in checks.values()) else 'unqualified'
    return report


class RSSSampler:
    """One contemporaneous process-tree sample every 50 ms, no peak summation."""
    def __init__(self,directory):
        import psutil
        self.psutil=psutil;self.root=psutil.Process();self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self.sample,daemon=True)
        self.path=directory/'rss-samples.jsonl';self.individual={};self.peak=0;self.phase_peaks={};self.samples=0;self.errors=[];self.phase='setup'
    def start(self):self.thread.start()
    def sample(self):
        with self.path.open('x') as f:
            while not self.stop_event.is_set():
                stamp=time.monotonic_ns();rows=[]
                try:
                    for process in [self.root,*self.root.children(recursive=True)]:
                        try:
                            rss=process.memory_info().rss;rows.append({'pid':process.pid,'rss':rss})
                            self.individual[process.pid]=max(self.individual.get(process.pid,0),rss)
                        except (self.psutil.NoSuchProcess,self.psutil.ZombieProcess):pass
                    total=sum(row['rss'] for row in rows);self.peak=max(self.peak,total);self.phase_peaks[self.phase]=max(self.phase_peaks.get(self.phase,0),total);self.samples+=1
                    f.write(json.dumps({'monotonic_ns':stamp,'sample_duration_ns':time.monotonic_ns()-stamp,'phase':self.phase,'job_rss':total,'processes':rows},separators=(',',':'))+'\n');f.flush()
                except Exception as error:self.errors.append(f'{type(error).__name__}: {error}')
                self.stop_event.wait(.05)
    def stop(self):
        self.stop_event.set();self.thread.join()
        return dict(interval_seconds=.05,job_peak_rss_bytes=self.peak,job_peak_rss_bytes_by_phase=self.phase_peaks,individual_peak_rss_bytes=self.individual,
                    samples=self.samples,errors=self.errors,raw_samples=str(self.path),
                    scope='worker parent plus recursive children at each sample; individual peaks are not added; short-lived allocations below 50 ms may be missed')


def worker(args):
    base=args.base.resolve();directory=args.output.resolve();directory.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(base/'adapters'))
    start=time.monotonic();report=dict(schema='e1-atlas-process-v1',seed=args.seed,status='unqualified',
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),engine='Atlas',
        timer='coordinator full public execute; includes JSON/tempfile/subprocess/native compute/result parse',
        warmup_steps=10,measured_steps=50,case_budget_seconds=360,measured_public_api_ns=[])
    sampler=None
    try:
        sampler=RSSSampler(directory);sampler.start()
        from atlas_adapter import AtlasAdapter, failure_status
        import oracle
        manifest_path=base/'fixtures/e1-small/manifest.json';manifest=json.loads(manifest_path.read_text())
        case_path=manifest_path.parent/f'seed-{args.seed}.json'
        assert digest(case_path)==manifest['files'][case_path.name]['sha256']
        case=json.loads(case_path.read_text())
        report['allocation']={'native_compute_threads':1,'mpi_ranks':1,'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}}
        report['freeze_sha256']=digest(directory.parent/'freeze.json')
        report.update(case_sha256=digest(case_path),fixture_manifest_sha256=digest(manifest_path),
                      adapter_sha256=digest(base/'adapters/atlas_adapter.py'),oracle_sha256=digest(base/'adapters/oracle.py'),
                      benchmark_sha256=digest(__file__),runtime_sha256=digest(args.runner))
        constructor_start=time.perf_counter_ns()
        adapter=AtlasAdapter(case,args.source_root,args.runner)
        report['first_constructor_ns']=time.perf_counter_ns()-constructor_start
        sampler.phase='qualification'
        q=qualify(adapter,case,oracle)
        with gzip.open(directory/'first-step-qualification.json.gz','wt') as f:json.dump(q,f,separators=(',',':'),allow_nan=False)
        report['qualification_status']=q['status'];report['qualification_checks']=q.get('checks')
        report['first_public_api_ns']=q['execution'].get('public_api_ns')
        if q['status']!='qualified':report['status']=q['status'];return report
        del q  # Qualification trace storage does not remain live during performance.
        # Restart at identical arrays after qualification, then continuously train.
        adapter=AtlasAdapter(case,args.source_root,args.runner)
        report['initial_state']='fresh initial weights and zero Adam moments after separate qualification'
        with gzip.open(directory/'raw-steps.jsonl.gz','wt') as raw:
            for index in range(60):
                if time.monotonic()-start>=360:report['status']='timeout';break
                sampler.phase='warmup' if index<10 else 'measured'
                result=adapter.call('train')
                item=dict(index=index,phase='warmup' if index<10 else 'measured',**result)
                if result['status']=='executed':
                    item['spike_count']=int(np.asarray(result['result']['spikes']).sum())
                raw.write(json.dumps(item,separators=(',',':'),allow_nan=False)+'\n');raw.flush()
                adapter.records.clear()
                if result['status']!='executed':report['status']=result['status'];report['failed_step']=index;break
                if not result['tape_count_matches_runtime']:
                    report['status']='unqualified';report['reason']='native admission count mismatch';break
                if index>=10:report['measured_public_api_ns'].append(result['public_api_ns'])
            else:report['status']='completed'
        if report['status']=='completed':
            report['median_public_api_ns']=statistics.median(report['measured_public_api_ns'])
        report['completed_measured_steps']=len(report['measured_public_api_ns'])
    except Exception as error:
        report.update(status=locals().get('failure_status',lambda e:'dependency_failed')(error),
                      error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        if sampler is not None:report['rss']=sampler.stop()
        report['process_wall_seconds']=time.monotonic()-start
        write_new(directory/'result.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--source-root',type=Path);parser.add_argument('--runner',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--q0-report',type=Path,help='fresh matching adapter dense Q0 qualification report')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args();args.base=args.base.resolve()
    if args.source_root is None:args.source_root=args.base/'snapshot/brian2-rust'
    if args.runner is None:args.runner=args.base/'runtime/b2-train'
    if args.worker:
        print(json.dumps(worker(args),separators=(',',':')));return
    manifest=prepare(args.base)
    if args.prepare_only:print(json.dumps({'manifest':str(manifest),'sha256':digest(manifest)}));return
    if args.output is None:parser.error('--output is required to run; choose a fresh evidence directory')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    q0_path=args.q0_report or args.base/'evidence/q0-atlas-v2/report.json'
    q0=json.loads(q0_path.read_text())
    if q0.get('dense_qualification_status')!='passed':raise ValueError('dense Q0 qualification is required')
    if q0.get('identities',{}).get('adapters/atlas_adapter.py')!=digest(args.base/'adapters/atlas_adapter.py'):
        raise ValueError('Q0 adapter hash is stale; rerun qualification for the current adapter')
    source_manifest=args.base/'sources/snapshot-manifest.json'
    declared=json.loads(source_manifest.read_text())
    source_mismatches=[p for p,h in declared['source_hashes'].items() if digest(args.base/'snapshot'/p)!=h]
    if source_mismatches:raise ValueError(f'frozen snapshot changed: {source_mismatches}')
    freeze=dict(schema='e1-formal-freeze-v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        fixture_manifest_sha256=digest(manifest),source_manifest_sha256=digest(source_manifest),
        source_commit=declared['commit'],source_hashes=declared['source_hashes'],
        runner_sha256=digest(args.runner),adapter_sha256=digest(args.base/'adapters/atlas_adapter.py'),
        oracle_sha256=digest(args.base/'adapters/oracle.py'),benchmark_sha256=digest(__file__),
        q0_report_sha256=digest(q0_path),q0_scope='dense synchronous only',
        hardware_hashes={p.name:digest(p) for p in (args.base/'environment').glob('*hardware*.json')},
        rule=RULE,warmup_steps=10,measured_steps=50,per_seed_seconds=360,total_seconds=1800,
        threads=1,ranks=1,rss_sample_seconds=.05)
    write_new(args.output/'freeze.json',freeze)
    start=time.monotonic();out=dict(schema='e1-atlas-benchmark-v1',fixture_manifest_sha256=digest(manifest),
        frozen_rule=RULE,sampling='five independent processes; one seed each; 10 warmups + 50 measured',
        aggregate_budget_seconds=1800,per_process_budget_seconds=360,processes=[])
    for seed in SEEDS:
        remaining=1800-(time.monotonic()-start)
        row=dict(seed=seed)
        if remaining<=0:row['status']='timeout';row['reason']='aggregate budget exhausted before launch';out['processes'].append(row);continue
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--base',str(args.base),
                 '--source-root',str(args.source_root.resolve()),'--runner',str(args.runner.resolve()),
                 '--seed',str(seed),'--output',str(args.output/f'seed-{seed}')]
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[key]='1'
        row['command']=command
        with (args.output/f'seed-{seed}.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:row['exit_code']=process.wait(timeout=min(360,remaining))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();row['status']='timeout';row['exit_code']=process.returncode
        result_path=args.output/f'seed-{seed}/result.json'
        if result_path.exists():row.update(json.loads(result_path.read_text()))
        elif 'status' not in row:row['status']='software_rejected';row['reason']='worker exited without result'
        out['processes'].append(row)
    completed=[row for row in out['processes'] if row['status']=='completed']
    out['status']='completed' if len(completed)==5 else 'incomplete'
    out['aggregate_wall_seconds']=time.monotonic()-start
    if len(completed)==5:
        medians=[row['median_public_api_ns'] for row in completed]
        out['summary']={'median_of_process_medians_ns':statistics.median(medians),'min_process_median_ns':min(medians),'max_process_median_ns':max(medians),'independent_n':5}
    write_new(args.output/'summary.json',out)
    print(json.dumps({'status':out['status'],'output':str(args.output)}))


if __name__=='__main__':main()
