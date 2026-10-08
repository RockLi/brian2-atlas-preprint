"""Alternate both host readbacks on one corrected, fingerprinted GeNN model."""
import hashlib,json,random,statistics,sys,time
from pathlib import Path
import numpy as np
from gpu_genn_barrier_adapter import run as build
from gpu_genn_readback import prepare,replay as gather_replay
from gpu_stdp_compare import oracle,checks,configuration,genn_replay
from gpu_stdp_precompiled import artifact_hashes,forbid_compilation,write_result

N=4096;DEGREE=8;STEPS=256;REPEATS=5
OPTIONS=dict(drive=10/256,delay_span=16,post_delay=16,topology_kind='random-fixed-outdegree',topology_seed=42)


def order(seed=1729,repeats=REPEATS):
    rng=random.Random(seed)
    for i in range(-1,repeats):
        modes=['public','gather'];rng.shuffle(modes)
        for position,mode in enumerate(modes):yield i,position,mode


def timed_replay(context,mode,steps):
    if mode not in ('public','gather'):raise ValueError('Unknown readback mode')
    model=context['model'];cls=context['host_gather']['variable_class'];getter=cls.values
    start=time.perf_counter()
    with forbid_compilation():
        actual=(genn_replay if mode=='public' else gather_replay)(context,steps)
    elapsed=time.perf_counter()-start
    if context['model'] is not model or cls.values is not getter:
        raise RuntimeError('Model identity or public getter changed')
    return actual,dict(wall_seconds=elapsed,phases=dict(context['last_replay_timings']),
        gather=dict(context['last_gather']) if mode=='gather' else None)


def compare(output):
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='genn-same-model-readback-v1',configuration=configuration(N,DEGREE,STEPS,**OPTIONS),
        model_preparations=1,order_seed=1729,repeats=REPEATS,samples=[],bootstraps={},status='running',
        scope='One corrected GeNN model and compiled library. Load/reset, steps, complete readback and unload are timed; compilation, fingerprints, gates and NPZ output are excluded.')
    def persist():(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    f64=oracle(N,DEGREE,STEPS,**OPTIONS);f32=oracle(N,DEGREE,STEPS,dtype=np.float32,**OPTIONS)
    report['f32_oracle_gate']=checks(f32,f64)
    if not report['f32_oracle_gate']['passed']:raise RuntimeError('Independent f32 oracle violates original f64 gate')
    write_result(output,'reference-f64',f64);write_result(output,'reference-f32',f32)
    def capture(actual,name):
        result=write_result(output,name,actual);result['path']=Path(result['path']).name
        gate=dict(f64=checks(actual,f64),f32=checks(actual,f32))
        row=dict(result=result,gates=gate)
        return row,all(g['passed'] for g in gate.values())
    try:
        context={};initial,timing=build(N,DEGREE,STEPS,output,prepared=context,**OPTIONS)
        report['setup']=timing;fingerprints=artifact_hashes(output)
        if not fingerprints:raise RuntimeError('No compiled artifact')
        report['compiled_before']=fingerprints
        report['host_gather']=prepare(context)
        row,passed=capture(initial,'bootstrap-public');report['bootstraps']['public']=row;persist()
        if not passed:raise RuntimeError('Public bootstrap failed gate')
        with forbid_compilation():actual=gather_replay(context,STEPS,verify=True)
        report['gather_crosscheck']=dict(context['last_gather'])
        row,passed=capture(actual,'bootstrap-gather');report['bootstraps']['gather']=row;persist()
        if not passed:raise RuntimeError('Gather bootstrap failed gate')
        if artifact_hashes(output)!=fingerprints:raise RuntimeError('Artifact changed during bootstrap')
        for i,position,mode in order():
            actual,timing=timed_replay(context,mode,STEPS)
            after=artifact_hashes(output)
            if after!=fingerprints:raise RuntimeError('Compiled artifact changed between modes')
            row,passed=capture(actual,f'{mode}-{i}')
            row.update(mode=mode,round=i,position=position,**timing,compiled=after)
            report['samples'].append(row);persist()
            if not passed:raise RuntimeError(mode+' replay failed gate')
        report['compiled_after']=artifact_hashes(output)
        report['summary']={}
        for mode in ('public','gather'):
            rows=[r for r in report['samples'] if r['mode']==mode and r['round']>=0]
            values=[r['wall_seconds'] for r in rows]
            report['summary'][mode]=dict(samples_seconds=values,median_seconds=statistics.median(values),
                min_seconds=min(values),max_seconds=max(values),
                phase_medians={k:statistics.median(r['phases'][k] for r in rows) for k in rows[0]['phases']})
        report['sources']={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (output/'project').rglob('*') if p.is_file() and p.suffix in {'.cc','.h','.cpp','.hpp','.cu'}}
        report['status']='passed';persist()
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));persist();raise
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':compare(Path(sys.argv[1]))
