"""Observe an unchanged GeNN program; no throughput claim or relaxed gate."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import gpu_stdp_compare as adapter

N=4096;DEGREE=8;STEPS=256
OPTIONS=dict(drive=10/256,delay_span=16,post_delay=16,topology_kind='random-fixed-outdegree',topology_seed=42)

def first_difference(actual,expected):
    a=set(zip(actual['ticks'].tolist(),actual['indices'].tolist()))
    e=set(zip(expected['ticks'].tolist(),expected['indices'].tolist()))
    return dict(first=list(min(a^e)) if a!=e else None,
                missing=[list(x) for x in sorted(e-a)],extra=[list(x) for x in sorted(a-e)])

def diagnose(output,barrier=False):
    output.mkdir(parents=True,exist_ok=False)
    source,target,delay=adapter.topology(N,DEGREE,**{k:v for k,v in OPTIONS.items() if k in ('delay_span','topology_kind','topology_seed')})
    selected=np.flatnonzero(target==60);neurons=np.unique(np.r_[0,60,N-1,source[selected]])
    context={};report=dict(schema='b2-genn-delay-diagnostic-v0',configuration=adapter.configuration(N,DEGREE,STEPS,**OPTIONS),
        trials={},scope='One unchanged compiled GeNN model; ordinary replays and a replay with host state pulls after each step. Diagnostic only, no performance claim.')
    expected=adapter.oracle(N,DEGREE,STEPS,**OPTIONS)
    np.savez_compressed(output/'reference.npz',**expected)
    def save(name,actual):
        np.savez_compressed(output/(name+'.npz'),**actual)
        report['trials'][name]=dict(checks=adapter.checks(actual,expected),spike_difference=first_difference(actual,expected))
        (output/'report.json').write_text(json.dumps(report,indent=2))
    if barrier:
        from pygenn import GeNNModel
        from genn_postsynaptic_barrier import patch_and_build
        original_build=GeNNModel.build
        def build(self,*args,**kwargs):
            result=original_build(self,*args,**kwargs)
            report['diagnostic_patch']=patch_and_build(output/'project')
            return result
        GeNNModel.build=build
    try:initial,timing=adapter.genn_run(N,DEGREE,STEPS,output,prepared=context,trace_mode='device',**OPTIONS)
    finally:
        if barrier:GeNNModel.build=original_build
    save('bootstrap',initial);report['setup']=timing
    def compiled_hashes():
        return {str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (output/'project').rglob('*') if p.is_file() and p.suffix in {'.cc','.cpp','.cu','.h','.hpp','.so'}}
    compiled=compiled_hashes()
    assert compiled
    report['compiled_before']=compiled
    for i in range(2):save('ordinary-'+str(i),adapter.genn_replay(context,STEPS))
    model=context['model'];pop=context['population'];original_step=model.step_time
    observed={'v_start':[],'v_after_reset':[],'previous_spike':[],**{k:[] for k in adapter.FIELDS}}
    def step():
        original_step()
        for field,label in [('v_start','v_start'),('v','v_after_reset'),('previous_spike','previous_spike')]:
            pop.vars[field].pull_from_device()
            observed[label].append(pop.vars[field].current_values.reshape(-1)[neurons].copy())
        for field in adapter.FIELDS:
            values=np.empty(len(selected),np.float32)
            for syn,edges in context['groups']:
                canonical=edges[syn.synapse_order];mask=np.isin(canonical,selected)
                if not mask.any():continue
                syn.vars[field].pull_from_device()
                values[np.searchsorted(selected,canonical[mask])]=syn.vars[field].values.reshape(-1)[mask]
            observed[field].append(values)
    model.step_time=step
    try:save('observed',adapter.genn_replay(context,STEPS))
    finally:model.step_time=original_step
    np.savez_compressed(output/'observed-state.npz',neurons=neurons,edges=selected,sources=source[selected],
        targets=target[selected],delays=delay[selected],**{k:np.asarray(v) for k,v in observed.items()})
    save('ordinary-after',adapter.genn_replay(context,STEPS))
    report['sources']={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (output/'project').rglob('*') if p.is_file() and p.suffix in {'.cc','.cpp','.cu','.h','.hpp'}}
    assert compiled_hashes()==compiled,'Observation changed compiled files'
    report['completed']=True
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v['checks']['passed'] for k,v in report['trials'].items()}),flush=True)

if __name__=='__main__':diagnose(Path(sys.argv[1]),barrier='--barrier' in sys.argv[2:])
