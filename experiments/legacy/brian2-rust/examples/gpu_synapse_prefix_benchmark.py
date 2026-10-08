"""Interleaved target-owned/edge-prefix GPU plasticity with full gates."""
import argparse,hashlib,json,statistics,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from cuda_dag_benchmark import result_digest
from gpu_decode_benchmark import snapshot
from brian2_rust.metal import MetalExecutor
from brian2_rust.cuda import CudaExecutor
MODES=('target','prefix')


def benchmark(case,repeats,output,backend,degree=32,neurons=4096):
    import brian2 as b
    from brian2_rust.export import lower_network
    output.mkdir(parents=True,exist_ok=False);b.get_device().reinit()
    n=neurons;steps=256
    references={}
    assert case=='stdp-random'
    from gpu_stdp_compare import build_brian,oracle,DT
    b.prefs.core.default_float_dtype=np.float32
    b.set_device('rust_standalone',engine='reference',directory=output/'model',runner=Path(__file__).resolve().parents[1]/'target/release/b2-runner')
    opts=dict(drive=.0625,delay_span=8,post_delay=3,topology_kind='random-fixed-outdegree',topology_seed=42)
    net,*_=build_brian(n,degree,**opts);model=lower_network(net,steps*DT*b.second)
    for name,dtype in [('f64',np.float64),('f32',np.float32)]:
        references[name]=oracle(n,degree,steps,dtype=dtype,**opts)
        np.savez_compressed(output/('oracle-'+name+'.npz'),**references[name])
    encoded=json.dumps(model,sort_keys=True)+'\n';(output/'model.json').write_text(encoded)
    report=dict(schema='b2-gpu-synapse-prefix-v0',backend=backend,case=case,neurons=n,degree=degree,steps=steps,repeats=repeats,
        model_sha256=hashlib.sha256(encoded.encode()).hexdigest(),samples=[],snapshots={},setup={},gates={},
        scope='Independent prepared executors; seed-1729 randomized warm order. Wall includes state reset, commands, GPU synchronization and complete host result arrays; excludes setup, hashing and serialization. Cold and warmup are separate. STDP checks complete previous-f32 equivalence and independent f32/f64 oracles.')
    expected=None
    with ExitStack() as stack:
        executors={}
        for mode in MODES:
            start=time.perf_counter();ex=stack.enter_context((MetalExecutor if backend=='metal' else CudaExecutor)(model,output/mode,numeric_mode='float32',event_delivery='sparse',synapse_prefix=mode=='prefix',dag_execution='resident' if backend=='metal' else 'auto'));executors[mode]=ex
            report['setup'][mode]=dict(seconds=time.perf_counter()-start,compile_seconds=ex.compile_seconds,device=ex.device_name,plan_sha256=ex.plan.sha256)
            (output/(mode+'-plan.json')).write_text(ex.plan.to_json())
        assert any(d.role=='edge-synapse-prefix' for d in executors['prefix'].plan.dispatches)
        assert not any(d.role=='edge-synapse-prefix' for d in executors['target'].plan.dispatches)
        def measure(mode,index,order):
            nonlocal expected
            start=time.perf_counter();result=executors[mode].run();wall=time.perf_counter()-start
            digest=result_digest(result)
            if expected is None:expected=digest
            assert digest==expected,'prefix changed complete result'
            if mode not in report['snapshots']:
                report['snapshots'][mode]=snapshot(result,output,mode)
                if references:
                    from gpu_stdp_precompiled import native_arrays
                    from gpu_stdp_compare import checks
                    actual=native_arrays(result)
                    report['gates'][mode]={name:checks(actual,ref) for name,ref in references.items()}
                    assert all(g['passed'] for g in report['gates'][mode].values()),'STDP precision gate failed'
            report['samples'].append(dict(mode=mode,round=index,order=order,wall_seconds=wall,timings=result['timings'][0],runtime=result['metal_runtime'] if backend=='metal' else result['cuda_runtime']['dag_execution'],result_sha256=digest))
        for index in (-2,-1):
            for order,mode in enumerate(MODES):measure(mode,index,order)
        rng=np.random.default_rng(1729)
        for index in range(repeats):
            for order,mode in enumerate(rng.permutation(MODES)):measure(str(mode),index,order)
    report['summary']={}
    for mode in MODES:
        rows=[r for r in report['samples'] if r['mode']==mode and r['round']>=0]
        values=dict(wall_seconds=[r['wall_seconds'] for r in rows])
        values.update({k:[r['timings'][k] for r in rows] for k in rows[0]['timings']})
        report['summary'][mode]={k:dict(median=statistics.median(v),min=min(v),max=max(v)) for k,v in values.items()}
    report.update(passed=True,result_sha256=expected)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('metal','cuda'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--neurons',type=int,choices=(2048,4096),default=4096)
    p.add_argument('--degree',type=int,choices=(32,128),default=32)
    p.add_argument('--case',choices=('stdp-random',),required=True);a=p.parse_args()
    if not 3<=a.repeats<=15:p.error('repeats must be within 3..15')
    r=benchmark(a.case,a.repeats,a.output,a.backend,a.degree,a.neurons);print(json.dumps(r['summary'],indent=2))
if __name__=='__main__':main()
