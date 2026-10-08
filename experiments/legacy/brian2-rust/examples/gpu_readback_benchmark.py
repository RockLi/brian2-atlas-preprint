"""Interleaved previous/result-only GPU readback with complete-result gates."""
import argparse,hashlib,json,statistics,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from cuda_dag_benchmark import workload as cuba,result_digest
from gpu_decode_benchmark import snapshot
from brian2_rust.metal import MetalExecutor
from brian2_rust.cuda import CudaExecutor
from brian2_rust import gpu_readback
from unittest.mock import patch
MODES=('previous','results')


def benchmark(case,repeats,output,backend):
    import brian2 as b
    from brian2_rust.export import lower_network
    output.mkdir(parents=True,exist_ok=False);b.get_device().reinit()
    n=1024 if case=='cuba-small' else 4096;steps=256;degree=32
    references={}
    if case=='stdp-random':
        from gpu_stdp_compare import build_brian,oracle,DT
        b.prefs.core.default_float_dtype=np.float32
        b.set_device('rust_standalone',engine='reference',directory=output/'model',runner=Path(__file__).resolve().parents[1]/'target/release/b2-runner')
        opts=dict(drive=.0625,delay_span=8,post_delay=3,topology_kind='random-fixed-outdegree',topology_seed=42)
        net,*_=build_brian(n,degree,**opts);model=lower_network(net,steps*DT*b.second)
        for name,dtype in [('f64',np.float64),('f32',np.float32)]:
            references[name]=oracle(n,degree,steps,dtype=dtype,**opts)
            np.savez_compressed(output/('oracle-'+name+'.npz'),**references[name])
    else:model=cuba(n,steps,degree,output/'model')
    encoded=json.dumps(model,sort_keys=True)+'\n';(output/'model.json').write_text(encoded)
    report=dict(schema='b2-gpu-readback-v0',backend=backend,case=case,neurons=n,degree=degree,steps=steps,repeats=repeats,
        model_sha256=hashlib.sha256(encoded.encode()).hexdigest(),samples=[],snapshots={},setup={},gates={},
        scope='Independent prepared executors; seed-1729 randomized warm order. Wall includes state reset, commands, GPU synchronization and complete host result arrays; excludes setup, hashing and serialization. Cold and warmup are separate. CUBA checks previous-f32 equivalence only; STDP additionally checks independent f32/f64 oracles.')
    new_readback=gpu_readback.readback_bindings
    def old_readback(plan):
        return tuple(sorted({i for d in plan.dispatches for i,t in zip(d.bindings,d.types,strict=True) if not t.startswith('const ')}))
    expected=None
    with ExitStack() as stack:
        executors={}
        for mode in MODES:
            start=time.perf_counter();ex=stack.enter_context((MetalExecutor if backend=='metal' else CudaExecutor)(model,output/mode,numeric_mode='float32',event_delivery='sparse',dag_execution='resident' if backend=='metal' else 'auto'));executors[mode]=ex
            report['setup'][mode]=dict(seconds=time.perf_counter()-start,compile_seconds=ex.compile_seconds,device=ex.device_name,plan_sha256=ex.plan.sha256)
            (output/(mode+'-plan.json')).write_text(ex.plan.to_json())
        assert len({ex.plan.sha256 for ex in executors.values()})==1
        def measure(mode,index,order):
            nonlocal expected
            with patch.object(gpu_readback,'readback_bindings',old_readback if mode=='previous' else new_readback):
                start=time.perf_counter();result=executors[mode].run();wall=time.perf_counter()-start
            digest=result_digest(result)
            if expected is None:expected=digest
            assert digest==expected,'readback changed complete result'
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
    p.add_argument('--case',choices=('cuba-small','cuba-large','stdp-random'),required=True);a=p.parse_args()
    if not 3<=a.repeats<=15:p.error('repeats must be within 3..15')
    r=benchmark(a.case,a.repeats,a.output,a.backend);print(json.dumps(r['summary'],indent=2))
if __name__=='__main__':main()
