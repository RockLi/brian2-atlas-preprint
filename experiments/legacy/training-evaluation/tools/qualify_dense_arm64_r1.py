"""Independent ARM64 Atlas dense Q0: unchanged four-fixture full numerical checks.

Requires an explicit ARM64-only runner; does not build or substitute the binary.
"""
import argparse,copy,hashlib,json,sys,time,traceback
from pathlib import Path
from arm64_artifact_gate_r1 import ArchitectureGateError, gate_artifact
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'adapters'))
def load_numeric_dependencies():
    global np, forward_vjp, adam
    import numpy as np
    from oracle import forward_vjp, adam



def compare(actual,expected,exact=False):
    a=np.asarray(actual);b=np.asarray(expected)
    delta=np.abs(a-b)
    return dict(passed=bool(np.array_equal(a,b) if exact else np.all(delta<=1e-10+1e-8*np.abs(b))),
                max_abs=float(delta.max(initial=0)),shape=list(a.shape))


def evaluate(case,engine,layerwise=False,compile_model=False,*,runner,runner_identity):
    if engine=='atlas':
        from atlas_adapter import run_case
        gate_artifact(runner,'runner',runner_identity['sha256'])
        raw=run_case(case,ROOT/'snapshot/brian2-rust',runner,steps=3)
        if raw['status']!='executed':raise RuntimeError(json.dumps(raw))
        result=raw['gradients']['result']
        result=dict(loss=result['loss'],logits=result['logits'],states=raw['state_trace']['states'],spikes=result['spikes'],gradients=result['gradients'],initial_vjp=result['initial_gradients'],updates=[dict(weights=u['result']['state']['weights'],first=u['result']['state']['first_moment'],second=u['result']['state']['second_moment'],step=u['result']['state']['step']) for u in raw['updates']])
        gate_artifact(runner,'runner',runner_identity['sha256'])
        sgd=run_case(case,ROOT/'snapshot/brian2-rust',runner,optimizer='sgd',steps=1,collect_states=False)
        raw['sgd']=sgd
        result['one_sgd_update']=sgd['updates'][0]['result']['state']['weights']
    else:
        from torch_adapter import run_case
        raw=run_case(case,engine=engine,steps=3,layerwise=layerwise,compile_model=compile_model);result=raw
    ref=forward_vjp(case)
    checks={k:compare(result[k],ref[k],k=='spikes') for k in ['loss','logits','states','spikes','initial_vjp']}
    for i,(a,b) in enumerate(zip(result['gradients'],ref['gradients'])):checks[f'gradient_{i}']=compare(a,b)
    for i,(w,g,update) in enumerate(zip(case['weights'],ref['gradients'],result['one_sgd_update'])):checks[f'sgd_{i}']=compare(update,np.asarray(w)-.001*g)
    if 'local_surrogate_vjp' in result:
        local=result['local_surrogate_vjp'];margin=np.asarray(local['margins']);cot=np.asarray(local['cotangents'])
        checks['local_surrogate_forward']=compare(local['values'],(margin>0).astype(float),True)
        checks['local_surrogate_vjp']=compare(local['vjp'],cot/(1+5*np.abs(margin))**2)
    weights=[np.asarray(w) for w in case['weights']];m=[np.zeros_like(w) for w in weights];v=copy.deepcopy(m)
    for step,update in enumerate(result['updates'],1):
        gradients=forward_vjp(case,weights)['gradients']
        weights,m,v=adam(weights,gradients,m,v,step)
        for key,expected in [('weights',weights),('first',m),('second',v)]:
            for i,(a,b) in enumerate(zip(update[key],expected)):checks[f'adam_{step}_{key}_{i}']=compare(a,b)
        actual_step=update['step'];checks[f'adam_{step}_counter']=dict(passed=bool(np.all(np.asarray(actual_step)==step)))
    return raw,checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',required=True,choices=('atlas',));p.add_argument('--runner',required=True);p.add_argument('--output',required=True);p.add_argument('--layerwise',action='store_true');p.add_argument('--compile',action='store_true');args=p.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    if (dest/'report.json').exists():raise FileExistsError(dest/'report.json')
    runner=Path(args.runner).resolve()
    try:
        runner_identity=gate_artifact(runner,'runner')
    except Exception as error:
        report=dict(schema='dense-Q0-arm64-r1',engine=args.engine,performance_run=False,
                    native_profile='arm64-r1',execution_status='not_executed',
                    dense_qualification_status='failed',qualification_status='unqualified',
                    stage='runner_architecture_gate',error_type=type(error).__name__,error=str(error),
                    runner=getattr(error,'evidence',dict(path=str(runner))),
                    cases={n:dict(passed=False,status='not_launched') for n in (
                        'base_negative_count_input','batch_duplicate','initial_threshold_boundary','single_sample_no_carry')})
        (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));return 1
    (dest/'runner-identity.json').write_text(json.dumps(runner_identity,indent=2)+'\n')
    load_numeric_dependencies()
    base=json.loads((ROOT/'fixtures/q0.json').read_text());cases={'base_negative_count_input':base}
    duplicate=copy.deepcopy(base);duplicate['inputs']*=2;duplicate['labels']*=2;cases['batch_duplicate']=duplicate
    initial=copy.deepcopy(base);initial['initial']=[[1/.95,1/.95-1e-7,1/.95+1e-7,-.5,1/.95,0],[.2,-.3,0,1.8,.7,-.8]];cases['initial_threshold_boundary']=initial
    isolated=copy.deepcopy(base);isolated['inputs']=isolated['inputs'][1:];isolated['labels']=isolated['labels'][1:];cases['single_sample_no_carry']=isolated
    report=dict(schema='dense-Q0-arm64-r1',performance_run=False,native_profile='arm64-r1',runner=runner_identity,engine=args.engine,scope='dense synchronous Q0 only; SGD, Adam and CE-induced complete-network VJP',oracle='NumPy explicit analytic reverse VJP, no hard-spike finite difference',cases={},not_yet_qualified=['recurrent','delay','tied_weights','trainable_dynamics','arbitrary_external_spike_cotangent'],atol=1e-10,rtol=1e-8)
    start=time.perf_counter()
    for name,case in cases.items():
        try:
            raw,checks=evaluate(case,args.engine,args.layerwise,args.compile,runner=runner,runner_identity=runner_identity)
            (dest/(name+'.json')).write_text(json.dumps(dict(case=case,raw=raw,checks=checks),indent=2,allow_nan=False)+'\n')
            report['cases'][name]=dict(passed=all(c['passed'] for c in checks.values()),checks=checks)
        except Exception as e:
            report['cases'][name]=dict(passed=False,error=str(e),traceback=traceback.format_exc())
    report['dense_qualification_status']='passed' if all(r['passed'] for r in report['cases'].values()) else 'failed'
    report['qualification_status']='partial' if report['dense_qualification_status']=='passed' else 'unqualified'
    report['elapsed_s']=time.perf_counter()-start
    report['identities']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'adapters/oracle.py',ROOT/'adapters/torch_adapter.py',ROOT/'adapters/atlas_adapter.py',ROOT/'fixtures/q0.json',Path(__file__),ROOT/'tools/arm64_artifact_gate_r1.py']}
    report['identities'][str(runner)]=runner_identity['sha256']
    report['execution_status']='completed'
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['dense_qualification_status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
