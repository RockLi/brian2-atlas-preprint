"""One process/seed CPU E1 public training API; shared frozen fixture manifest.

Run through the bounded phase supervisor. No kernel-only timing. Compile mode
is a distinct implementation, qualified before its timings can be compared.
"""
import argparse,hashlib,json,os,sys,time,traceback,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'adapters'))
from oracle import forward_vjp,adam


def check(a,b,exact=False):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return dict(passed=False,actual_shape=list(a.shape),expected_shape=list(b.shape))
    d=np.abs(a-b)
    return dict(passed=bool(np.array_equal(a,b) if exact else np.all(d<=1e-10+1e-8*np.abs(b))),max_abs=float(d.max(initial=0)))


def torch_setup(case,engine,compiled,layerwise):
    import torch
    from torch_adapter import Model
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64)
    model=Model(case,engine,layerwise=layerwise)
    fn=torch.compile(model,fullgraph=True) if compiled else model
    x=torch.tensor(case['inputs'],dtype=torch.float64);y=torch.tensor(case['labels'],dtype=torch.long)
    initial=torch.zeros((len(y),sum(case['sizes'][1:])),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam(model.weights,lr=.001,betas=(.9,.999),eps=1e-8,foreach=False)
    def step(diagnostic=False):
        model.record_states=diagnostic
        opt.zero_grad(set_to_none=True);initial.grad=None
        logits,states,spikes=fn(x,initial);loss=torch.nn.functional.cross_entropy(logits,y)
        loss.backward();opt.step()
        if not diagnostic:return loss
        asnp=lambda t:t.detach().cpu().numpy()
        return dict(loss=asnp(loss),logits=asnp(logits),spikes=asnp(spikes),states=asnp(states),initial_vjp=asnp(initial.grad),gradients=[asnp(w.grad).ravel() for w in model.weights],weights=[asnp(w).ravel().copy() for w in model.weights],first=[asnp(opt.state[w]['exp_avg']).ravel() for w in model.weights],second=[asnp(opt.state[w]['exp_avg_sq']).ravel() for w in model.weights])
    # Checkpoint reset after separate qualification; same compiled graph stays.
    def reset():
        with torch.no_grad():
            for w,z in zip(model.weights,case['weights']):w.copy_(torch.tensor(z).reshape_as(w))
        opt.state.clear()
    def state():
        asnp=lambda t:t.detach().cpu().numpy().ravel()
        return dict(weights=[asnp(w) for w in model.weights],first=[asnp(opt.state[w]['exp_avg']) for w in model.weights],second=[asnp(opt.state[w]['exp_avg_sq']) for w in model.weights])
    def performance_mode():model.record_states=False
    step.state=state;step.performance_mode=performance_mode
    return step,reset,dict(torch=torch.__version__,compile=compiled,layerwise_causal_retiming=layerwise,requested_torch_threads=1)


def jax_setup(case,engine):
    import jax,jax.numpy as jnp,optax
    from jax_adapter import make_loss
    if engine=='brainx_state':
        import brainstate as bs
        bs.environ.set(precision=64)
    else:jax.config.update('jax_enable_x64',True)
    original=tuple(jnp.asarray(w,dtype=jnp.float64).reshape(a,b) for w,a,b in zip(case['weights'],case['sizes'][:-1],case['sizes'][1:]))
    x=jnp.asarray(case['inputs'],dtype=jnp.float64);y=jnp.asarray(case['labels'],dtype=jnp.int32)
    initial=jnp.zeros((len(case['labels']),sum(case['sizes'][1:])),dtype=jnp.float64)
    derivative,jit,metadata=make_loss(case,engine);tx=optax.adam(.001,b1=.9,b2=.999,eps=1e-8)
    params=original;opt_state=tx.init(params)
    diagnostic_fn=jit(derivative)
    def full_step(p,o):
        (loss,_),(grad,_)=derivative(p,initial,x,y)
        updates,no=tx.update(grad,o,p)
        return optax.apply_updates(p,updates),no,loss
    compiled=jit(full_step)
    def step(diagnostic=False):
        nonlocal params,opt_state
        if diagnostic:
            (loss,(logits,states,spikes)),(grad,ivjp)=diagnostic_fn(params,initial,x,y)
            updates,opt_state=tx.update(grad,opt_state,params);params=optax.apply_updates(params,updates)
            jax.block_until_ready((params,opt_state,loss))
            return dict(loss=np.asarray(loss),logits=np.asarray(logits),states=np.asarray(states),spikes=np.asarray(spikes),initial_vjp=np.asarray(ivjp),gradients=[np.asarray(g).ravel() for g in grad],weights=[np.asarray(p).ravel() for p in params],first=[np.asarray(v).ravel() for v in opt_state[0].mu],second=[np.asarray(v).ravel() for v in opt_state[0].nu])
        params,opt_state,loss=compiled(params,opt_state)
        jax.block_until_ready((params,opt_state,loss));return loss
    def reset():
        nonlocal params,opt_state
        params=original;opt_state=tx.init(params)
    def state():
        return dict(weights=[np.asarray(p).ravel() for p in params],first=[np.asarray(v).ravel() for v in opt_state[0].mu],second=[np.asarray(v).ravel() for v in opt_state[0].nu])
    step.state=state
    metadata.update(jax=jax.__version__,optax=optax.__version__,jit_full_train_step=True,device=str(jax.devices()))
    return step,reset,metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--compile',action='store_true');p.add_argument('--layerwise',action='store_true');args=p.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    report=dict(engine=args.engine,seed=args.seed,performance_run=False,status='unqualified',measured_seconds=[],warmup_steps=10,measured_steps=50,
        timing='complete Python training function including forward/loss/backward/Adam and required synchronization',cpu_resource='exclusive benchmark phase on same 20-core host; requested torch/OMP threads=1, JAX worker pool disclosed separately; no CPU affinity claim')
    start=time.perf_counter()
    try:
        f=ROOT/f'fixtures/e1-small/seed-{args.seed}.json';manifest=json.loads((f.parent/'manifest.json').read_text())
        assert hashlib.sha256(f.read_bytes()).hexdigest()==manifest['files'][f.name]['sha256']
        case=json.loads(f.read_text());report['arrays_sha256']=manifest['files'][f.name]['sha256']
        ref=forward_vjp(case)
        cold=time.perf_counter()
        if args.engine in ('spyx','brainx_state'):step,reset,meta=jax_setup(case,args.engine)
        else:step,reset,meta=torch_setup(case,args.engine,args.compile,args.layerwise)
        # The first invocation is the actual timing path, with a fresh per-job
        # compiler cache supplied by the supervisor. Diagnose only afterwards.
        if hasattr(step,'performance_mode'):step.performance_mode()
        actual_loss=step();report['cold_construction_and_first_update_s']=time.perf_counter()-cold
        actual_loss=float(actual_loss)
        actual_state={k:[np.array(z,copy=True) for z in arr] for k,arr in step.state().items()}
        actual_updates=[dict(loss=actual_loss,**actual_state)]
        for _ in range(2):
            loss=float(step());state={k:[np.array(z,copy=True) for z in arr] for k,arr in step.state().items()}
            actual_updates.append(dict(loss=loss,**state))
        reset();raw=step(True);report['implementation']=meta
        checks={k:check(raw[k],ref[k],k=='spikes') for k in ['loss','logits','spikes','states','initial_vjp']}
        w=[np.asarray(z) for z in case['weights']];zero=[np.zeros_like(z) for z in w]
        ww,mm,vv=adam(w,ref['gradients'],zero,zero,1)
        for name,expected in [('gradients',ref['gradients']),('weights',ww),('first',mm),('second',vv)]:
            for i,(a,b) in enumerate(zip(raw[name],expected)):checks[f'{name}_{i}']=check(a,b)
        checks['actual_public_loss']=check(actual_loss,ref['loss'])
        for name,expected in [('weights',ww),('first',mm),('second',vv)]:
            for i,(a,b) in enumerate(zip(actual_state[name],expected)):checks[f'actual_public_{name}_{i}']=check(a,b)
        for count,actual in enumerate(actual_updates[1:],2):
            next_ref=forward_vjp(case,ww)
            ww,mm,vv=adam(ww,next_ref['gradients'],mm,vv,count)
            checks[f'actual_public_step{count}_loss']=check(actual['loss'],next_ref['loss'])
            for name,expected in [('weights',ww),('first',mm),('second',vv)]:
                for i,(a,b) in enumerate(zip(actual[name],expected)):checks[f'actual_public_step{count}_{name}_{i}']=check(a,b)
        np.savez_compressed(dest/'actual-public-updates.npz',**{f'step{n+1}_{k}_{i}':z for n,update in enumerate(actual_updates) for k in ['weights','first','second'] for i,z in enumerate(update[k])},loss=np.asarray([u['loss'] for u in actual_updates]))
        report['checks']=checks
        np.savez_compressed(dest/'first-step.npz',**{k:v for k,v in raw.items() if k not in ['weights','first','second','gradients']},**{f'{k}_{i}':z for k in ['weights','first','second','gradients'] for i,z in enumerate(raw[k])})
        if not all(c['passed'] for c in checks.values()):report['status']='semantic_mismatch';return
        report['qualification_status']='passed_E1_dense';report['performance_run']=True;reset()
        if hasattr(step,'performance_mode'):step.performance_mode()
        with (dest/'raw-steps.jsonl').open('x') as log:
            for i in range(60):
                a=time.perf_counter();loss=step();elapsed=time.perf_counter()-a
                scalar=float(loss)
                log.write(json.dumps(dict(index=i,phase='warmup' if i<10 else 'measured',elapsed_s=elapsed,loss=scalar))+'\n');log.flush()
                if not np.isfinite(scalar):report['status']='numerical_divergence';return
                if i>=10:report['measured_seconds'].append(elapsed)
        report['status']='completed';report['median_s']=statistics.median(report['measured_seconds'])
    except Exception as e:
        message=str(e).lower()
        status='oom' if isinstance(e,MemoryError) or 'out of memory' in message else ('numerical_divergence' if 'nonfinite' in message else ('dependency_error' if isinstance(e,(ImportError,ModuleNotFoundError)) else 'software_rejected'))
        report.update(status=status,error=str(e),error_type=type(e).__name__,traceback=traceback.format_exc())
    finally:
        report['wall_s']=time.perf_counter()-start
        report['identities']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'adapters/torch_adapter.py',ROOT/'adapters/jax_adapter.py',ROOT/'adapters/oracle.py']}
        (dest/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'engine':args.engine,'seed':args.seed,'status':report['status'],'wall_s':report['wall_s']}))


if __name__=='__main__':main()
