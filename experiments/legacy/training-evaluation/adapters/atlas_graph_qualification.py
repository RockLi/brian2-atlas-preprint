"""Independent NumPy VJP oracle for same-tick recurrent and tied-weight graphs.

No Atlas math implementation is imported. Atlas is invoked only by qualify().
No finite differences of hard spikes are used. Full common arrays are persisted
in each result, including projection edge lists and bank parameter identities.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def graph_fixture(case, *, shared=False):
    case=copy.deepcopy(case)
    sizes=case['sizes']
    projections=[]
    for layer,(source_count,target_count) in enumerate(zip(sizes[:-1],sizes[1:])):
        projections.append(dict(source_layer=layer,target_layer=layer+1,
            parameter_count=source_count*target_count,
            sources=[i for j in range(target_count) for i in range(source_count)],
            targets=[j for j in range(target_count) for i in range(source_count)],
            parameter_ids=[i*target_count+j for j in range(target_count) for i in range(source_count)]))
    n=sizes[1]
    projections.append(dict(source_layer=1,target_layer=1,parameter_count=n*n,
        sources=[i for j in range(n) for i in range(n)],
        targets=[j for j in range(n) for i in range(n)],
        parameter_ids=[i*n+j for j in range(n) for i in range(n)]))
    case['weights'].append([.04*((i*3+j*2)%7-2) for i in range(n) for j in range(n)])
    if shared:
        # One parameter per presynaptic neuron, shared over all target sites.
        p=projections[-1];p['parameter_count']=n
        p['parameter_ids']=p['sources'].copy()
        case['weights'][-1]=[.04*(i-1) for i in range(n)]
    case['projections']=projections
    case['id']='Q0-recurrent-shared' if shared else 'Q0-recurrent'
    return case


def forward_vjp(case, weights=None):
    x=np.asarray(case['inputs'],dtype=np.float64);B,T,_=x.shape
    sizes=case['sizes'];offsets=np.r_[0,np.cumsum(sizes[1:])];N=int(offsets[-1]);C=sizes[-1]
    w=[np.asarray(a,dtype=np.float64) for a in (case['weights'] if weights is None else weights)]
    beta=case.get('beta',.95);theta=case.get('threshold',case.get('theta',1.))
    beta=np.repeat(np.broadcast_to(beta,(len(sizes)-1,)),sizes[1:])
    theta=np.repeat(np.broadcast_to(theta,(len(sizes)-1,)),sizes[1:])
    v=np.array(case.get('initial',np.zeros((B,N))),dtype=np.float64)
    history=[];states=[];spikes=[]
    for t in range(T):
        u=beta*v;s=(u>theta).astype(np.float64)
        v=u-theta*s
        for p,bank in zip(case['projections'],w):
            # The oracle follows the graph equations, not native target loops.
            source=np.asarray(p['sources']);target=np.asarray(p['targets'])+offsets[p['target_layer']-1]
            feed=x[:,t,source] if p['source_layer']==0 else s[:,source+offsets[p['source_layer']-1]]
            values=feed*bank[np.asarray(p['parameter_ids'])]
            for b in range(B):np.add.at(v[b],target,values[b])
        history.append(u);states.append(v.copy());spikes.append(s)
    logits=5*np.mean(spikes,axis=0)[:,-C:]
    shift=logits-logits.max(1,keepdims=True);exp=np.exp(shift);p=exp/exp.sum(1,keepdims=True)
    labels=np.asarray(case['labels']);loss=np.mean(np.log(exp.sum(1))-shift[np.arange(B),labels])
    dlogits=p.copy();dlogits[np.arange(B),labels]-=1;dlogits/=B
    carry=np.zeros((B,N));grads=[np.zeros_like(bank) for bank in w]
    for t in range(T-1,-1,-1):
        ds=np.zeros((B,N));ds[:,-C:]=5/T*dlogits
        for q,(projection,bank) in enumerate(zip(case['projections'],w)):
            source=np.asarray(projection['sources']);target=np.asarray(projection['targets'])+offsets[projection['target_layer']-1]
            ids=np.asarray(projection['parameter_ids'])
            feed=x[:,t,source] if projection['source_layer']==0 else spikes[t][:,source+offsets[projection['source_layer']-1]]
            np.add.at(grads[q],ids,np.sum(feed*carry[:,target],axis=0))
            if projection['source_layer']:
                for b in range(B):np.add.at(ds[b],source+offsets[projection['source_layer']-1],carry[b,target]*bank[ids])
        carry=beta*(carry+ds/(1+5*np.abs(history[t]-theta))**2)
    return dict(loss=float(loss),logits=logits,states=np.stack(states,axis=1),spikes=np.stack(spikes,axis=1),gradients=grads,initial_vjp=carry)


def comparison(actual,expected,*,exact=False):
    a=np.asarray(actual);b=np.asarray(expected);diff=np.abs(a-b)
    return dict(passed=bool(np.array_equal(a,b) if exact else np.all(diff<=1e-10+1e-8*np.abs(b))),
                max_absolute_error=float(np.max(diff,initial=0)),shape=list(b.shape))


def qualify(case,source_root,runner):
    from atlas_adapter import run_case
    actual=run_case(case,source_root,runner,steps=3,collect_states=True)
    report=dict(case=case,execution=actual,status=actual['status'])
    if actual['status']!='executed':return report
    ref=forward_vjp(case);g=actual['gradients']['result']
    checks={k:comparison(g[k],ref[k],exact=k=='spikes') for k in ('loss','logits','spikes')}
    checks['states']=comparison(actual['state_trace']['states'],ref['states'])
    checks['initial_vjp']=comparison(g['initial_gradients'],ref['initial_vjp'])
    for j,gradient in enumerate(ref['gradients']):checks[f'gradient_{j}']=comparison(g['gradients'][j],gradient)
    w=[np.asarray(bank,dtype=np.float64) for bank in case['weights']];m=[np.zeros_like(bank) for bank in w];v=copy.deepcopy(m)
    for step,update in enumerate(actual['updates'],1):
        ref=forward_vjp(case,weights=w)
        for j,gradient in enumerate(ref['gradients']):
            m[j]=.9*m[j]+.1*gradient;v[j]=.999*v[j]+.001*gradient*gradient
            w[j]=w[j]-.001*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
            s=update['result']['state']
            for key,expect in [('weights',w),('first_moment',m),('second_moment',v)]:
                checks[f'adam_{step}_{key}_{j}']=comparison(s[key][j],expect[j])
        checks[f'adam_{step}_step']={'passed':update['result']['state']['step']==step}
    report['checks']=checks;report['status']='qualified' if all(c['passed'] for c in checks.values()) else 'unqualified'
    report['qualification_scope']='same-tick graph, CE surrogate VJP, recurrent/tied gradients and three native Adam steps; not arbitrary-cotangent VJP, delay or trainable dynamics'
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path);args=parser.parse_args();base=args.base.resolve()
    path=args.output or base/'evidence/atlas-graph-q0.json'
    if path.exists():raise FileExistsError('preserve old evidence; choose a new run filename')
    case=json.loads((base/'fixtures/q0.json').read_text())
    report=dict(schema='atlas-graph-qualification-v1',oracle_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                results=[qualify(graph_fixture(case,shared=shared),base/'snapshot/brian2-rust',base/'runtime/b2-train') for shared in (False,True)])
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(case=r['case']['id'],status=r['status'],failed=[k for k,v in r.get('checks',{}).items() if not v['passed']]) for r in report['results']]))


if __name__=='__main__':main()
