"""Independent FP64 synchronous LIF forward and explicitly derived surrogate VJP.

No Atlas, Torch or JAX import. Axes are B,T,I and source-major W[I,O].
Reverse equations follow the contract, not finite differences of hard spikes.
"""
import numpy as np


def forward_vjp(case, weights=None):
    x = np.asarray(case['inputs'], dtype=np.float64)
    sizes = case['sizes']; B,T,_ = x.shape
    weights = [np.asarray(w,dtype=np.float64).reshape(a,b) for w,a,b in zip(
        case['weights'] if weights is None else weights, sizes[:-1], sizes[1:])]
    beta=case.get('beta',.95); theta=case.get('theta',1.)
    split=np.cumsum(sizes[1:])[:-1]
    initial=np.asarray(case.get('initial', np.zeros((B,sum(sizes[1:])))),dtype=np.float64)
    v=[z.copy() for z in np.split(initial,split,axis=1)]
    vs=[]; us=[]; ss=[]
    for t in range(T):
        u=[beta*z for z in v]; s=[(z>theta).astype(np.float64) for z in u]
        v=[z-theta*sp+(x[:,t] if k==0 else s[k-1])@weights[k]
           for k,(z,sp) in enumerate(zip(u,s))]
        vs.append([z.copy() for z in v]);us.append(u);ss.append(s)
    logits=5*np.mean([s[-1] for s in ss],axis=0)
    y=np.asarray(case['labels'],dtype=np.int64)
    z=logits-logits.max(axis=1,keepdims=True)
    p=np.exp(z);p/=p.sum(axis=1,keepdims=True)
    loss=np.mean(-z[np.arange(B),y]+np.log(np.exp(z).sum(axis=1)))
    logit_bar=p.copy();logit_bar[np.arange(B),y]-=1;logit_bar/=B
    carry=[np.zeros_like(vv) for vv in v]
    grads=[np.zeros_like(w) for w in weights]
    for t in range(T-1,-1,-1):
        spike_bar=[np.zeros_like(vv) for vv in v]
        spike_bar[-1]+=5/T*logit_bar
        for k,w in enumerate(weights):
            feed=x[:,t] if k==0 else ss[t][k-1]
            grads[k]+=feed.T@carry[k]
            if k:spike_bar[k-1]+=carry[k]@w.T
        carry=[beta*(c+ds/(1+5*np.abs(u-theta))**2)
               for c,ds,u in zip(carry,spike_bar,us[t])]
    return dict(loss=float(loss),logits=logits,states=np.stack([np.concatenate(v,axis=1) for v in vs],axis=1),
        spikes=np.stack([np.concatenate(s,axis=1) for s in ss],axis=1),
        gradients=[g.ravel() for g in grads],initial_vjp=np.concatenate(carry,axis=1))


def adam(weights, gradients, m, v, step, lr=.001):
    out=[]; mm=[]; vv=[]
    for w,g,a,b in zip(weights,gradients,m,v):
        a=.9*a+.1*g;b=.999*b+.001*g*g
        out.append(w-lr*(a/(1-.9**step))/(np.sqrt(b/(1-.999**step))+1e-8))
        mm.append(a);vv.append(b)
    return out,mm,vv


def fixture(seed=11,sizes=(2,4,2),B=2,T=32):
    rng=np.random.default_rng(seed)
    x=rng.choice([-.25,0.,1.,2.],size=(B,T,sizes[0]),p=[.08,.4,.44,.08])
    weights=[rng.uniform(-.2,1.2,size=(a,b))/np.sqrt(a/2) for a,b in zip(sizes[:-1],sizes[1:])]
    return dict(sizes=list(sizes),inputs=x.tolist(),labels=(np.arange(B)%sizes[-1]).tolist(),weights=[w.ravel().tolist() for w in weights],beta=.95,theta=1.,seed=seed)


if __name__=='__main__':
    from pathlib import Path
    import json,hashlib
    root=Path(__file__).resolve().parents[1]
    case=fixture();path=root/'fixtures/q0.json';path.write_text(json.dumps(case,sort_keys=True,separators=(',',':'))+'\n')
    np.savez(root/'fixtures/q0.npz',x=np.asarray(case['inputs']).transpose(1,0,2),w1=np.asarray(case['weights'][0]).reshape(2,4),w2=np.asarray(case['weights'][1]).reshape(4,2),beta=.95,theta=1.)
    out=forward_vjp(case)
    np.savez(root/'fixtures/q0-oracle.npz',**{k:v for k,v in out.items() if k!='gradients'},gradient0=out['gradients'][0],gradient1=out['gradients'][1])
    (root/'fixtures/manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'fixtures').glob('*')) if p.name!='manifest.json'},indent=2)+'\n')
    print({'loss':out['loss'],'spikes':int(out['spikes'].sum()),'gradient_max':[float(np.abs(g).max()) for g in out['gradients']]})
