"""Actual library neuron cells, explicitly adapted to synchronous scheduling.

SpikingJelly accepts a user surrogate; ours freezes strict > and fast sigmoid
VJP. snnTorch uses its official fast_sigmoid. No NumPy oracle code is imported.
"""
import torch
from torch import nn


class StrictFastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return (x>0).to(x.dtype)
    @staticmethod
    def backward(ctx,grad):
        x,=ctx.saved_tensors
        return grad/(1+5*x.abs()).square()


class SJSurrogate(nn.Module):
    def forward(self,x):return StrictFastSigmoid.apply(x)


class Model(nn.Module):
    def __init__(self,case,engine='snntorch',layerwise=False):
        super().__init__();self.engine=engine;self.sizes=case['sizes'];self.record_states=True
        self.beta=case.get('beta',.95);self.theta=case.get('theta',1.);self.layerwise=layerwise
        self.weights=nn.ParameterList([nn.Parameter(torch.tensor(w,dtype=torch.float64).reshape(a,b)) for w,a,b in zip(case['weights'],self.sizes[:-1],self.sizes[1:])])
        cells=[]
        if engine.startswith('snntorch'):
            import snntorch as snn
            from snntorch import surrogate
            for _ in self.sizes[1:]:
                spike_grad=StrictFastSigmoid.apply if engine=='snntorch_fp64' else surrogate.fast_sigmoid(slope=5)
                cells.append(snn.Leaky(beta=torch.tensor(self.beta,dtype=torch.float64),threshold=torch.tensor(self.theta,dtype=torch.float64),reset_mechanism='none',spike_grad=spike_grad).double())
        elif engine.startswith('spikingjelly'):
            from spikingjelly.activation_based import neuron
            for _ in self.sizes[1:]:
                cells.append(neuron.LIFNode(tau=1/(1-self.beta),decay_input=False,v_threshold=self.theta,v_reset=None,surrogate_function=SJSurrogate(),detach_reset=True,step_mode='s',backend='torch').double())
        else:raise ValueError(engine)
        self.cells=nn.ModuleList(cells)

    def forward(self,x,initial):
        if self.layerwise:return self.forward_layerwise(x,initial)
        v=list(torch.split(initial,self.sizes[1:],dim=1));spikes=[];states=[]
        for t in range(x.shape[1]):
            s=[];post=[]
            for cell,vv in zip(self.cells,v):
                if self.engine.startswith('snntorch'):
                    ss,u=cell(torch.zeros_like(vv),vv)
                    # snnTorch's official surrogate returns FP32 even for FP64
                    # input. Retain this fact in the native qualification trial.
                    ss=ss.to(vv.dtype);pp=u-self.theta*ss.detach()
                else:
                    cell.v=vv;ss=cell(torch.zeros_like(vv));pp=cell.v
                s.append(ss);post.append(pp)
            v=[vv+(x[:,t] if k==0 else s[k-1])@w for k,(vv,w) in enumerate(zip(post,self.weights))]
            spikes.append(torch.cat(s,dim=1) if self.record_states else s[-1])
            if self.record_states:states.append(torch.cat(v,dim=1))
        spikes=torch.stack(spikes,dim=1);states=torch.stack(states,dim=1) if self.record_states else None
        return 5*spikes[:,:,-self.sizes[-1]:].mean(dim=1),states,spikes if self.record_states else None

    def forward_layerwise(self,x,initial):
        # Exact causal retiming: input at canonical tick t changes threshold at
        # t+1. Precompute the projection across B,T and scan each actual cell.
        # This is only legal for this acyclic feedforward fixture.
        initials=torch.split(initial,self.sizes[1:],dim=1);feed=x
        all_spikes=[];all_states=[]
        for cell,w,v in zip(self.cells,self.weights,initials):
            current=feed@w;ss=[];vv=[]
            for t in range(x.shape[1]):
                charge=torch.zeros_like(v) if t==0 else self.beta*current[:,t-1]
                if self.engine.startswith('snntorch'):
                    s,u=cell(charge,v);s=s.to(v.dtype);v=u-self.theta*s.detach()
                else:
                    cell.v=v;s=cell(charge);v=cell.v
                ss.append(s)
                if self.record_states:vv.append(v+current[:,t])
            feed=torch.stack(ss,dim=1)
            if self.record_states:
                all_spikes.append(feed);all_states.append(torch.stack(vv,dim=1))
        spikes=torch.cat(all_spikes,dim=2) if self.record_states else None;states=torch.cat(all_states,dim=2) if self.record_states else None
        return 5*feed.mean(dim=1),states,spikes


def run_case(case,engine='snntorch',steps=3,compile_model=False,layerwise=False):
    import time,importlib.metadata
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    model=Model(case,engine,layerwise=layerwise)
    x=torch.tensor(case['inputs'],dtype=torch.float64);labels=torch.tensor(case['labels'],dtype=torch.long)
    init=torch.tensor(case.get('initial',[[0.]*sum(case['sizes'][1:]) for _ in case['labels']]),dtype=torch.float64,requires_grad=True)
    run=torch.compile(model,fullgraph=True) if compile_model else model
    optimizer=torch.optim.Adam(model.weights,lr=.001,betas=(.9,.999),eps=1e-8,foreach=False)
    logits,states,spikes=run(x,init);loss=nn.functional.cross_entropy(logits,labels);loss.backward()
    plain=lambda z:z.detach().cpu().numpy().tolist()
    out=dict(engine=engine,version=importlib.metadata.version('snntorch' if engine.startswith('snntorch') else 'spikingjelly'),torch=torch.__version__,compile=compile_model,
             loss=float(loss.detach()),logits=plain(logits),states=plain(states),spikes=plain(spikes),gradients=[plain(w.grad.flatten()) for w in model.weights],initial_vjp=plain(init.grad),updates=[])
    sgd_params=[nn.Parameter(w.detach().clone()) for w in model.weights]
    for w,source in zip(sgd_params,model.weights):w.grad=source.grad.detach().clone()
    sgd=torch.optim.SGD(sgd_params,lr=.001);sgd.step()
    out['one_sgd_update']=[plain(w.flatten()) for w in sgd_params]
    margins=torch.tensor([-2.,-.5,0.,.5,2.],requires_grad=True)
    cotangents=torch.tensor([.3,-.2,.5,.7,-1.])
    if engine.startswith('snntorch'):values=model.cells[0].spike_grad(margins)
    else:values=model.cells[0].surrogate_function(margins)
    vjp=torch.autograd.grad(values,margins,cotangents)[0]
    out['local_surrogate_vjp']=dict(margins=plain(margins),cotangents=plain(cotangents),values=plain(values),vjp=plain(vjp))
    for _ in range(steps):
        start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        logits,_,_=run(x,init);loss=nn.functional.cross_entropy(logits,labels);loss.backward();optimizer.step()
        elapsed=time.perf_counter()-start
        out['updates'].append(dict(elapsed_s=elapsed,weights=[plain(w.flatten()) for w in model.weights],first=[plain(optimizer.state[w]['exp_avg'].flatten()) for w in model.weights],second=[plain(optimizer.state[w]['exp_avg_sq'].flatten()) for w in model.weights],step=[float(optimizer.state[w]['step']) for w in model.weights]))
    return out
