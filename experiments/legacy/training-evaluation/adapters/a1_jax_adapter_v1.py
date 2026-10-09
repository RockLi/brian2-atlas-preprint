"""A1 JAX phase v1: genuine Spyx LIF or Brainstate stateful transformations.

Canonical dense synchronous FP64 LIF; no oracle import and no plain-JAX
substitute for a selected framework. All actual train/eval calls synchronize.
Lazy per-(B,T) graph bundles handle the formal 32/24/8/16 batch shapes.
"""
from pathlib import Path
import time

class A1Jax:
    checkpoint_suffix = '.npz'

    def __init__(self, root, weights, seed, engine):
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        if engine not in ('spyx', 'brainx_state'):
            raise ValueError(engine)
        if engine == 'brainx_state':
            import brainstate as bs
            bs.environ.set(precision=64)
        jax.config.update('jax_enable_x64', True)
        if not bool(jax.config.jax_enable_x64) or any(d.platform != 'cpu' for d in jax.devices()):
            raise RuntimeError('A1 JAX phase requires CPU FP64; no alternate device')
        self.jax, self.jnp, self.np = jax, jnp, np
        self.engine, self.seed, self.root = engine, seed, Path(root)
        self.original = tuple(jnp.asarray(w, dtype=jnp.float64).reshape(a,b)
                              for w,a,b in zip(weights,[784,128],[128,10]))
        self.optimizer = optax.adam(.001, b1=.9, b2=.999, eps=1e-8,
                                    eps_root=0., mu_dtype=jnp.float64)
        self.bundles = {}
        self.reset()
        jax.block_until_ready((self.params,self.opt_state))

    def reset(self):
        self.params = self.original
        self.opt_state = self.optimizer.init(self.params)

    def _bundle(self, batch, ticks):
        key = (batch,ticks)
        if key in self.bundles:
            return self.bundles[key]
        jax,jnp = self.jax,self.jnp
        import optax
        sizes=(784,128,10);beta=.95;theta=1.
        if self.engine == 'spyx':
            from flax import nnx
            from spyx import nn as spy_nn, axn
            activation=axn.superspike(k=5)
            cells=[spy_nn.LIF((n,),beta=1.,threshold=theta,activation=activation,rngs=nnx.Rngs(0))
                   for n in sizes[1:]]
            def forward(weights,initial,x):
                def tick(v,xt):
                    ss,post,margins=[],[],[]
                    for cell,vv in zip(cells,v):
                        drift=beta*vv
                        s,raw=cell(jnp.zeros_like(vv),drift)
                        # Forward-zero correction removes only reset-spike VJP.
                        post.append(raw+theta*(s-jax.lax.stop_gradient(s)))
                        ss.append(s);margins.append(drift-theta)
                    nxt=tuple(z+(xt if k==0 else ss[k-1])@weights[k] for k,z in enumerate(post))
                    return nxt,(jnp.concatenate(nxt,axis=1),jnp.concatenate(ss,axis=1),jnp.concatenate(margins,axis=1))
                _,values=jax.lax.scan(tick,tuple(jnp.split(initial,(128,),axis=1)),jnp.swapaxes(x,0,1))
                return tuple(jnp.swapaxes(v,0,1) for v in values)
            transform=jax.value_and_grad
            compile_=jax.jit
            identity=dict(neuron='spyx.nn.LIF',surrogate='spyx.axn.superspike(k=5)',
                          scan='jax.lax.scan',gradient='jax.value_and_grad',jit='jax.jit',
                          adaptation='drift before genuine LIF(beta=1), forward-zero detached-reset correction')
        else:
            import brainstate as bs
            from jax_adapter import canonical_spike
            voltage=bs.HiddenState(jnp.zeros((batch,138),dtype=jnp.float64))
            def forward(weights,initial,x):
                voltage.value=initial
                def tick(carry,xt):
                    u=tuple(beta*vv for vv in jnp.split(voltage.value,(128,),axis=1))
                    margins=tuple(vv-theta for vv in u)
                    ss=tuple(canonical_spike(m) for m in margins)
                    nxt=tuple(z-theta*jax.lax.stop_gradient(ss[k])+
                              (xt if k==0 else ss[k-1])@weights[k] for k,z in enumerate(u))
                    voltage.value=jnp.concatenate(nxt,axis=1)
                    return carry,(voltage.value,jnp.concatenate(ss,axis=1),jnp.concatenate(margins,axis=1))
                _,values=bs.transform.scan(tick,None,jnp.swapaxes(x,0,1))
                return tuple(jnp.swapaxes(v,0,1) for v in values)
            compile_=bs.transform.jit
            identity=dict(neuron='custom canonical cell in brainstate.HiddenState',
                          surrogate='frozen jax_adapter.canonical_spike custom VJP',
                          scan='brainstate.transform.scan',gradient='brainstate.transform.grad',
                          jit='brainstate.transform.jit',
                          adaptation='actual stateful transforms; no claim to a BrainPy neuron implementation')
        def loss(weights,initial,x,labels):
            states,spikes,margins=forward(weights,initial,x)
            logits=5*jnp.mean(spikes[:,:,-10:],axis=1)
            value=jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits,labels))
            return value,(logits,states,spikes,margins)
        if self.engine == 'brainx_state':
            transformed=bs.transform.grad(loss,argnums=(0,1),has_aux=True,return_value=True)
            def derivative(*args):
                (gradient,initial_gradient),value,aux=transformed(*args)
                return (value,aux),(gradient,initial_gradient)
        else:
            derivative=transform(loss,argnums=(0,1),has_aux=True)
        def train(params,state,initial,x,labels):
            (value,aux),(gradient,initial_gradient)=derivative(params,initial,x,labels)
            updates,new_state=self.optimizer.update(gradient,state,params)
            new_params=optax.apply_updates(params,updates)
            logits,_,spikes,margins=aux
            # Identical output signature in ordinary training and qualification.
            return new_params,new_state,value,logits,gradient,initial_gradient,spikes[:,0,:],margins[:,0,:]
        def evaluate(params,initial,x,labels):
            # Pure forward: no derivative transform and no optimizer update.
            value,(logits,_,spikes,margins)=loss(params,initial,x,labels)
            return value,logits,spikes[:,0,:],margins[:,0,:]
        bundle=dict(train=compile_(train),evaluate=compile_(evaluate),identity=identity)
        self.bundles[key]=bundle
        return bundle

    def batch(self,x,y,training,diagnostic=False,explicit_initial=None):
        jax,jnp,np=self.jax,self.jnp,self.np
        before=time.perf_counter()
        tx=jnp.asarray(x,dtype=jnp.float64);ty=jnp.asarray(y,dtype=jnp.int32)
        initial=(jnp.zeros((len(y),138),dtype=jnp.float64) if explicit_initial is None
                 else jnp.asarray(explicit_initial,dtype=jnp.float64))
        if tuple(initial.shape)!=(len(y),138) or tuple(tx.shape)!=(len(y),tx.shape[1],784):
            raise ValueError('A1 full-width batch/initial shape differs')
        bundle=self._bundle(len(y),int(tx.shape[1]))
        if training:
            raw=bundle['train'](self.params,self.opt_state,initial,tx,ty)
            jax.block_until_ready(raw)
            self.params,self.opt_state,loss,logits,gradient,initial_gradient,spikes,margins=raw
        else:
            raw=bundle['evaluate'](self.params,initial,tx,ty)
            jax.block_until_ready(raw)
            loss,logits,spikes,margins=raw
        plain=lambda value:np.array(jax.device_get(value),copy=True)
        result=dict(loss=float(loss),logits=plain(logits),public_api_s=time.perf_counter()-before)
        if not np.isfinite(result['loss']) or not np.all(np.isfinite(result['logits'])):
            raise FloatingPointError('nonfinite actual JAX A1 batch')
        if explicit_initial is not None:
            result.update(first_tick_spikes=plain(spikes),first_tick_margins=plain(margins))
        if diagnostic:
            if not training:raise ValueError('training diagnostics require training=True')
            result.update(self.state_arrays())
            result.update(gradients=[plain(w).reshape(-1) for w in gradient],initial_vjp=plain(initial_gradient))
        return result

    def counter(self):
        return int(self.jax.device_get(self.opt_state[0].count))

    def state_arrays(self):
        plain=lambda x:self.np.array(self.jax.device_get(x),copy=True).reshape(-1)
        return dict(weights=[plain(w) for w in self.params],
                    first=[plain(w) for w in self.opt_state[0].mu],
                    second=[plain(w) for w in self.opt_state[0].nu])

    def implementation(self):
        return dict(engine=self.engine,device=[str(d) for d in self.jax.devices()],
                    jax_enable_x64=bool(self.jax.config.jax_enable_x64),
                    parameter_dtype=str(self.params[0].dtype),
                    resource_qualification=False,timing_class='JAX host-config diagnostics; no strict1thread ranking',
                    bundles=[dict(B=b,T=t,**v['identity']) for (b,t),v in sorted(self.bundles.items())])

    def save(self,path):
        state=self.state_arrays()
        arrays={f'{name}_{i}':value for name,banks in state.items() for i,value in enumerate(banks)}
        arrays.update(step=self.np.asarray(self.counter(),dtype=self.np.int64),
                      engine=self.np.asarray(self.engine),schema=self.np.asarray('a1-jax-checkpoint-v1'))
        with Path(path).open('xb') as stream:self.np.savez_compressed(stream,**arrays)

    def restore(self,path):
        np,jnp=self.np,self.jnp
        with np.load(path,allow_pickle=False) as data:
            if data['schema'].item()!='a1-jax-checkpoint-v1' or data['engine'].item()!=self.engine:
                raise ValueError('Checkpoint schema/engine differs')
            shape=((784,128),(128,10))
            banks={}
            for name in ['weights','first','second']:
                source=[data[f'{name}_{i}'].copy() for i in range(2)]
                if any(v.dtype!=np.dtype('float64') or v.size!=a*b or not np.all(np.isfinite(v))
                       for v,(a,b) in zip(source,shape)):
                    raise ValueError('Checkpoint precision/shape/finite state differs')
                banks[name]=tuple(jnp.asarray(v).reshape(s) for v,s in zip(source,shape))
            step=int(data['step'])
            if step<0 or step>17190:raise ValueError('Checkpoint Adam counter outside A1 bounds')
        self.params=banks['weights']
        state=self.opt_state[0]._replace(count=jnp.asarray(step,dtype=self.opt_state[0].count.dtype),
                                         mu=banks['first'],nu=banks['second'])
        self.opt_state=(state,*self.opt_state[1:])
        self.jax.block_until_ready((self.params,self.opt_state))
