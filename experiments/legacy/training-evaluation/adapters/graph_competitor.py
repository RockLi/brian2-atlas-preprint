"""Canonical synchronous graph adapters using real competitor library APIs.

Parameter banks are one-dimensional: indexed gather and scatter preserve tied
edge identities and let each framework accumulate their VJP. All cells cross
their threshold before any projection runs, including recurrent projections.
The independent NumPy oracle is deliberately not imported by this module.
"""
import importlib.metadata
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _graph(case):
    sizes = case['sizes']
    offsets = [0]
    for n in sizes[1:]:
        offsets.append(offsets[-1] + n)
    if len(case['projections']) != len(case['weights']):
        raise ValueError('Every projection must have its own explicit parameter bank')
    edges = []
    for p, bank in zip(case['projections'], case['weights']):
        source_layer, target_layer = p['source_layer'], p['target_layer']
        if not 0 <= source_layer < len(sizes) or not 1 <= target_layer < len(sizes):
            raise ValueError('Graph layer outside declared sizes')
        source, target, parameter = p['sources'], p['targets'], p['parameter_ids']
        if len(source) != len(target) or len(source) != len(parameter):
            raise ValueError('Projection edge array lengths differ')
        if len(bank) != p['parameter_count']:
            raise ValueError('Parameter bank length differs from parameter_count')
        if any(not 0 <= i < sizes[source_layer] for i in source):
            raise ValueError('Source edge outside layer')
        if any(not 0 <= i < sizes[target_layer] for i in target):
            raise ValueError('Target edge outside layer')
        if any(not 0 <= i < len(bank) for i in parameter):
            raise ValueError('Parameter index outside bank')
        edges.append((source_layer == 0,
                      [i + (0 if source_layer == 0 else offsets[source_layer - 1]) for i in source],
                      [i + offsets[target_layer - 1] for i in target], parameter))
    def per_layer(value):
        return list(value) if isinstance(value, (list, tuple)) else [value] * (len(sizes) - 1)
    beta = per_layer(case.get('beta', .95))
    theta = per_layer(case.get('threshold', case.get('theta', 1.)))
    if len(beta) != len(sizes) - 1 or len(theta) != len(sizes) - 1:
        raise ValueError('One beta and threshold value is required per neuron layer')
    return sizes, offsets, edges, beta, theta


def run_torch(case, engine, *, steps=3, compiled=False):
    import torch
    from torch import nn
    # Import only the actual cell's surrogate adapter, never reference math.
    from torch_adapter import StrictFastSigmoid, SJSurrogate

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    sizes, offsets, edges, beta, theta = _graph(case)

    class Graph(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.ParameterList([nn.Parameter(torch.tensor(w, dtype=torch.float64)) for w in case['weights']])
            cells = []
            if engine == 'snntorch_fp64':
                import snntorch as snn
                for b, th in zip(beta, theta):
                    cells.append(snn.Leaky(beta=torch.tensor(b, dtype=torch.float64),
                                           threshold=torch.tensor(th, dtype=torch.float64),
                                           reset_mechanism='none', spike_grad=StrictFastSigmoid.apply).double())
            elif engine == 'spikingjelly':
                from spikingjelly.activation_based import neuron
                for b, th in zip(beta, theta):
                    cells.append(neuron.LIFNode(tau=1 / (1 - b), decay_input=False,
                                               v_threshold=th, v_reset=None,
                                               surrogate_function=SJSurrogate(), detach_reset=True,
                                               step_mode='s', backend='torch').double())
            else:
                raise ValueError(engine)
            self.cells = nn.ModuleList(cells)
            for q, (_, src, dst, ids) in enumerate(edges):
                for name, values in [('src', src), ('dst', dst), ('ids', ids)]:
                    self.register_buffer(f'{name}_{q}', torch.tensor(values, dtype=torch.long))

        def forward(self, inputs, initial):
            v = initial
            all_states, all_spikes = [], []
            for t in range(inputs.shape[1]):
                spike_parts, post_parts = [], []
                for k, (cell, vv) in enumerate(zip(self.cells, torch.split(v, sizes[1:], dim=1))):
                    if engine == 'snntorch_fp64':
                        s, u = cell(torch.zeros_like(vv), vv)
                        post = u - theta[k] * s.detach()
                    else:
                        cell.v = vv
                        s = cell(torch.zeros_like(vv))
                        post = cell.v
                    spike_parts.append(s)
                    post_parts.append(post)
                spikes = torch.cat(spike_parts, dim=1)
                v = torch.cat(post_parts, dim=1)
                # Source spikes are all from this tick's threshold stage.
                for q, (external, _, _, _) in enumerate(edges):
                    src, dst, ids = (getattr(self, f'{key}_{q}') for key in ('src', 'dst', 'ids'))
                    feed = (inputs[:, t] if external else spikes)[:, src]
                    contribution = feed * self.weights[q][ids]
                    v = v.scatter_add(1, dst.unsqueeze(0).expand(inputs.shape[0], -1), contribution)
                all_states.append(v)
                all_spikes.append(spikes)
            states, spikes = torch.stack(all_states, dim=1), torch.stack(all_spikes, dim=1)
            return 5 * spikes[:, :, -sizes[-1]:].mean(dim=1), states, spikes

    model = Graph()
    forward = torch.compile(model, fullgraph=True) if compiled else model
    inputs = torch.tensor(case['inputs'], dtype=torch.float64)
    labels = torch.tensor(case['labels'], dtype=torch.long)
    initial = torch.tensor(case.get('initial', [[0.] * offsets[-1] for _ in case['labels']]),
                           dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam(model.weights, lr=.001, betas=(.9, .999), eps=1e-8, foreach=False)
    plain = lambda a: a.detach().cpu().numpy().tolist()

    def differentiate():
        optimizer.zero_grad(set_to_none=True)
        initial.grad = None
        logits, states, spikes = forward(inputs, initial)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        return dict(loss=float(loss.detach()), logits=plain(logits), states=plain(states), spikes=plain(spikes),
                    gradients=[plain(w.grad) for w in model.weights], initial_vjp=plain(initial.grad))

    before = time.perf_counter()
    out = differentiate()
    out.update(engine=engine, version=importlib.metadata.version('snntorch' if engine == 'snntorch_fp64' else 'spikingjelly'),
               torch=torch.__version__, qualification_cold_s=time.perf_counter() - before,
               identity=dict(neuron='snntorch.Leaky' if engine == 'snntorch_fp64' else 'spikingjelly.activation_based.neuron.LIFNode',
                             scheduler='all threshold/reset stages followed by projection gather/scatter; recurrent, no layerwise retiming',
                             parameter_sharing='1D parameter bank indexed gather with autograd accumulation',
                             compile_forward=compiled, compile_fullgraph=compiled, optimizer='native torch.optim.Adam, FP64 moments',
                             surrogate='strict FP64 fast sigmoid, slope=5, detached reset'), updates=[])
    sgd_params = [nn.Parameter(w.detach().clone()) for w in model.weights]
    for w, source in zip(sgd_params, model.weights):
        w.grad = source.grad.detach().clone()
    torch.optim.SGD(sgd_params, lr=.001).step()
    out['one_sgd_update'] = [plain(w) for w in sgd_params]
    margins = torch.tensor([-2., -.5, 0., .5, 2.], dtype=torch.float64, requires_grad=True)
    cotangents = torch.tensor([.3, -.2, .5, .7, -1.], dtype=torch.float64)
    activation = model.cells[0].spike_grad if engine == 'snntorch_fp64' else model.cells[0].surrogate_function
    values = activation(margins)
    vjp, = torch.autograd.grad(values, margins, cotangents)
    out['local_surrogate_vjp'] = dict(margins=plain(margins), cotangents=plain(cotangents), values=plain(values), vjp=plain(vjp))
    for _ in range(steps):
        before = time.perf_counter()
        update = differentiate()
        optimizer.step()
        update.update(elapsed_s=time.perf_counter() - before, weights=[plain(w) for w in model.weights],
                      first=[plain(optimizer.state[w]['exp_avg']) for w in model.weights],
                      second=[plain(optimizer.state[w]['exp_avg_sq']) for w in model.weights],
                      step=[int(optimizer.state[w]['step']) for w in model.weights])
        out['updates'].append(update)
    out['timing_role'] = 'qualification diagnostics only, includes conversion and first compilation; not performance evidence'
    return out


def run_jax(case, engine, *, steps=3):
    os.environ.setdefault('BRAINEVENT_CACHE_DIR', str(ROOT / 'cache/brainevent'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'cache'))
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(ROOT / 'cache/jax'))
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import jax
    import jax.numpy as jnp
    import optax
    if engine == 'brainx_state':
        import brainstate as bs
        bs.environ.set(precision=64)
    else:
        jax.config.update('jax_enable_x64', True)
    from jax_adapter import canonical_spike
    sizes, offsets, edges, beta, theta = _graph(case)
    edges = [(external, jnp.asarray(src, dtype=jnp.int32), jnp.asarray(dst, dtype=jnp.int32),
              jnp.asarray(ids, dtype=jnp.int32)) for external, src, dst, ids in edges]
    splits = tuple(offsets[1:-1])

    def project(post, spikes, xt, weights):
        for (external, src, dst, ids), bank in zip(edges, weights):
            feed = (xt if external else spikes)[:, src]
            post = post.at[:, dst].add(feed * bank[ids])
        return post

    if engine == 'spyx':
        from flax import nnx
        from spyx import nn as spy_nn, axn
        activation = axn.superspike(k=5)
        cells = [spy_nn.LIF((n,), beta=1., threshold=th, activation=activation, rngs=nnx.Rngs(0))
                 for n, th in zip(sizes[1:], theta)]

        def forward(weights, initial, inputs):
            def tick(v, xt):
                ss, post = [], []
                for k, (cell, vv) in enumerate(zip(cells, jnp.split(v, splits, axis=1))):
                    s, raw = cell(jnp.zeros_like(vv), beta[k] * vv)
                    ss.append(s)
                    post.append(raw + theta[k] * (s - jax.lax.stop_gradient(s)))
                spikes = jnp.concatenate(ss, axis=1)
                v = project(jnp.concatenate(post, axis=1), spikes, xt, weights)
                return v, (v, spikes)
            _, result = jax.lax.scan(tick, initial, jnp.swapaxes(inputs, 0, 1))
            return tuple(jnp.swapaxes(value, 0, 1) for value in result)

        identity = dict(neuron='spyx.nn.LIF', surrogate='spyx.axn.superspike(k=5)',
                        scheduler='jax.lax.scan; explicit drift before LIF(beta=1); detached reset correction')
        compile_ = jax.jit
    elif engine == 'brainx_state':
        voltage = bs.HiddenState(jnp.zeros((len(case['labels']), offsets[-1]), dtype=jnp.float64))
        expanded_beta = jnp.asarray([b for b, n in zip(beta, sizes[1:]) for _ in range(n)], dtype=jnp.float64)
        expanded_theta = jnp.asarray([th for th, n in zip(theta, sizes[1:]) for _ in range(n)], dtype=jnp.float64)
        activation = canonical_spike

        def forward(weights, initial, inputs):
            voltage.value = initial
            def tick(carry, xt):
                u = expanded_beta * voltage.value
                spikes = canonical_spike(u - expanded_theta)
                post = u - expanded_theta * jax.lax.stop_gradient(spikes)
                voltage.value = project(post, spikes, xt, weights)
                return carry, (voltage.value, spikes)
            _, result = bs.transform.scan(tick, None, jnp.swapaxes(inputs, 0, 1))
            return tuple(jnp.swapaxes(value, 0, 1) for value in result)

        identity = dict(neuron='canonical equations in brainstate.HiddenState', surrogate='jax.custom_vjp strict fast sigmoid',
                        scheduler='brainstate.transform.scan', gradient='brainstate.transform.grad', jit='brainstate.transform.jit',
                        limitation='Does not qualify a BrainPy default neuron package')
        compile_ = bs.transform.jit
    else:
        raise ValueError(engine)

    def loss(weights, initial, inputs, labels):
        states, spikes = forward(weights, initial, inputs)
        logits = 5 * spikes[:, :, -sizes[-1]:].mean(axis=1)
        value = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return value, (logits, states, spikes)

    if engine == 'brainx_state':
        grad = bs.transform.grad(loss, argnums=(0, 1), has_aux=True, return_value=True)
        def derivative(*args):
            gradients, value, aux = grad(*args)
            return (value, aux), gradients
    else:
        derivative = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
    inputs = jnp.asarray(case['inputs'], dtype=jnp.float64)
    labels = jnp.asarray(case['labels'], dtype=jnp.int32)
    weights = tuple(jnp.asarray(w, dtype=jnp.float64) for w in case['weights'])
    initial = jnp.asarray(case.get('initial', [[0.] * offsets[-1] for _ in case['labels']]), dtype=jnp.float64)
    plain = lambda value: jax.device_get(value).tolist()

    def unpack(result):
        (value, (logits, states, spikes)), (gradients, ivjp) = result
        return dict(loss=float(value), logits=plain(logits), states=plain(states), spikes=plain(spikes),
                    gradients=[plain(g) for g in gradients], initial_vjp=plain(ivjp))

    before = time.perf_counter()
    result = compile_(derivative)(weights, initial, inputs, labels)
    jax.block_until_ready(result)
    out = unpack(result)
    identity.update(parameter_sharing='1D bank indexed gather with autodiff accumulation',
                    projection='all cells threshold first; indexed scatter-add, same-tick recurrence, no layerwise retiming',
                    optimizer='Optax Adam, FP64 parameters/moments, eps_root=0, full update JIT')
    out.update(engine=engine, version=importlib.metadata.version('spyx' if engine == 'spyx' else 'brainstate'),
               jax=jax.__version__, optax=optax.__version__, jax_enable_x64=bool(jax.config.jax_enable_x64),
               parameter_dtype=str(weights[0].dtype), device=str(jax.devices()), identity=identity,
               qualification_cold_s=time.perf_counter() - before, updates=[])
    gradients = result[1][0]
    sgd = optax.sgd(.001)
    delta, _ = sgd.update(gradients, sgd.init(weights), weights)
    out['one_sgd_update'] = [plain(w) for w in optax.apply_updates(weights, delta)]
    margins = jnp.asarray([-2., -.5, 0., .5, 2.], dtype=jnp.float64)
    cotangents = jnp.asarray([.3, -.2, .5, .7, -1.], dtype=jnp.float64)
    values, pullback = jax.vjp(activation, margins)
    out['local_surrogate_vjp'] = dict(margins=plain(margins), cotangents=plain(cotangents), values=plain(values), vjp=plain(pullback(cotangents)[0]))
    optimizer = optax.adam(.001, b1=.9, b2=.999, eps=1e-8, eps_root=0, mu_dtype=jnp.float64)
    opt_state = optimizer.init(weights)

    def full_update(params, state, init, x, y):
        raw = derivative(params, init, x, y)
        delta, state = optimizer.update(raw[1][0], state, params)
        return optax.apply_updates(params, delta), state, raw
    update = compile_(full_update)
    for _ in range(steps):
        before = time.perf_counter()
        weights, opt_state, result = update(weights, opt_state, initial, inputs, labels)
        jax.block_until_ready((weights, opt_state, result))
        record = unpack(result)
        state = opt_state[0]
        record.update(elapsed_s=time.perf_counter() - before, weights=[plain(w) for w in weights],
                      first=[plain(m) for m in state.mu], second=[plain(v) for v in state.nu], step=int(state.count))
        out['updates'].append(record)
    out['timing_role'] = 'qualification diagnostics only, includes conversion and first compilation; not performance evidence'
    return out


def run_case(case, engine, *, steps=3, compiled=False):
    if engine in ('spyx', 'brainx_state'):
        return run_jax(case, engine, steps=steps)
    return run_torch(case, engine, steps=steps, compiled=compiled)
