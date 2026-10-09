"""True conv2d tied-parameter competitor cells, with legal acyclic retiming.

Torch F.conv2d and JAX lax.conv_general_dilated operate on shared OIHW banks;
the graph oracle and edge scatter are never imported by this implementation.
"""
import importlib.metadata
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def torch_case(case, engine, *, compiled=False, steps=3):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch_adapter import StrictFastSigmoid, SJSurrogate
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    beta, theta = case['beta'], case['theta']
    sizes = case['sizes']

    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.ParameterList([nn.Parameter(torch.tensor(w, dtype=torch.float64)) for w in case['weights']])
            cells = []
            if engine == 'snntorch_fp64':
                import snntorch as snn
                for _ in sizes[1:]:
                    cells.append(snn.Leaky(beta=torch.tensor(beta, dtype=torch.float64), threshold=torch.tensor(theta, dtype=torch.float64),
                                           reset_mechanism='none', spike_grad=StrictFastSigmoid.apply).double())
            elif engine == 'spikingjelly':
                from spikingjelly.activation_based import neuron
                for _ in sizes[1:]:
                    cells.append(neuron.LIFNode(tau=1/(1-beta), decay_input=False, v_threshold=theta, v_reset=None,
                                               surrogate_function=SJSurrogate(), detach_reset=True, step_mode='s', backend='torch').double())
            else:
                raise ValueError(engine)
            self.cells = nn.ModuleList(cells)

        def forward(self, inputs, initial):
            feed = inputs
            batch, ticks = inputs.shape[:2]
            all_states, all_spikes = [], []
            for cell, w, shape, op, initial_layer in zip(self.cells, self.weights, case['parameter_shapes'], case['operators'], torch.split(initial, sizes[1:], dim=1)):
                if op['type'] == 'conv2d':
                    images = feed.reshape(batch*ticks, *op['input_chw'])
                    current = F.conv2d(images, w.reshape(shape), bias=None, stride=op['stride'], padding=op['padding']).reshape(batch, ticks, -1)
                else:
                    current = feed@w.reshape(shape)
                v = initial_layer
                spike_sequence, state_sequence = [], []
                for t in range(ticks):
                    charge = torch.zeros_like(v) if t == 0 else beta*current[:, t-1]
                    if engine == 'snntorch_fp64':
                        spikes, u = cell(charge, v)
                        v = u-theta*spikes.detach()
                    else:
                        cell.v = v
                        spikes = cell(charge)
                        v = cell.v
                    spike_sequence.append(spikes)
                    state_sequence.append(v+current[:, t])
                feed = torch.stack(spike_sequence, dim=1)
                all_spikes.append(feed)
                all_states.append(torch.stack(state_sequence, dim=1))
            return 5*feed.mean(dim=1), torch.cat(all_states, dim=2), torch.cat(all_spikes, dim=2)

    model = ConvModel()
    forward = torch.compile(model, fullgraph=True) if compiled else model
    x = torch.tensor(case['inputs'], dtype=torch.float64)
    labels = torch.tensor(case['labels'], dtype=torch.long)
    initial = torch.tensor(case.get('initial', [[0.]*sum(sizes[1:]) for _ in case['labels']]), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam(model.weights, lr=.001, betas=(.9, .999), eps=1e-8, foreach=False)
    plain = lambda value: value.detach().cpu().numpy().tolist()

    def derivative():
        optimizer.zero_grad(set_to_none=True)
        initial.grad = None
        logits, states, spikes = forward(x, initial)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        return dict(loss=float(loss.detach()), logits=plain(logits), states=plain(states), spikes=plain(spikes),
                    gradients=[plain(w.grad) for w in model.weights], initial_vjp=plain(initial.grad))

    before = time.perf_counter()
    out = derivative()
    out.update(engine=engine, version=importlib.metadata.version('snntorch' if engine == 'snntorch_fp64' else 'spikingjelly'),
               torch=torch.__version__, compile=compiled, qualification_cold_s=time.perf_counter()-before,
               identity=dict(projection='torch.nn.functional.conv2d, OIHW shared banks, NCHW',
                             neuron='snntorch.Leaky' if engine == 'snntorch_fp64' else 'spikingjelly.activation_based.neuron.LIFNode',
                             scheduling='legal feedforward layerwise retiming; batched B*T convolutions; no recurrence',
                             parameter_gradient='native conv2d autograd sums all spatial uses into each shared kernel coefficient'), updates=[])
    sgd_params = [nn.Parameter(w.detach().clone()) for w in model.weights]
    for dest, source in zip(sgd_params, model.weights):
        dest.grad = source.grad.detach().clone()
    torch.optim.SGD(sgd_params, lr=.001).step()
    out['one_sgd_update'] = [plain(w) for w in sgd_params]
    margin = torch.tensor([-2., -.5, 0., .5, 2.], dtype=torch.float64, requires_grad=True)
    cot = torch.tensor([.3, -.2, .5, .7, -1.], dtype=torch.float64)
    activation = model.cells[0].spike_grad if engine == 'snntorch_fp64' else model.cells[0].surrogate_function
    values = activation(margin)
    vjp, = torch.autograd.grad(values, margin, cot)
    out['local_surrogate_vjp'] = dict(margins=plain(margin), cotangents=plain(cot), values=plain(values), vjp=plain(vjp))
    for _ in range(steps):
        before = time.perf_counter()
        update = derivative()
        optimizer.step()
        update.update(elapsed_s=time.perf_counter()-before, weights=[plain(w) for w in model.weights],
                      first=[plain(optimizer.state[w]['exp_avg']) for w in model.weights],
                      second=[plain(optimizer.state[w]['exp_avg_sq']) for w in model.weights],
                      step=[int(optimizer.state[w]['step']) for w in model.weights])
        out['updates'].append(update)
    out['timing_role'] = 'qualification including full diagnostics and conversion; no performance score'
    return out


def jax_case(case, engine, *, steps=3):
    os.environ.setdefault('BRAINEVENT_CACHE_DIR', str(ROOT/'cache/brainevent'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'cache'))
    import jax
    import jax.numpy as jnp
    import optax
    if engine == 'brainx_state':
        import brainstate as bs
        bs.environ.set(precision=64)
    else:
        jax.config.update('jax_enable_x64', True)
    from jax_adapter import canonical_spike
    sizes, beta, theta = case['sizes'], case['beta'], case['theta']
    splits = tuple(sum(sizes[1:i]) for i in range(2, len(sizes)))
    if engine == 'spyx':
        from flax import nnx
        from spyx import nn as spy_nn, axn
        activation = axn.superspike(k=5)
        cells = [spy_nn.LIF((n,), beta=1., threshold=theta, activation=activation, rngs=nnx.Rngs(0)) for n in sizes[1:]]
        compile_ = jax.jit
    elif engine == 'brainx_state':
        activation = canonical_spike
        voltages = [bs.HiddenState(jnp.zeros((len(case['labels']), n), dtype=jnp.float64)) for n in sizes[1:]]
        compile_ = bs.transform.jit
    else:
        raise ValueError(engine)

    def loss(weights, initial, inputs, labels):
        feed = inputs
        batch, ticks = inputs.shape[:2]
        all_states, all_spikes = [], []
        for layer, (w, shape, op, initial_layer) in enumerate(zip(weights, case['parameter_shapes'], case['operators'], jnp.split(initial, splits, axis=1))):
            if op['type'] == 'conv2d':
                images = feed.reshape(batch*ticks, *op['input_chw'])
                current = jax.lax.conv_general_dilated(images, w.reshape(shape), window_strides=(op['stride'], op['stride']),
                                                       padding=((op['padding'], op['padding']), (op['padding'], op['padding'])),
                                                       dimension_numbers=('NCHW', 'OIHW', 'NCHW'), precision=jax.lax.Precision.HIGHEST).reshape(batch, ticks, -1)
            else:
                current = feed@w.reshape(shape)
            charge = jnp.concatenate((jnp.zeros_like(current[:, :1]), beta*current[:, :-1]), axis=1)
            if engine == 'spyx':
                cell = cells[layer]
                def tick(v, q):
                    spikes, post = cell(jnp.zeros_like(v), beta*v+q)
                    post = post+theta*(spikes-jax.lax.stop_gradient(spikes))
                    return post, (post, spikes)
                _, (post, spikes) = jax.lax.scan(tick, initial_layer, jnp.swapaxes(charge, 0, 1))
            else:
                voltage = voltages[layer]
                voltage.value = initial_layer
                def tick(carry, q):
                    u = beta*voltage.value+q
                    spikes = canonical_spike(u-theta)
                    voltage.value = u-theta*jax.lax.stop_gradient(spikes)
                    return carry, (voltage.value, spikes)
                _, (post, spikes) = bs.transform.scan(tick, None, jnp.swapaxes(charge, 0, 1))
            feed = jnp.swapaxes(spikes, 0, 1)
            all_spikes.append(feed)
            all_states.append(jnp.swapaxes(post, 0, 1)+current)
        logits = 5*feed.mean(axis=1)
        value = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return value, (logits, jnp.concatenate(all_states, axis=2), jnp.concatenate(all_spikes, axis=2))

    if engine == 'brainx_state':
        transformed = bs.transform.grad(loss, argnums=(0, 1), has_aux=True, return_value=True)
        def derivative(*args):
            gradients, value, aux = transformed(*args)
            return (value, aux), gradients
    else:
        derivative = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
    weights = tuple(jnp.asarray(w, dtype=jnp.float64) for w in case['weights'])
    initial = jnp.asarray(case.get('initial', [[0.]*sum(sizes[1:]) for _ in case['labels']]), dtype=jnp.float64)
    x, labels = jnp.asarray(case['inputs'], dtype=jnp.float64), jnp.asarray(case['labels'], dtype=jnp.int32)
    plain = lambda value: jax.device_get(value).tolist()

    def unpack(result):
        (value, (logits, states, spikes)), (gradients, ivjp) = result
        return dict(loss=float(value), logits=plain(logits), states=plain(states), spikes=plain(spikes),
                    gradients=[plain(g) for g in gradients], initial_vjp=plain(ivjp))

    before = time.perf_counter()
    result = compile_(derivative)(weights, initial, x, labels)
    jax.block_until_ready(result)
    out = unpack(result)
    out.update(engine=engine, version=importlib.metadata.version('spyx' if engine == 'spyx' else 'brainstate'), jax=jax.__version__,
               optax=optax.__version__, parameter_dtype=str(weights[0].dtype), device=str(jax.devices()),
               qualification_cold_s=time.perf_counter()-before,
               identity=dict(projection='jax.lax.conv_general_dilated NCHW/OIHW/NCHW', neuron='spyx.nn.LIF' if engine == 'spyx' else 'canonical cell in brainstate.HiddenState',
                             transforms='jax.jit/scan/value_and_grad' if engine == 'spyx' else 'brainstate.transform.jit/scan/grad',
                             scheduling='legal feedforward layerwise retiming, B*T batched native convolutions',
                             parameter_gradient='native convolution VJP accumulates tied spatial uses'), updates=[])
    gradients = result[1][0]
    sgd = optax.sgd(.001)
    delta, _ = sgd.update(gradients, sgd.init(weights), weights)
    out['one_sgd_update'] = [plain(w) for w in optax.apply_updates(weights, delta)]
    margins, cot = jnp.asarray([-2., -.5, 0., .5, 2.], dtype=jnp.float64), jnp.asarray([.3, -.2, .5, .7, -1.], dtype=jnp.float64)
    values, pullback = jax.vjp(activation, margins)
    out['local_surrogate_vjp'] = dict(margins=plain(margins), cotangents=plain(cot), values=plain(values), vjp=plain(pullback(cot)[0]))
    tx = optax.adam(.001, b1=.9, b2=.999, eps=1e-8, eps_root=0, mu_dtype=jnp.float64)
    state = tx.init(weights)
    def update(params, state):
        raw = derivative(params, initial, x, labels)
        updates, state = tx.update(raw[1][0], state, params)
        return optax.apply_updates(params, updates), state, raw
    update = compile_(update)
    for _ in range(steps):
        before = time.perf_counter()
        weights, state, raw = update(weights, state)
        jax.block_until_ready((weights, state, raw))
        row = unpack(raw)
        row.update(elapsed_s=time.perf_counter()-before, weights=[plain(w) for w in weights],
                   first=[plain(w) for w in state[0].mu], second=[plain(w) for w in state[0].nu], step=int(state[0].count))
        out['updates'].append(row)
    out['timing_role'] = 'qualification including full diagnostics and conversion; no performance score'
    return out


def run_case(case, engine, *, compiled=False, steps=3):
    if engine in ('spyx', 'brainx_state'):
        return jax_case(case, engine, steps=steps)
    return torch_case(case, engine, compiled=compiled, steps=steps)
