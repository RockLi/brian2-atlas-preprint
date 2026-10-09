"""Native matrix projections for fixed, unique-weight synchronous R graphs.

This version is independent of the frozen indexed-scatter adapter. Only the
cell/surrogate and diagnostic conventions are retained. No oracle is imported.
Matrices are constructed once per forward, outside the time loop. Sparse COO
or BCOO is an explicit native view; unsupported operators are never silently
replaced. Qualification diagnostics are not a performance implementation.
"""
import importlib.metadata
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def topology(case, storage='auto'):
    """Validate without B*T*E arrays; return sizes, offsets and layout metadata.

    Sorted (target, source) coordinates and unique parameter IDs are required.
    An O(E)-byte bitmap checks the permutation; no Python set of E tuples is
    constructed. Existing frozen R fixtures meet these stricter requirements.
    """
    if storage not in ('auto', 'dense'):
        raise ValueError('storage must be preregistered as auto or dense')
    allowed = {'id', 'seed', 'sizes', 'inputs', 'labels', 'weights', 'projections',
               'beta', 'theta', 'threshold', 'initial', 'workload_ids'}
    if set(case) - allowed:
        raise ValueError('Unsupported graph fields: '+str(sorted(set(case)-allowed)))
    sizes = case['sizes']
    if len(sizes) < 2 or any(not isinstance(n, int) or n < 1 for n in sizes):
        raise ValueError('Positive integer layer sizes required')
    offsets = [0]
    for n in sizes[1:]:
        offsets.append(offsets[-1]+n)
    if len(case['weights']) != len(case['projections']):
        raise ValueError('Exactly one bank per projection is required')
    plans = []
    fields = {'source_layer', 'target_layer', 'sources', 'targets', 'parameter_ids', 'parameter_count'}
    for q, (p, bank) in enumerate(zip(case['projections'], case['weights'])):
        if set(p) != fields:
            raise ValueError('Projection must contain only fixed, nondelayed edge fields')
        sl, tl = p['source_layer'], p['target_layer']
        if not isinstance(sl, int) or not isinstance(tl, int) or not 0 <= sl < len(sizes) or not 1 <= tl < len(sizes):
            raise ValueError('Layer index outside graph')
        ns, nt = sizes[sl], sizes[tl]
        src, dst, ids = p['sources'], p['targets'], p['parameter_ids']
        count = len(src)
        if count != len(dst) or count != len(ids) or count != len(bank) or count != p['parameter_count']:
            raise ValueError('Only one distinct trainable parameter per edge is supported')
        seen = bytearray(count)
        previous = -1
        source_major = target_major = True
        for index, (s, d, k) in enumerate(zip(src, dst, ids)):
            if not all(isinstance(v, int) for v in (s, d, k)) or not 0 <= s < ns or not 0 <= d < nt or not 0 <= k < count:
                raise ValueError('Invalid edge coordinate or bank index')
            coordinate = d*ns+s
            if coordinate <= previous:
                raise ValueError('Edges must be unique and sorted by (target, source)')
            previous = coordinate
            if seen[k]:
                raise ValueError('Tied parameters are outside matrix-v1 scope')
            seen[k] = 1
            source_major = source_major and k == s*nt+d
            target_major = target_major and k == index
        del seen
        complete = count == ns*nt
        layout = ('source_major' if source_major else 'target_major' if target_major else 'permuted') if complete else 'indexed'
        kind = 'dense' if complete or storage == 'dense' else 'sparse'
        plans.append(dict(source_layer=sl, target_layer=tl, source_count=ns, target_count=nt,
                          edge_count=count, complete=complete, layout=layout, kind=kind))
    def per_layer(value):
        return list(value) if isinstance(value, (list, tuple)) else [value]*(len(sizes)-1)
    beta = per_layer(case.get('beta', .95))
    theta = per_layer(case.get('threshold', case.get('theta', 1.)))
    if len(beta) != len(sizes)-1 or len(theta) != len(sizes)-1:
        raise ValueError('Dynamics array length differs from layer count')
    if any(not math.isfinite(b) or not 0 < b < 1 for b in beta) or any(not math.isfinite(v) or v <= 0 for v in theta):
        raise ValueError('Matrix-v1 requires fixed 0<beta<1 and positive finite threshold')
    if not case['labels'] or len(case['inputs']) != len(case['labels']):
        raise ValueError('Nonempty input batch and matching labels required')
    ticks = len(case['inputs'][0])
    if ticks < 1 or any(len(sample) != ticks or any(len(row) != sizes[0] for row in sample) for sample in case['inputs']):
        raise ValueError('Input shape must be [B,T,input]')
    if any(not isinstance(label, int) or not 0 <= label < sizes[-1] for label in case['labels']):
        raise ValueError('Invalid class label')
    if 'initial' in case and (len(case['initial']) != len(case['labels']) or any(len(row) != offsets[-1] for row in case['initial'])):
        raise ValueError('Initial membrane must be [B,sum(neuron sizes)]')
    return sizes, offsets, plans, beta, theta


def _phase(callback, name):
    if callback is not None:
        callback(name)


def run_torch(case, engine, *, steps=3, compiled=False, storage="auto", phase=None):
    _phase(phase, "engine_imports_and_setup")
    import torch
    from torch import nn
    # Import only the actual cell's surrogate adapter, never reference math.
    from torch_adapter import StrictFastSigmoid, SJSurrogate

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    sizes, offsets, plans, beta, theta = topology(case, storage)

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
            for q, (plan, projection) in enumerate(zip(plans, case['projections'])):
                # No index tensors are needed for a reshape/transpose dense bank.
                if plan['layout'] not in ('source_major', 'target_major'):
                    self.register_buffer(f'ids_{q}', torch.tensor(projection['parameter_ids'], dtype=torch.long))
                if not plan['complete']:
                    self.register_buffer(f'coordinates_{q}', torch.tensor([projection['targets'], projection['sources']], dtype=torch.long))

        def matrices(self):
            result = []
            for q, (bank, plan) in enumerate(zip(self.weights, plans)):
                ns, nt = plan['source_count'], plan['target_count']
                if plan['layout'] == 'source_major':
                    matrix = bank.reshape(ns, nt).T
                elif plan['layout'] == 'target_major':
                    matrix = bank.reshape(nt, ns)
                elif plan['complete']:
                    matrix = bank[getattr(self, f'ids_{q}')].reshape(nt, ns)
                else:
                    values = bank[getattr(self, f'ids_{q}')]
                    coordinates = getattr(self, f'coordinates_{q}')
                    if plan['kind'] == 'sparse':
                        matrix = torch.sparse_coo_tensor(coordinates, values, (nt, ns), is_coalesced=True)
                    else:
                        # Explicit dense-storage view: absent edges stay fixed zero;
                        # they never become optimizer parameters. One scatter/call.
                        flat = coordinates[0]*ns+coordinates[1]
                        matrix = bank.new_zeros(nt*ns).scatter(0, flat, values).reshape(nt, ns)
                result.append(matrix)
            return result

        def forward(self, inputs, initial):
            matrices = self.matrices()
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
                # Every threshold completes before any projection updates a layer.
                # Source is the whole layer, not a per-edge B x E gather.
                for plan, matrix in zip(plans, matrices):
                    sl, tl = plan['source_layer'], plan['target_layer']
                    feed = inputs[:, t] if sl == 0 else spike_parts[sl-1]
                    contribution = (torch.sparse.mm(matrix, feed.T).T if plan['kind'] == 'sparse'
                                    else nn.functional.linear(feed, matrix))
                    post_parts[tl-1] = post_parts[tl-1] + contribution
                v = torch.cat(post_parts, dim=1)
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
        _phase(phase, "engine_forward_backward")
        logits, states, spikes = forward(inputs, initial)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        _phase(phase, "diagnostic_conversion")
        return dict(loss=float(loss.detach()), logits=plain(logits), states=plain(states), spikes=plain(spikes),
                    gradients=[plain(w.grad) for w in model.weights], initial_vjp=plain(initial.grad))

    before = time.perf_counter()
    out = differentiate()
    out.update(engine=engine, version=importlib.metadata.version('snntorch' if engine == 'snntorch_fp64' else 'spikingjelly'),
               torch=torch.__version__, qualification_cold_s=time.perf_counter() - before,
               identity=dict(neuron='snntorch.Leaky' if engine == 'snntorch_fp64' else 'spikingjelly.activation_based.neuron.LIFNode',
                             scheduler='all threshold/reset stages, then native matrix projections; recurrent, no layerwise retiming',
                             parameters='one bank scalar per edge; no tied weights; fixed topology',
                             storage_policy=storage, matrix_plans=plans,
                             projection='torch.nn.functional.linear / torch.sparse.mm COO',
                             construction='once per forward outside time loop; no explicit B x E feed tensor per tick',
                             compile_forward=compiled, compile_fullgraph=compiled, optimizer='native torch.optim.Adam, FP64 moments',
                             surrogate='strict FP64 fast sigmoid, slope=5, detached reset'), updates=[])
    sgd_params = [nn.Parameter(w.detach().clone()) for w in model.weights]
    for w, source in zip(sgd_params, model.weights):
        w.grad = source.grad.detach().clone()
    _phase(phase, "engine_sgd_and_local_surrogate")
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
        _phase(phase, "engine_optimizer_step")
        optimizer.step()
        _phase(phase, "diagnostic_conversion")
        update.update(elapsed_s=time.perf_counter() - before, weights=[plain(w) for w in model.weights],
                      first=[plain(optimizer.state[w]['exp_avg']) for w in model.weights],
                      second=[plain(optimizer.state[w]['exp_avg_sq']) for w in model.weights],
                      step=[int(optimizer.state[w]['step']) for w in model.weights])
        out['updates'].append(update)
    out['timing_role'] = 'qualification diagnostics only, includes conversion and first compilation; not performance evidence'
    return out


def run_jax(case, engine, *, steps=3, storage="auto", phase=None):
    _phase(phase, "engine_imports_and_setup")
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
    sizes, offsets, plans, beta, theta = topology(case, storage)
    from jax.experimental.sparse import BCOO
    indices = []
    for plan, projection in zip(plans, case['projections']):
        ids = None if plan['layout'] in ('source_major', 'target_major') else jnp.asarray(projection['parameter_ids'], dtype=jnp.int32)
        coords = None if plan['complete'] else jnp.stack((jnp.asarray(projection['targets'], dtype=jnp.int32),
                                                        jnp.asarray(projection['sources'], dtype=jnp.int32)), axis=1)
        indices.append((ids, coords))
    splits = tuple(offsets[1:-1])

    def matrices(weights):
        result = []
        for bank, plan, (ids, coordinates) in zip(weights, plans, indices):
            ns, nt = plan['source_count'], plan['target_count']
            if plan['layout'] == 'source_major':
                matrix = bank.reshape(ns, nt).T
            elif plan['layout'] == 'target_major':
                matrix = bank.reshape(nt, ns)
            elif plan['complete']:
                matrix = bank[ids].reshape(nt, ns)
            elif plan['kind'] == 'sparse':
                matrix = BCOO((bank[ids], coordinates), shape=(nt, ns), indices_sorted=True, unique_indices=True)
            else:
                matrix = jnp.zeros((nt, ns), dtype=bank.dtype).at[coordinates[:, 0], coordinates[:, 1]].set(bank[ids])
            result.append(matrix)
        return result

    def project(post, spike_parts, xt, projections):
        for plan, matrix in zip(plans, projections):
            sl, tl = plan['source_layer'], plan['target_layer']
            feed = xt if sl == 0 else spike_parts[sl-1]
            # BCOO binds the native sparse-dot primitive; no explicit edgewise
            # feed*bank operations live in this adapter's temporal scan.
            post[tl-1] = post[tl-1] + (matrix @ feed.T).T
        return jnp.concatenate(post, axis=1)

    if engine == 'spyx':
        from flax import nnx
        from spyx import nn as spy_nn, axn
        activation = axn.superspike(k=5)
        cells = [spy_nn.LIF((n,), beta=1., threshold=th, activation=activation, rngs=nnx.Rngs(0))
                 for n, th in zip(sizes[1:], theta)]

        def forward(weights, initial, inputs):
            projections = matrices(weights)
            def tick(v, xt):
                ss, post = [], []
                for k, (cell, vv) in enumerate(zip(cells, jnp.split(v, splits, axis=1))):
                    s, raw = cell(jnp.zeros_like(vv), beta[k] * vv)
                    ss.append(s)
                    post.append(raw + theta[k] * (s - jax.lax.stop_gradient(s)))
                spikes = jnp.concatenate(ss, axis=1)
                v = project(post, ss, xt, projections)
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
            projections = matrices(weights)
            voltage.value = initial
            def tick(carry, xt):
                u = expanded_beta * voltage.value
                spikes = canonical_spike(u - expanded_theta)
                post = u - expanded_theta * jax.lax.stop_gradient(spikes)
                voltage.value = project(list(jnp.split(post, splits, axis=1)),
                                        jnp.split(spikes, splits, axis=1), xt, projections)
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
        _phase(phase, "diagnostic_conversion")
        (value, (logits, states, spikes)), (gradients, ivjp) = result
        return dict(loss=float(value), logits=plain(logits), states=plain(states), spikes=plain(spikes),
                    gradients=[plain(g) for g in gradients], initial_vjp=plain(ivjp))

    before = time.perf_counter()
    _phase(phase, "engine_compile_forward_backward")
    result = compile_(derivative)(weights, initial, inputs, labels)
    jax.block_until_ready(result)
    out = unpack(result)
    identity.update(parameters='one bank scalar per edge; no tied weights; fixed topology',
                    projection='native dense matmul / experimental JAX BCOO sparse matmul',
                    storage_policy=storage, matrix_plans=plans,
                    construction='once per loss before scan; no explicit B x E feed tensor per tick',
                    sparse_limit='BCOO is an experimental reference implementation, not a strongest-performance claim',
                    optimizer='Optax Adam, FP64 parameters/moments, eps_root=0, full update JIT')
    out.update(engine=engine, version=importlib.metadata.version('spyx' if engine == 'spyx' else 'brainstate'),
               jax=jax.__version__, optax=optax.__version__, jax_enable_x64=bool(jax.config.jax_enable_x64),
               parameter_dtype=str(weights[0].dtype), device=str(jax.devices()), identity=identity,
               qualification_cold_s=time.perf_counter() - before, updates=[])
    gradients = result[1][0]
    _phase(phase, "engine_sgd_and_local_surrogate")
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
        _phase(phase, "engine_compile_full_adam_step")
        weights, opt_state, result = update(weights, opt_state, initial, inputs, labels)
        jax.block_until_ready((weights, opt_state, result))
        record = unpack(result)
        state = opt_state[0]
        record.update(elapsed_s=time.perf_counter() - before, weights=[plain(w) for w in weights],
                      first=[plain(m) for m in state.mu], second=[plain(v) for v in state.nu], step=int(state.count))
        out['updates'].append(record)
    out['timing_role'] = 'qualification diagnostics only, includes conversion and first compilation; not performance evidence'
    return out


def run_case(case, engine, *, steps=3, compiled=False, storage="auto", phase=None):
    if engine in ('spyx', 'brainx_state'):
        return run_jax(case, engine, steps=steps, storage=storage, phase=phase)
    return run_torch(case, engine, steps=steps, compiled=compiled, storage=storage, phase=phase)
