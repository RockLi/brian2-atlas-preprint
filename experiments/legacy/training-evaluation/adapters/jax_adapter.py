"""Q0 adapters using real Spyx LIF and brainstate stateful transforms.

This is the adapted-canonical view, not a claim about default neuron semantics.
No oracle import. All time loops and value/gradient/optimizer steps are JIT-able.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('BRAINEVENT_CACHE_DIR', str(ROOT / 'cache/brainevent'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'cache'))
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(ROOT / 'cache/jax'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax


@jax.custom_vjp
def canonical_spike(x):
    return (x > 0).astype(x.dtype)


def spike_fwd(x):
    return canonical_spike(x), x


def spike_bwd(x, cotangent):
    return (cotangent / (1 + 5 * jnp.abs(x)) ** 2,)


canonical_spike.defvjp(spike_fwd, spike_bwd)


def make_loss(case, engine):
    sizes = case['sizes']
    splits = tuple(sum(sizes[1:i]) for i in range(2, len(sizes)))
    beta, theta = case.get('beta', .95), case.get('theta', 1.)
    if engine == 'spyx':
        from flax import nnx
        from spyx import nn as spy_nn, axn
        activation = axn.superspike(k=5)
        cells = [spy_nn.LIF((n,), beta=1., threshold=theta, activation=activation, rngs=nnx.Rngs(0)) for n in sizes[1:]]

        def forward(weights, initial, x):
            def tick(v, xt):
                ss, post = [], []
                for cell, vv in zip(cells, v):
                    # Spyx normally tests V before beta*V; feed it the drifted
                    # voltage with beta=1 to match the declared threshold stage.
                    s, raw = cell(jnp.zeros_like(vv), beta * vv)
                    # Forward-zero correction removes only reset's spike VJP.
                    post.append(raw + theta * (s - jax.lax.stop_gradient(s)))
                    ss.append(s)
                vv = tuple(z + (xt if k == 0 else ss[k - 1]) @ weights[k] for k, z in enumerate(post))
                return vv, (jnp.concatenate(vv, axis=1), jnp.concatenate(ss, axis=1))
            _, (states, spikes) = jax.lax.scan(tick, tuple(jnp.split(initial, splits, axis=1)), jnp.swapaxes(x, 0, 1))
            return jnp.swapaxes(states, 0, 1), jnp.swapaxes(spikes, 0, 1)

        metadata = {'neuron': 'spyx.nn.LIF', 'surrogate': 'spyx.axn.superspike(k=5)', 'scan': 'jax.lax.scan',
                    'adaptation': 'Explicit drift before actual LIF(beta=1) cell; zero-forward gradient correction detaches reset; synchronous layers.'}
    elif engine == 'brainx_state':
        import brainstate as bs
        # Brainstate is an equation/state transformation framework; this fixture
        # intentionally supplies its own canonical cell and uses actual state,
        # scan, autodiff and compilation APIs. No BrainPy default cell is implied.
        voltage = bs.HiddenState(jnp.zeros((len(case['labels']), sum(sizes[1:])), dtype=jnp.float64))

        def forward(weights, initial, x):
            voltage.value = initial
            def tick(carry, xt):
                u = tuple(beta * vv for vv in jnp.split(voltage.value, splits, axis=1))
                ss = tuple(canonical_spike(vv - theta) for vv in u)
                vv = tuple(z - theta * jax.lax.stop_gradient(ss[k]) + (xt if k == 0 else ss[k - 1]) @ weights[k] for k, z in enumerate(u))
                voltage.value = jnp.concatenate(vv, axis=1)
                return carry, (voltage.value, jnp.concatenate(ss, axis=1))
            _, (states, spikes) = bs.transform.scan(tick, None, jnp.swapaxes(x, 0, 1))
            return jnp.swapaxes(states, 0, 1), jnp.swapaxes(spikes, 0, 1)
        metadata = {'neuron': 'custom canonical cell in brainstate.HiddenState', 'surrogate': 'independent adapter jax.custom_vjp',
                    'scan': 'brainstate.transform.scan', 'gradient': 'brainstate.transform.grad', 'jit': 'brainstate.transform.jit',
                    'adaptation': 'Native stateful transforms; brainpy.state neuron package is not installed/qualified by this result.'}
    else:
        raise ValueError(engine)

    def loss(weights, initial, x, labels):
        states, spikes = forward(weights, initial, x)
        logits = 5 * jnp.mean(spikes[:, :, -sizes[-1]:], axis=1)
        value = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        return value, (logits, states, spikes)

    if engine == 'brainx_state':
        import brainstate as bs
        transformed = bs.transform.grad(loss, argnums=(0, 1), has_aux=True, return_value=True)
        def derivative(*args):
            (gradients, initial_gradients), value, aux = transformed(*args)
            return (value, aux), (gradients, initial_gradients)
        compile_ = bs.transform.jit
    else:
        derivative = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
        compile_ = jax.jit
    return derivative, compile_, metadata


def run_case(case, engine='spyx', steps=3):
    import importlib.metadata
    import time
    if engine == 'brainx_state':
        import brainstate as bs
        # Import initializes its own global precision; set the framework policy
        # before materializing arrays, not only JAX's flag before import.
        bs.environ.set(precision=64)
    else:
        jax.config.update('jax_enable_x64', True)
    weights = tuple(jnp.asarray(w, dtype=jnp.float64).reshape(a, b) for w, a, b in zip(case['weights'], case['sizes'][:-1], case['sizes'][1:]))
    initial = jnp.asarray(case.get('initial', [[0.] * sum(case['sizes'][1:]) for _ in case['labels']]), dtype=jnp.float64)
    x = jnp.asarray(case['inputs'], dtype=jnp.float64)
    labels = jnp.asarray(case['labels'], dtype=jnp.int32)
    derivative, compile_, metadata = make_loss(case, engine)
    full_derivative = compile_(derivative)
    before = time.perf_counter()
    (loss, aux), (grads, init_grad) = full_derivative(weights, initial, x, labels)
    jax.block_until_ready((loss, aux, grads, init_grad))
    qualification_cold_s = time.perf_counter() - before
    plain = lambda z: jax.device_get(z).tolist()
    out = dict(engine=engine, version=importlib.metadata.version('spyx' if engine == 'spyx' else 'brainstate'),
               jax=jax.__version__, jax_enable_x64=bool(jax.config.jax_enable_x64), parameter_dtype=str(weights[0].dtype),
               input_dtype=str(x.dtype), device=str(jax.devices()[0]), implementation='adapted_canonical_jit',
               identity=metadata, qualification_cold_s=qualification_cold_s, loss=float(loss), logits=plain(aux[0]),
               states=plain(aux[1]), spikes=plain(aux[2]), gradients=[plain(g.ravel()) for g in grads], initial_vjp=plain(init_grad), updates=[])
    sgd = optax.sgd(learning_rate=.001)
    deltas, _ = sgd.update(grads, sgd.init(weights), weights)
    out['one_sgd_update'] = [plain(w.ravel()) for w in optax.apply_updates(weights, deltas)]
    optimizer = optax.adam(learning_rate=.001, b1=.9, b2=.999, eps=1e-8, eps_root=0, mu_dtype=jnp.float64)
    opt_state = optimizer.init(weights)
    def update(params, state, init, inputs, targets):
        (value, _), (gradient, _) = derivative(params, init, inputs, targets)
        updates, new_state = optimizer.update(gradient, state, params)
        return optax.apply_updates(params, updates), new_state, value
    update = compile_(update)
    for _ in range(steps):
        start = time.perf_counter()
        weights, opt_state, value = update(weights, opt_state, initial, x, labels)
        jax.block_until_ready((weights, opt_state, value))
        elapsed = time.perf_counter() - start
        a = opt_state[0]
        out['updates'].append(dict(elapsed_s=elapsed, weights=[plain(w.ravel()) for w in weights], first=[plain(w.ravel()) for w in a.mu],
                                   second=[plain(w.ravel()) for w in a.nu], step=int(a.count)))
    margins = jnp.asarray([-3., -.2, 0., .1, 2.], dtype=jnp.float64)
    cotangents = jnp.asarray([2., -1., .3, -.7, 4.], dtype=jnp.float64)
    if engine == 'spyx':
        from spyx import axn
        activation = axn.superspike(k=5)
    else:
        activation = canonical_spike
    values, pullback = jax.vjp(activation, margins)
    out['local_surrogate_vjp'] = dict(margins=plain(margins), cotangents=plain(cotangents), values=plain(values), vjp=plain(pullback(cotangents)[0]))
    out['timing_role'] = 'qualification diagnostics only; includes first-JIT in first update, not a performance benchmark'
    return out
