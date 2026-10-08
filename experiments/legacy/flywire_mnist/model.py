"""Fixed conductance LIF dynamics with observational spike-window counters."""
import copy
from pathlib import Path
import numpy as np
from .encoding import encode, keyed_rng, projection, schedule

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "target/release/b2-runner"


def build(graph, config, *, recurrent=True, monitor=False, counters=True):
    import brian2 as b
    import brian2_rust as rust
    from brian2.devices.device import all_devices
    from brian2_rust.export import lower_network
    previous = b.get_device()
    device = all_devices["rust_standalone"]
    device.reinit()
    b.set_device("rust_standalone", runner=RUNNER)
    try:
        n = len(graph.root_ids)
        equations = """dv/dt=(-(v+52*mV)-ge*v-gi*(v+70*mV))/(20*ms) : volt (unless refractory)
        dge/dt=-ge/(5*ms) : 1
        dgi/dt=-gi/(5*ms) : 1
        transmission : 1"""
        reset = "v=-52*mV"
        windows = zip(config.edges, config.edges[1:]) if counters else []
        for k, (start, stop) in enumerate(windows):
            equations += f"\ncount{k} : 1"
            reset += f"\ncount{k} += int(timestep(t, dt)>={start})*int(timestep(t, dt)<{stop})"
        g = b.NeuronGroup(n, equations, threshold="v>-45*mV", reset=reset,
                          refractory=2.2*b.ms, method="euler", dt=config.dt_ms*b.ms,
                          name="flywire_neurons")
        rng = np.random.default_rng(config.seed)
        g.v = (-52+rng.uniform(-.8, .8, n))*b.mV
        g.ge = config.background_rate_hz*.005*config.background_weight_mv/52
        g.gi = 0
        g.transmission = int(recurrent)
        syn = b.Synapses(g, g, "signed_contacts : 1 (constant)",
                        on_pre="""ge_post += w*signed_contacts*int(signed_contacts>0)*transmission_pre
                        gi_post -= w*inhibition*signed_contacts*int(signed_contacts<0)*transmission_pre""",
                        delay=1.8*b.ms, clock=g.clock, name="flywire_recurrent",
                        namespace={"w": .275/52, "inhibition": 4.})
        if graph.csr:
            rust.connect_binary_csr(syn, graph.csr, parameters={"signed_contacts": 0})
        else:
            syn.connect(i=graph.source, j=graph.target, namespace={})
            syn.signed_contacts = graph.contacts
        channels = min(n, config.background_channels)
        bg_i, bg_t = schedule(np.full(channels, config.background_rate_hz), 0,
                              config.ticks, config.dt_ms, keyed_rng(config.seed, "background"))
        bg = b.SpikeGeneratorGroup(channels, bg_i, bg_t*g.clock.dt,
                                  clock=g.clock, sorted=True, name="background")
        bs = b.Synapses(bg, g, "amplitude : 1 (constant)", on_pre="ge_post += amplitude",
                        delay=0*b.ms, clock=g.clock, name="background_connections")
        bs.connect(i=rng.permutation(n)%channels, j=np.arange(n), namespace={})
        bs.amplitude = config.background_weight_mv/52
        pixels = b.SpikeGeneratorGroup(784, [], []*b.ms, clock=g.clock,
                                      sorted=True, name="pixels")
        drive = b.Synapses(pixels, g, "amplitude : 1 (constant)", on_pre="ge_post += amplitude",
                           delay=0*b.ms, clock=g.clock, name="pixel_connections")
        sources, targets = projection(graph.inputs, config.fanout, config.seed)
        drive.connect(i=sources, j=targets, namespace={})
        drive.amplitude = config.input_weight_mv/(52*config.fanout)
        objects = [g, syn, bg, bs, pixels, drive]
        if monitor:
            objects.append(b.SpikeMonitor(g, name="diagnostic_spikes"))
        return lower_network(b.Network(*objects), config.ticks*config.dt_ms*b.ms,
                             namespace={}, rng_seed=config.seed)
    finally:
        b.set_device(previous)
        device.reinit()


def instance(template, image, sample_id, config):
    from brian2_rust.protocol import attach_protocol
    model = copy.deepcopy(template)
    p = next(i for i, p in enumerate(model["definition"]["populations"]) if p["name"] == "pixels")
    indices, ticks = encode(image, sample_id, config)
    model["instance"]["populations"][p]["spike_generator"] = {
        "spike_ticks": ticks.tolist(), "spike_indices": indices.tolist()}
    attach_protocol(model)
    return model


def features(model, result, readouts, bins):
    p = next(i for i, p in enumerate(model["definition"]["populations"])
             if p["name"] == "flywire_neurons")
    states = result["populations"][p]["states"]
    values = np.concatenate([np.asarray(states[f"count{k}"])[readouts] for k in range(bins)])
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("invalid spike counts")
    return values
