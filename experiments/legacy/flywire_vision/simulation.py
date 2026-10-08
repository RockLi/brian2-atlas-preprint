"""Mapped artificial visual drive into the fixed full FlyWire conductance-LIF graph."""
from dataclasses import asdict, dataclass
import copy
import csv
import hashlib
from pathlib import Path
import struct
import numpy as np
from .mapping import audit
from .stimuli import MovieConfig, sample_hexels

READOUT_TYPES = tuple(f"T{n}{suffix}" for n in (4, 5) for suffix in "abcd") + ("LC4",)


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 783
    rate_hz: float = 120.
    input_weight_mv: float = 16.
    dt_ms: float = .1
    warmup_ms: float = 100.
    tail_ms: float = 100.
    frame_ms: float = 10.
    frames: int = 40
    background_rate_hz: float = 300.
    background_weight_mv: float = 3.5

    def __post_init__(self):
        values = np.array(list(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("invalid simulation parameter")
        if type(self.seed) is not int or type(self.frames) is not int or self.frames < 1:
            raise ValueError("invalid seed or frame count")
        if self.dt_ms != .1 or self.frame_ms <= 0:
            raise ValueError("retain dt 0.1 ms and positive frame duration")
        for value in (self.warmup_ms, self.tail_ms, self.frame_ms):
            if not np.isclose(value/self.dt_ms, round(value/self.dt_ms), rtol=0, atol=1e-9):
                raise ValueError("timing must align with simulation ticks")
        if max(self.rate_hz, self.background_rate_hz)*self.dt_ms/1000 >= 1:
            raise ValueError("at most one source spike per tick")

    @property
    def steps(self):
        return round((self.warmup_ms+self.frames*self.frame_ms+self.tail_ms)/self.dt_ms)


def bind_graph(graph, source_dir):
    report, eligible = audit(source_dir)
    lookup = {str(int(rid)): i for i, rid in enumerate(graph.root_ids)}
    with (Path(source_dir)/"annotations-v2.1.0.tsv").open() as f:
        old = {r["root_id"]:r for r in csv.DictReader(f, delimiter="\t")}
    if set(lookup) != set(old):
        raise ValueError("CSR node IDs differ from pinned annotation IDs")
    channels = sorted((r for r in eligible if r["cell_type"] in ("Mi1", "Tm1")),
                      key=lambda r:(r["cell_type"], int(r["root_id"])))
    for row in channels:
        row["index"] = lookup[row["root_id"]]
    groups = {name: np.array([r["index"] for r in channels if r["cell_type"] == name], dtype=int)
              for name in ("Mi1", "Tm1")}
    with (Path(source_dir)/"visual-system-parts-list/data/neuron_table.csv").open() as f:
        visual = list(csv.DictReader(f))
    for name in READOUT_TYPES:
        groups[name] = np.array(sorted(lookup[r["cell ID"]] for r in visual
                                      if r["resolved type"] == name and r["side (most synapses)"] == "right"
                                      and (old[r["cell ID"]]["cell_type"] or old[r["cell ID"]]["hemibrain_type"]) == name
                                      and old[r["cell ID"]]["side"] == "right"), dtype=int)
        if not len(groups[name]):
            raise ValueError(f"missing concordant readout {name}")
    input_indices = np.array([r["index"] for r in channels], dtype=int)
    if len(np.unique(input_indices)) != len(input_indices):
        raise ValueError("duplicate visual input target")
    if np.intersect1d(input_indices, np.concatenate([groups[k] for k in READOUT_TYPES])).size:
        raise ValueError("input/readout overlap")
    return channels, groups, report


def encode_movie(frames, channels, config):
    """One-to-one contrast-rate drive: Mi1 positive, Tm1 negative luminance contrast.

    No direction, class label or prior prediction enters this interface. Phases
    carry across frames and reset only at the beginning of the entire movie.
    """
    frames = np.asarray(frames)
    if frames.shape != (config.frames,48,48) or not np.isfinite(frames).all():
        raise ValueError("expected finite 40-frame 48x48 input (or configured frame count)")
    if np.any(frames < 0) or np.any(frames > 1):
        raise ValueError("luminance outside [0,1]")
    if not channels or any(r["cell_type"] not in ("Mi1", "Tm1") for r in channels):
        raise ValueError("expected mapped Mi1/Tm1 channels")
    sampled = sample_hexels(frames,[[r["p"],r["q"]] for r in channels])
    polarity = np.array([1 if r["cell_type"] == "Mi1" else -1 for r in channels])
    strength = np.clip(2*(sampled-.5)*polarity,0,1)
    phase = np.random.default_rng(config.seed).random(len(channels))
    start = round(config.warmup_ms/config.dt_ms)
    width = round(config.frame_ms/config.dt_ms)
    indices, ticks = [], []
    for frame, values in enumerate(strength):
        increment = values*config.rate_hz*config.dt_ms/1000
        for offset in range(width):
            phase += increment
            fired = np.flatnonzero(phase >= 1)
            phase[fired] -= 1
            indices.extend(fired.tolist())
            ticks.extend([start+frame*width+offset]*len(fired))
    return np.asarray(indices,dtype=np.int32), np.asarray(ticks,dtype=np.int64), strength


def build_model(graph, channels, config, runner):
    import brian2 as b
    import brian2_rust as rust
    from brian2.devices.device import all_devices
    from brian2_rust.export import lower_network
    from flywire_mnist.encoding import keyed_rng, schedule
    device = all_devices["rust_standalone"]
    previous = b.get_device()
    device.reinit()
    b.set_device("rust_standalone", runner=Path(runner))
    try:
        n = len(graph.root_ids)
        g = b.NeuronGroup(n, """dv/dt=(-(v+52*mV)-ge*v-gi*(v+70*mV))/(20*ms) : volt (unless refractory)
        dge/dt=-ge/(5*ms) : 1
        dgi/dt=-gi/(5*ms) : 1
        transmission : 1""", threshold="v>-45*mV", reset="v=-52*mV", refractory=2.2*b.ms,
                          method="euler", dt=config.dt_ms*b.ms, name="flywire_neurons")
        rng = np.random.default_rng(config.seed)
        g.v = (-52+rng.uniform(-.8,.8,n))*b.mV
        g.ge = config.background_rate_hz*.005*config.background_weight_mv/52
        g.gi = 0
        g.transmission = 1
        syn = b.Synapses(g,g,"signed_contacts : 1 (constant)", on_pre="""ge_post += w*signed_contacts*int(signed_contacts>0)*transmission_pre
        gi_post -= w*inhibition*signed_contacts*int(signed_contacts<0)*transmission_pre""",
                         delay=1.8*b.ms, clock=g.clock, name="flywire_recurrent",
                         namespace={"w":.275/52,"inhibition":4.})
        if graph.csr:
            rust.connect_binary_csr(syn,graph.csr,parameters={"signed_contacts":0})
        else:
            syn.connect(i=graph.source,j=graph.target,namespace={})
            syn.signed_contacts=graph.contacts
        bg_i,bg_t = schedule(np.full(min(n,512),config.background_rate_hz),0,config.steps,
                            config.dt_ms,keyed_rng(config.seed,"background"))
        bg=b.SpikeGeneratorGroup(min(n,512),bg_i,bg_t*g.clock.dt,clock=g.clock,sorted=True,name="background")
        bs=b.Synapses(bg,g,"amplitude : 1 (constant)",on_pre="ge_post += amplitude",
                      delay=0*b.ms,clock=g.clock,name="background_connections")
        bs.connect(i=rng.permutation(n)%min(n,512),j=np.arange(n),namespace={})
        bs.amplitude=config.background_weight_mv/52
        stimulus=b.SpikeGeneratorGroup(len(channels),[],[]*b.ms,clock=g.clock,sorted=True,name="visual_input")
        drive=b.Synapses(stimulus,g,"amplitude : 1 (constant)",on_pre="ge_post += amplitude",
                         delay=0*b.ms,clock=g.clock,name="visual_connections")
        drive.connect(i=np.arange(len(channels)),j=np.array([r["index"] for r in channels]),namespace={})
        drive.amplitude=config.input_weight_mv/52
        monitor=b.SpikeMonitor(g,name="vision_spikes")
        return lower_network(b.Network(g,syn,bg,bs,stimulus,drive,monitor),
                             config.steps*config.dt_ms*b.ms,namespace={},rng_seed=config.seed)
    finally:
        b.set_device(previous)
        device.reinit()


def cut_model(model, channels):
    from brian2_rust.protocol import attach_protocol
    result=copy.deepcopy(model)
    ni=next(i for i,p in enumerate(result["definition"]["populations"]) if p["name"]=="flywire_neurons")
    state=result["instance"]["populations"][ni]["initial_state"]["transmission"]
    one=struct.pack(">d",1.).hex()
    zero=struct.pack(">d",0.).hex()
    for row in channels:
        if state[row["index"]] != one:
            raise ValueError("input transmission already modified")
        state[row["index"]]=zero
    attach_protocol(result)
    return result


def summarize(result, model, groups, channels, config):
    ni=next(i for i,p in enumerate(model["definition"]["populations"]) if p["name"]=="flywire_neurons")
    population=result["populations"][ni]
    indices,ticks=np.asarray(population["indices"]),np.asarray(population["spike_ticks"])
    bin_ticks=round(config.frame_ms/config.dt_ms)
    bins=(config.steps+bin_ticks-1)//bin_ticks
    input_ids=np.array([r["index"] for r in channels])
    neuron_count=model["definition"]["populations"][ni]["count"]
    mapping=np.full(neuron_count,-1,dtype=int);mapping[input_ids]=np.arange(len(input_ids))
    selected=mapping[indices]>=0
    input_counts=np.zeros((bins,len(input_ids)),dtype=np.int32)
    np.add.at(input_counts,(ticks[selected]//bin_ticks,mapping[indices[selected]]),1)
    names=list(groups)
    series=[];active={}
    for name in names:
        ids=groups[name]
        mask=np.isin(indices,ids)
        series.append(np.bincount(ticks[mask]//bin_ticks,minlength=bins))
        during=mask&(ticks>=round(config.warmup_ms/config.dt_ms))&(ticks<round((config.warmup_ms+config.frames*config.frame_ms)/config.dt_ms))
        active[name]=len(np.unique(indices[during]))
    noninput=~np.isin(indices,input_ids)
    event_hash=lambda mask: hashlib.sha256(np.stack([ticks[mask],indices[mask]],axis=1).astype('<i8').tobytes()).hexdigest()
    report={"total_spikes":len(indices),"neural_event_sha256":event_hash(np.ones(len(indices),dtype=bool)),
            "noninput_event_sha256":event_hash(noninput),"active_during_stimulus":active,
            "group_names":names,"group_sizes":{k:len(v) for k,v in groups.items()},
            "group_counts":np.stack(series,axis=1).tolist(),
            "simulation_wall_seconds":result["wall_seconds"]}
    return report,input_counts
