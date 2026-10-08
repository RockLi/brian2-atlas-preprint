"""LK2014 Fig. 5 triplet variant, with the membrane/coupling Tables 1-2.

Weights ``w`` are numerical values in pF, so physical charge-like synaptic
kernel areas are ``w*pF``. The population code uses SI Brian quantities.
All backends receive the same explicitly materialized topology and initial state.
See LITWIN_KUMAR.md for numerical choices and the acceptance protocol.
"""
from dataclasses import asdict, dataclass

import brian2 as b
import numpy as np


TRIPLET_EQUATIONS = '''
dr1/dt = -r1/tau_r1 : 1 (event-driven)
dr2/dt = -r2/tau_r2 : 1 (event-driven)
do1/dt = -o1/tau_o1 : 1 (event-driven)
do2/dt = -o2/tau_o2 : 1 (event-driven)
w : 1
'''
# Weight reads precede slow-trace increments: Pfister & Gerstner Eqs. 3-4.
TRIPLET_PRE = '''
w = clip(w - learning*(A2_minus*o1 + A3_minus*o1*r2), wmin, wmax)
r1 += 1
r2 += 1
'''
TRIPLET_POST = '''
w = clip(w + learning*(A2_plus*r1 + A3_plus*r1*o2), wmin, wmax)
o1 += 1
o2 += 1
'''
TRIPLET_PARAMETERS = {
    'tau_r1': 16.8*b.ms, 'tau_r2': 101*b.ms,
    'tau_o1': 33.7*b.ms, 'tau_o2': 125*b.ms,
    'A2_plus': 7.5e-10, 'A3_plus': 9.3e-3,
    'A2_minus': 7e-3, 'A3_minus': 2.3e-4,
}

INHIBITORY_EQUATIONS = '''
dxpre/dt = -xpre/tau_i : 1 (event-driven)
dxpost/dt = -xpost/tau_i : 1 (event-driven)
w : 1
'''
INHIBITORY_PRE = 'w = clip(w+learning*eta*(xpost-alpha), wmin, wmax)\nxpre += 1'
INHIBITORY_POST = 'w = clip(w+learning*eta*xpre, wmin, wmax)\nxpost += 1'


@dataclass(frozen=True)
class LKConfig:
    scale: float = 1.0
    seed: int = 20260906
    dt_ms: float = 0.1
    mode: str = 'learn'
    warmup_s: float = 10.0
    stimulus_s: float = 1.0
    gap_s: float = 3.0
    repetitions: int = 20
    spontaneous_s: float = 1000.0
    normalization_ms: float = 20.0
    assemblies: int = 20
    membership_probability: float = 0.05
    connection_probability: float = 0.2
    plasticity: bool = True
    inhibitory_plasticity: bool = True
    normalization: bool = True
    stimulation: bool = True
    learning_multiplier: float = 1.0
    delay_ms: float = 1.5
    delay_distribution: str = 'fixed'
    trace_integration: str = 'event-driven'
    input_sources: int = 1000
    inhibitory_stimulus_factor: float = 0.0

    def __post_init__(self):
        if self.delay_distribution not in {'fixed', 'uniform'}:
            raise ValueError('delay distribution must be fixed or uniform')
        if self.trace_integration not in {'event-driven', 'euler', 'euler-event'}:
            raise ValueError('trace integration must be event-driven, euler, or euler-event')
        if self.trace_integration == 'euler-event' and self.dt_ms >= 16.8:
            raise ValueError('Euler-equivalent event decay requires dt below every trace time constant')
        if any(not np.isfinite(value) for value in asdict(self).values()
               if isinstance(value, (float, int))):
            raise ValueError('all numerical configuration values must be finite')
        if any(type(getattr(self, name)) is not int for name in
               ['seed', 'repetitions', 'assemblies', 'input_sources']):
            raise ValueError('seed, repetitions, assemblies and input_sources must be integers')
        if not 0.01 <= self.scale <= 1 or self.mode not in {'learn', 'clustered'}:
            raise ValueError('scale must be in [0.01, 1]; mode is learn or clustered')
        if (self.dt_ms <= 0 or self.normalization_ms < self.dt_ms or
                self.stimulus_s <= 0 or self.gap_s < 0 or self.warmup_s < 0 or
                self.spontaneous_s < 0 or self.repetitions < 0 or self.assemblies < 1 or
                self.input_sources < 1 or self.learning_multiplier < 0 or self.delay_ms < 0 or
                self.inhibitory_stimulus_factor < 0):
            raise ValueError('invalid protocol dimensions')
        if not (0 < self.connection_probability <= 1 and
                0 < self.membership_probability < 1):
            raise ValueError('invalid connection/membership probability')
        if max(8000, 8000*self.inhibitory_stimulus_factor)*self.dt_ms/1000 > self.input_sources:
            raise ValueError('external source firing probability exceeds one per timestep')
        for duration_ms in [self.normalization_ms, self.delay_ms]:
            if not np.isclose(duration_ms/self.dt_ms, round(duration_ms/self.dt_ms)):
                raise ValueError('normalization and delay must align with the neuron clock')

    @property
    def ne(self):
        return round(4000*self.scale)

    @property
    def ni(self):
        return round(1000*self.scale)

    @property
    def training_s(self):
        return self.assemblies*self.repetitions*(self.stimulus_s+self.gap_s)

    @property
    def duration_s(self):
        return self.warmup_s+self.training_s+self.spontaneous_s

    def to_dict(self):
        return {**asdict(self), 'ne': self.ne, 'ni': self.ni,
                'training_s': self.training_s, 'duration_s': self.duration_s}


def bernoulli_edges(ns, nt, probability, rng, recurrent=False):
    sources, targets = [], []
    for i in range(ns):
        mask = rng.random(nt) < probability
        if recurrent:
            mask[i] = False
        j = np.flatnonzero(mask).astype(np.int32)
        sources.append(np.full(len(j), i, dtype=np.int32))
        targets.append(j)
    return np.concatenate(sources), np.concatenate(targets)


def project_incoming(weights, targets, target_sum, lower, upper):
    """Euclidean box/row-sum projection, used only for clustered initialization."""
    low = np.full(len(target_sum), -upper)
    high = np.full(len(target_sum), upper)
    for _ in range(60):
        shift = (low+high)/2
        candidate = np.clip(weights-shift[targets], lower, upper)
        sums = np.bincount(targets, weights=candidate, minlength=len(target_sum))
        low = np.where(sums > target_sum, shift, low)
        high = np.where(sums <= target_sum, shift, high)
    return np.clip(weights-((low+high)/2)[targets], lower, upper)


@dataclass
class LKNetwork:
    config: LKConfig
    network: b.Network
    exc: b.NeuronGroup
    inh: b.NeuronGroup
    synapses: dict
    spikes: dict
    voltage: b.StateMonitor
    membership: dict
    edges: dict
    initial_weights: dict

    def snapshot(self):
        result = {}
        for label, group in [('exc', self.exc), ('inh', self.inh)]:
            for name in group.equations.names:
                equation = group.equations[name]
                if equation.type != 'subexpression':
                    result[f'{label}_{name}'] = np.asarray(getattr(group, name)[:]).copy()
            result[f'{label}_lastspike'] = np.asarray(group.lastspike[:]).copy()
            result[f'{label}_not_refractory'] = np.asarray(group.not_refractory[:]).copy()
        for label, synapse in self.synapses.items():
            equations = set(synapse.equations.names)
            if synapse.event_driven is not None:
                equations |= set(synapse.event_driven.names)
            for name in equations | {'lastupdate'}:
                if name in synapse.variables and not name.endswith('_post'):
                    result[f'{label}_{name}'] = np.asarray(getattr(synapse, name)[:]).copy()
        for label, monitor in self.spikes.items():
            result[f'{label}_spike_i'] = np.asarray(monitor.i[:]).copy()
            result[f'{label}_spike_t'] = np.asarray(monitor.t[:]/b.second).copy()
        result['voltage_t'] = np.asarray(self.voltage.t[:]/b.second).copy()
        result['voltage_v'] = np.asarray(self.voltage.v[:]/b.mV).copy()
        return result


def make_network(config: LKConfig, replay_duration_s=None, replay_base_dt_ms=None):
    b.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    # Delay sensitivity must not change topology, membership, or initial voltage.
    delay_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x4C4B444C]))
    def trace_tau(tau):
        if config.trace_integration != 'euler-event':
            return tau
        # exp(-k*dt/tau_eff) = (1-dt/tau)**k in exact arithmetic.
        # This avoids touching every silent synapse at every clock tick, while
        # retaining Euler's geometric attenuation. Roundoff differs from k
        # successive multiplications; literal 'euler' is the validation oracle.
        dt = config.dt_ms*b.ms
        return -dt/np.log1p(-float(dt/tau))
    triplet_parameters = {name: trace_tau(value) if name.startswith('tau_') else value
                          for name, value in TRIPLET_PARAMETERS.items()}
    clock = b.Clock(dt=config.dt_ms*b.ms, name='lk_clock')
    membership = {label: rng.random((config.assemblies, n)) < config.membership_probability
                  for label, n in [('exc', config.ne), ('inh', config.ni)]}
    # One block is repeated by modulo; storage is independent of repetitions.
    pattern_functions = {label: b.TimedArray(values.astype(float),
                         dt=(config.stimulus_s+config.gap_s)*b.second,
                         name=f'lk_patterns_{label}') for label, values in membership.items()}
    namespace = {
        'tau_m': 20*b.ms, 'capacitance': 300*b.pF,
        'Eexc': 0*b.mV, 'Einh': -75*b.mV,
        'tau_re': 1*b.ms, 'tau_de': 6*b.ms,
        'tau_ri': .5*b.ms, 'tau_di': 2*b.ms,
        'Vreset': -60*b.mV, 'Vthreshold': -52*b.mV,
        'DeltaT': 2*b.mV, 'tau_threshold': 30*b.ms, 'threshold_jump': 10*b.mV,
        'tau_adapt': 150*b.ms, 'adapt_a': 4*b.nS, 'adapt_b': .805*b.pA,
        'warmup': config.warmup_s*b.second,
        'training_end': (config.warmup_s+config.training_s)*b.second,
        'stimulus_duration': config.stimulus_s*b.second,
        'period': (config.stimulus_s+config.gap_s)*b.second,
        'block_duration': config.assemblies*(config.stimulus_s+config.gap_s)*b.second,
        'stim_enabled': float(config.stimulation),
    }
    synaptic_dynamics = '''
    dxe/dt = -xe/tau_re : siemens
    dge/dt = (xe-ge)/tau_de : siemens
    dxinh/dt = -xinh/tau_ri : siemens
    dgi/dt = (xinh-gi)/tau_di : siemens
    stimulus = stim_enabled*int(t >= warmup)*int(t < training_end)*int((t-warmup)%period < stimulus_duration)*patterns((t-warmup)%block_duration, i) : 1
    spike_total : 1
    '''
    exc = b.NeuronGroup(config.ne, '''
    dv/dt = (Erest-v+DeltaT*exp((v-vthreshold)/DeltaT))/tau_m + (ge*(Eexc-v)+gi*(Einh-v)-adapt)/capacitance : volt (unless refractory)
    dvthreshold/dt = (Vthreshold-vthreshold)/tau_threshold : volt
    dadapt/dt = (adapt_a*(v-Erest)-adapt)/tau_adapt : amp
    exc_weight_sum : 1
    weight_target : 1
    incoming_count : 1
    '''+synaptic_dynamics,
        threshold='v > 20*mV',
        reset='v = Vreset; vthreshold = Vthreshold+threshold_jump; adapt += adapt_b; spike_total += 1',
        refractory=1*b.ms, method='euler', clock=clock, name='lk_exc',
        namespace={**namespace, 'Erest': -70*b.mV, 'patterns': pattern_functions['exc']})
    inh = b.NeuronGroup(config.ni, '''
    dv/dt = (Erest-v)/tau_m + (ge*(Eexc-v)+gi*(Einh-v))/capacitance : volt (unless refractory)
    '''+synaptic_dynamics,
        threshold='v > Vthreshold', reset='v = Vreset; spike_total += 1',
        refractory=1*b.ms, method='euler', clock=clock, name='lk_inh',
        namespace={**namespace, 'Erest': -62*b.mV, 'patterns': pattern_functions['inh']})
    exc.v = rng.uniform(-70, -60, config.ne)*b.mV
    exc.vthreshold = -52*b.mV
    inh.v = rng.uniform(-62, -52, config.ni)*b.mV
    # Per-target independent finite-source binomial drive. At full scale each
    # of 1000 sources contributes 4.5/2.25 Hz baseline and 8 Hz when stimulated.
    drives = []
    for label, group, rate, weight in [('exc', exc, 4500, 1.78), ('inh', inh, 2250, 1.27)]:
        group.namespace['external_jump'] = weight*b.pF/(1*b.ms)
        stimulus_rate = 8000*(1. if label == 'exc' else config.inhibitory_stimulus_factor)
        if replay_duration_s is None:
            drives += [b.PoissonInput(group, 'xe', N=config.input_sources,
                                     rate=rate/config.input_sources*b.Hz,
                                     weight='external_jump'),
                       b.PoissonInput(group, 'xe', N=config.input_sources,
                                      rate=stimulus_rate/config.input_sources*b.Hz,
                                      weight='external_jump*stim_enabled*int(t >= warmup)*int(t < training_end)*int((t-warmup)%period < stimulus_duration)*patterns((t-warmup)%block_duration, i)')]
        else:
            base_dt_ms = config.dt_ms if replay_base_dt_ms is None else replay_base_dt_ms
            if not np.isfinite(base_dt_ms) or base_dt_ms <= 0:
                raise ValueError('replay base dt must be finite and positive')
            factor = config.dt_ms/base_dt_ms
            if not np.isclose(factor, round(factor), rtol=0, atol=1e-10) or factor < 1:
                raise ValueError('replay base dt must divide the simulation timestep')
            factor = round(factor)
            steps = round(replay_duration_s*1000/base_dt_ms)
            if steps % factor:
                raise ValueError('replay duration must align with simulation timestep')
            if steps*len(group) > 10_000_000:
                raise ValueError('replay inputs are a short correctness fixture (max 10M values/population)')
            input_rng = np.random.default_rng(config.seed ^ (0x1234 if label == 'exc' else 0x5678))
            times = np.arange(steps)*base_dt_ms/1000
            phase = times-config.warmup_s
            period = config.stimulus_s+config.gap_s
            pattern_index = (np.floor(phase/period).astype(int) % config.assemblies)
            active = ((phase >= 0) & (phase < config.training_s) &
                      (np.remainder(phase, period) < config.stimulus_s) & config.stimulation)
            counts = input_rng.binomial(config.input_sources,
                         rate/config.input_sources*base_dt_ms/1000, (steps, len(group)))
            counts += input_rng.binomial(config.input_sources,
                         stimulus_rate/config.input_sources*base_dt_ms/1000, (steps, len(group))) * (
                             membership[label][pattern_index] & active[:, None])
            counts = counts.reshape(steps//factor, factor, len(group)).sum(axis=1)
            group.namespace['input_events'] = b.TimedArray(
                counts.astype(float), dt=config.dt_ms*b.ms, name=f'lk_replay_{label}')
            # End-of-tick delivery commutes with resets (which never write xe)
            # and is available to the next integration step, like PoissonInput.
            group.run_regularly('xe += external_jump*input_events(t, i)',
                                when='end', name=f'lk_drive_{label}')
    edges, projections, initial = {}, {}, {}
    populations = {'e': exc, 'i': inh}
    for label in ['ee', 'ei', 'ie', 'ii']:
        source, target = (populations[c] for c in label)
        edges[label] = bernoulli_edges(len(source), len(target),
                                      config.connection_probability, rng,
                                      recurrent=label[0] == label[1])
        if label == 'ee':
            equations = TRIPLET_EQUATIONS
            if config.trace_integration == 'euler':
                equations = equations.replace('(event-driven)', '(clock-driven)')
            synapse = b.Synapses(source, target,
                equations+'exc_weight_sum_post = w : 1 (summed)\n'
                'learning = plasticity_scale*int(t >= warmup) : 1',
                on_pre='xe_post += w*pF/tau_re\n'+TRIPLET_PRE,
                on_post=TRIPLET_POST,
                namespace={**triplet_parameters, 'tau_re': 1*b.ms,
                           'warmup': config.warmup_s*b.second,
                           'plasticity_scale': config.learning_multiplier*int(config.plasticity),
                           'wmin': 1.78, 'wmax': 21.4},
                clock=clock, method='euler', name='lk_ee')
            start_weight = 2.76
        elif label == 'ie':
            equations = INHIBITORY_EQUATIONS
            if config.trace_integration == 'euler':
                equations = equations.replace('(event-driven)', '(clock-driven)')
            synapse = b.Synapses(source, target, equations+
                'learning = plasticity_scale*int(t >= warmup) : 1',
                on_pre='xinh_post += w*pF/tau_ri\n'+INHIBITORY_PRE,
                on_post=INHIBITORY_POST,
                namespace={'tau_i': trace_tau(20*b.ms), 'tau_ri': .5*b.ms, 'eta': 1.0,
                           'wmin': 48.7, 'wmax': 243.,
                           'alpha': .12, 'warmup': config.warmup_s*b.second,
                           'plasticity_scale': config.learning_multiplier*int(config.inhibitory_plasticity)},
                clock=clock, method='euler', name='lk_ie')
            start_weight = 48.7
        else:
            synapse = b.Synapses(source, target, 'w : 1',
                on_pre=('xe_post += w*pF/tau_re' if label == 'ei' else 'xinh_post += w*pF/tau_ri'),
                namespace={'tau_re': 1*b.ms, 'tau_ri': .5*b.ms}, clock=clock, name=f'lk_{label}')
            start_weight = 1.27 if label == 'ei' else 16.2
        synapse.connect(i=edges[label][0], j=edges[label][1])
        synapse.w = start_weight
        if config.delay_distribution == 'fixed':
            synapse.pre.delay = config.delay_ms*b.ms
        else:
            # A discrete uniform distribution over every inclusive clock tick.
            ticks = delay_rng.integers(0, round(config.delay_ms/config.dt_ms)+1,
                                       size=len(edges[label][0]))
            synapse.pre.delay = ticks*config.dt_ms*b.ms
        projections[label] = synapse
        initial[label] = np.full(len(edges[label][0]), start_weight)
    sources, targets = edges['ee']
    counts = np.bincount(targets, minlength=config.ne)
    exc.incoming_count = np.maximum(counts, 1)
    exc.weight_target = counts*2.76
    if config.mode == 'clustered':
        shared = np.any(membership['exc'][:, sources] & membership['exc'][:, targets], axis=0)
        initial['ee'] = project_incoming(np.where(shared, 21.4, 1.78), targets,
                                         counts*2.76, 1.78, 21.4)
        projections['ee'].w = initial['ee']
    if config.normalization:
        normalizer = projections['ee'].run_regularly(
            'w = clip(w-(exc_weight_sum_post-weight_target_post)/incoming_count_post, wmin, wmax)',
            dt=config.normalization_ms*b.ms, when='groups', order=0,
            name='lk_normalization')
        # Brian exposes the updater as a BrianObject: use the same 20 ms
        # clock, with reduction order -1 followed by normalization order 0.
        projections['ee'].summed_updaters['exc_weight_sum_post']._clock = normalizer.clock
    spikes = {'exc': b.SpikeMonitor(exc, name='lk_exc_spikes'),
              'inh': b.SpikeMonitor(inh, name='lk_inh_spikes')}
    voltage = b.StateMonitor(exc, 'v', record=np.arange(min(8, config.ne)), name='lk_voltage')
    network = b.Network(exc, inh, *projections.values(), *drives, *spikes.values(), voltage)
    return LKNetwork(config, network, exc, inh, projections, spikes, voltage,
                     membership, edges, initial)
