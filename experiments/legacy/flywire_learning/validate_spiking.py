"""Independent tick reference vs Brian NumPy/C++ and Rust CPU trajectories."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from spiking import NEURON_EQUATIONS, plastic, setup

SPIKE_TICKS = np.array([0, 9, 10, 29, 45, 49, 61], dtype=np.int32)
SPIKE_IDS = np.array([0, 1, 0, 0, 1, 0, 1], dtype=np.int32)
PRE, POST = np.array([0, 0, 1]), np.array([0, 1, 1])
CONTACTS = np.array([80., 50., 100.])
BOUNDARIES = [10, 20, 33, 50, 80]


def signals():
    reward, gate = np.zeros(80), np.ones(80)
    reward[12:20] = 1
    reward[32:42] = 1
    gate[50:] = 0
    return reward, gate


def independent():
    dt = .0001
    reward, gate = signals()
    v, ge, gi = np.array([-.0445, -.052]), np.zeros(2), np.zeros(2)
    gain, eligibility = np.ones(3), np.zeros(3)
    last = np.full(2, -100000000, dtype=np.int64)
    traces, samples, spike_i, spike_t = [], [], [], []
    for tick in range(80):
        traces.append(np.stack([v.copy(), ge.copy(), gi.copy()]))
        active = tick - last >= 22
        v[active] += dt * (-(v[active]+.052)-ge[active]*v[active]-gi[active]*(v[active]+.070))/.020
        ge *= 1-dt/.005
        gi *= 1-dt/.005
        old = eligibility.copy()
        eligibility *= 1-dt/.001
        gain += dt*gate[tick]*old*(40*(1-reward[tick])*(1-gain)-80*reward[tick]*gain)
        spikes = np.flatnonzero((v > -.045) & active)
        last[spikes] = tick
        spike_i.extend(spikes.tolist())
        spike_t.extend([tick*.0001]*len(spikes))
        for source in SPIKE_IDS[SPIKE_TICKS+2 == tick]:
            for edge in np.flatnonzero(PRE == source):
                ge[POST[edge]] += (.275/52)*CONTACTS[edge]*gain[edge]
                eligibility[edge] = 1
        v[spikes] = -.052
        if tick+1 in BOUNDARIES:
            samples.append(np.stack([gain.copy(), eligibility.copy()]))
    return dict(v=v, ge=ge, gi=gi, gain=gain, eligibility=eligibility,
                neuron_trace=np.transpose(traces, (1, 2, 0)),
                synapse_samples=np.array(samples),
                spike_i=np.array(spike_i, dtype=np.int32), spike_t=np.array(spike_t))


def child(backend, output, threads=1):
    import brian2 as b
    output.mkdir(parents=True, exist_ok=False)
    setup(backend, output/'project', threads)
    g = b.NeuronGroup(2, NEURON_EQUATIONS, threshold='v>-45*mV', reset='v=-52*mV',
                      refractory=2.2*b.ms, dt=.1*b.ms, method='euler', name='neurons')
    g.v = [-44.5, -52]*b.mV
    source = b.SpikeGeneratorGroup(2, SPIKE_IDS, SPIKE_TICKS*.1*b.ms,
                                   clock=g.clock, sorted=True, name='source')
    reward, gate = signals()
    teaching = b.TimedArray(reward, dt=.1*b.ms, name='teaching')
    learning = b.TimedArray(gate, dt=.1*b.ms, name='learning')
    s = plastic(source, g, PRE, POST, CONTACTS, teaching, learning,
                delay_ms=.2, tau_ms=1, depression_hz=80, recovery_hz=40)
    m = b.StateMonitor(g, ['v', 'ge', 'gi'], record=True, name='neuron_trace')
    spikes = b.SpikeMonitor(g, name='spikes')
    net = b.Network(g, source, s, m, spikes)
    samples = []
    if backend == 'cpp':
        net.run(8*b.ms)
        b.device.build(directory=str(output/'project'), compile=True, run=True, with_output=False)
    else:
        previous = 0
        for end in BOUNDARIES:
            net.run((end-previous)*.1*b.ms)
            samples.append(np.stack([np.array(s.gain[:]), np.array(s.eligibility[:])]))
            previous = end
    snapshot = dict(v=np.array(g.v[:]/b.volt), ge=np.array(g.ge[:]), gi=np.array(g.gi[:]),
                    gain=np.array(s.gain[:]), eligibility=np.array(s.eligibility[:]),
                    neuron_trace=np.array([m.v/b.volt, m.ge, m.gi]),
                    spike_i=np.array(spikes.i[:]), spike_t=np.array(spikes.t[:]/b.second))
    if samples:
        snapshot['synapse_samples'] = np.array(samples)
    np.savez(output/'snapshot.npz', **snapshot)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    expected = independent()
    np.savez(output/'independent.npz', **expected)
    report = {}
    for backend, threads in [('numpy', 1), ('cpp', 1), ('reference', 1), ('aot', 1), ('aot', 4)]:
        name = f'{backend}-{threads}'
        with (output/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, __file__, '--child', backend, '--threads', str(threads),
                            '--output', str(output/name)], check=True, stdout=log, stderr=subprocess.STDOUT)
        with np.load(output/name/'snapshot.npz') as actual:
            errors = {}
            for field in actual.files:
                np.testing.assert_allclose(actual[field], expected[field], rtol=1e-12, atol=1e-14, err_msg=f'{name}/{field}')
                errors[field] = float(np.max(np.abs(actual[field]-expected[field]), initial=0))
            np.testing.assert_array_equal(actual['spike_i'], expected['spike_i'])
            if 'synapse_samples' in actual:
                np.testing.assert_array_equal(actual['synapse_samples'][-1, 0], actual['synapse_samples'][-2, 0])
            report[name] = {'max_absolute_errors': errors, 'learning_off_gain_exact': backend != 'cpp'}
    with np.load(output/'aot-1/snapshot.npz') as one, np.load(output/'aot-4/snapshot.npz') as four:
        for field in one.files:
            np.testing.assert_array_equal(one[field], four[field])
    report['aot_1_4_exact'] = True
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', choices=['numpy', 'cpp', 'reference', 'aot'])
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args()
    child(a.child, a.output, a.threads) if a.child else run(a.output)
