"""Diagnose recurrent precision differences without modifying comparison gates.

Run on Apple hardware against a completed Metal recurrent comparison directory.
The scalar f32 mirror checks device execution; independent f64 Rust execution
checks the mathematical oracle. Neither control replaces the accepted reference.
"""
import argparse
import json
from pathlib import Path

import brian2 as b
import brian2_rust
import numpy as np

from brian2_rust.metal import MetalExecutor
from gpu_baseline import canonical_result
from gpu_recurrent import arrays, oracle, DT_MS, REF_TICKS, DELAY_TICKS


def first_spike_difference(actual, expected):
    a = list(zip(actual['ticks'].tolist(), actual['indices'].tolist()))
    e = list(zip(expected['ticks'].tolist(), expected['indices'].tolist()))
    for index, (av, ev) in enumerate(zip(a, e)):
        if av != ev:
            return dict(ordinal=index, actual=av, expected=ev)
    return None if len(a) == len(e) else dict(actual_count=len(a), expected_count=len(e))


def compare(actual, expected):
    return dict(
        exact={key:bool(np.array_equal(actual[key], expected[key])) for key in expected},
        max_abs_v=float(np.max(np.abs(actual['v']-expected['v']))),
        first_spike_difference=first_spike_difference(actual, expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metal-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.metal_result/'report.json').read_text())
    config = source['config']
    if config['case'] != 'recurrent-cuba-v0':
        parser.error('a recurrent-cuba-v0 result is required')
    args.output.mkdir(parents=True, exist_ok=False)
    neurons, steps, degree = (config[key] for key in ('neurons', 'steps', 'degree'))
    model = json.loads((args.metal_result/'project/model.json').read_text())
    with MetalExecutor(model, args.output/'mirror', numeric_mode='float32', event_delivery='sparse') as executor:
        mirrored = executor.run(compute='cpu-f32', workers=1)['populations'][0]
    spikes = mirrored['event_streams']['spike']
    mirror = dict(v=mirrored['states']['v'], synaptic_current=mirrored['states']['I_syn'],
                  ticks=spikes['ticks'], indices=spikes['indices'])
    order = np.lexsort((mirror['indices'], mirror['ticks']))
    for key in ('ticks', 'indices'): mirror[key] = np.asarray(mirror[key])[order]
    actual = dict(np.load(args.metal_result/'result.npz'))
    expected = oracle(neurons, steps, degree)

    # Reconstruct exactly the declared equations, initial f32 values and edges,
    # using f64 state/parameter storage in the independent Rust CPU backend.
    b.set_device('rust_standalone', engine='aot', directory=args.output/'f64',
                 runner=Path(__file__).resolve().parents[1]/'target/release/b2-runner')
    b.prefs.core.default_float_dtype = np.float64
    b.defaultclock.dt = DT_MS*b.ms
    v, drive, projections = arrays(neurons, degree)
    pop = b.NeuronGroup(neurons,
        'dv/dt=(drive-v+I_syn)/(20*ms):1 (unless refractory)\ndI_syn/dt=-I_syn/(5*ms):1\ndrive:1 (constant)',
        threshold='v>1', reset='v=0', refractory=REF_TICKS*DT_MS*b.ms,
        method='euler', name='population')
    pop.v = v
    pop.drive = drive
    synapses = []
    for label, (sources, targets, weight) in zip(('exc', 'inh'), projections):
        syn = b.Synapses(pop, pop, 'w:1 (constant)', on_pre='I_syn_post+=w',
                         delay=DELAY_TICKS*DT_MS*b.ms, name=label)
        syn.connect(i=sources.astype(int), j=targets.astype(int))
        syn.w = float(weight)
        synapses.append(syn)
    monitor = b.SpikeMonitor(pop)
    b.Network(pop, *synapses, monitor).run(steps*DT_MS*b.ms)
    f64 = canonical_result(pop.v[:], monitor.t[:]/b.ms, monitor.i[:], DT_MS)
    f64['synaptic_current'] = np.asarray(pop.I_syn[:]).copy()
    report = dict(schema='b2-recurrent-precision-diagnosis-v0', model_sha256=source['model_sha256'],
                  config=config, metal_vs_cpu_f32=compare(actual, mirror),
                  rust_f64_vs_oracle=compare(f64, expected),
                  metal_vs_oracle=compare(actual, expected),
                  comparison_gate_unchanged=True)
    for name, data in (('cpu-f32', mirror), ('rust-f64', f64), ('oracle-f64', expected)):
        np.savez_compressed(args.output/(name+'.npz'), **data)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not all(report['metal_vs_cpu_f32']['exact'].values()):
        raise SystemExit('Metal differs from its scalar f32 mirror')
    if report['rust_f64_vs_oracle']['first_spike_difference'] is not None:
        raise SystemExit('Independent f64 spike schedule differs')


if __name__ == '__main__':
    main()
