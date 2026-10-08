"""Isolate delayed-STDP decay rounding in the pinned Brian2GeNN adapter.

Diagnostic only: the existing comparator and its qualification gate are unchanged.
Time and exp remain double; the variant rounds the decay factor to float before
multiplication, matching the explicitly selected f32 recurrence boundary.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from gpu_stdp_compare import DT, FIELDS, checks, configuration, oracle, topology
from gpu_brian2genn_stdp_adapter import round_decay_factor

OPTIONS = dict(drive=1/16, delay_span=8, post_delay=3,
               topology_kind='random-fixed-outdegree', topology_seed=42)
SIZE = (1024, 128, 1024)
POLICY = 'brian2genn-1.7.0-stdp-exp-factor-rounding-diagnostic-v1'


def plasticity_from_spikes(neurons, degree, steps, ticks, indices, *, mode, **options):
    """Replay edge-local plasticity with observed spikes; does not qualify neurons.

    The two modes differ solely in whether exp's double result is rounded before
    multiplying a float trace or the double product is rounded at assignment.
    """
    if mode not in {'f32-factor', 'double-product'}:
        raise ValueError('Unknown decay mode')
    ticks, indices = np.asarray(ticks), np.asarray(indices)
    if (ticks.ndim != 1 or ticks.shape != indices.shape or
            ticks.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu' or
            np.any(ticks < 0) or np.any(ticks >= steps) or
            np.any(indices < 0) or np.any(indices >= neurons)):
        raise ValueError('Invalid spike coordinates')
    pairs = np.stack((ticks, indices), axis=1)
    if len(np.unique(pairs, axis=0)) != len(ticks):
        raise ValueError('Duplicate neuron spikes in one tick')
    source, target, delay = topology(neurons, degree,
        **{k: options[k] for k in ('delay_span', 'topology_kind', 'topology_seed')})
    history = np.zeros((steps, neurons), bool)
    history[ticks, indices] = True
    apre = np.zeros(len(source), np.float32)
    apost = apre.copy()
    weight = np.full(len(source), .25, np.float32)
    last = np.zeros(len(source), np.float64)
    post_delay = options['post_delay']
    for tick in range(steps):
        pre = np.flatnonzero((tick >= delay) & history[np.maximum(tick-delay, 0), source])
        post = (np.flatnonzero(history[tick-post_delay, target]) if tick >= post_delay
                else np.empty(0, np.int64))
        # Independent edges; each edge's pre still precedes its post this tick.
        for events, is_pre in ((pre, True), (post, False)):
            elapsed = tick*DT-last[events]
            for trace, tau in ((apre, 16*DT), (apost, 32*DT)):
                factor = np.exp(-elapsed/tau)
                trace[events] = (trace[events] * factor.astype(np.float32)
                    if mode == 'f32-factor' else trace[events].astype(np.float64) * factor)
            if is_pre:
                apre[events] += np.float32(.0078125)
                weight[events] = np.clip(weight[events]+apost[events], 0, .5)
            else:
                apost[events] -= np.float32(.00390625)
                weight[events] = np.clip(weight[events]+apre[events], 0, .5)
            last[events] = tick*DT
    return dict(w=weight, Apre=apre, Apost=apost, lastupdate=last)


def retain_program(project, output):
    """Retain actual generated GPU source and binaries before the next build."""
    hashes = {}
    for path in sorted(project.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix not in {'.cpp', '.cc', '.cu', '.h', '.hpp', '.so'} and path.name not in {'main', 'Makefile'}:
            continue
        relative = path.relative_to(project)
        destination = output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def diagnose(output):
    import gpu_brian2genn_stdp_adapter as adapter
    output.mkdir(parents=True, exist_ok=False)
    config = configuration(*SIZE, **OPTIONS)
    report = dict(policy=POLICY, configuration=config, variants={}, passed=False,
        scope='Same generated model; decay-factor cast only. No throughput ranking.')
    reference = oracle(*SIZE, dtype=np.float32, **OPTIONS)
    np.savez_compressed(output/'reference-f32.npz', **reference)
    context = {}
    original_result, _ = adapter.run(*SIZE, output, prepared=context, **OPTIONS)
    project = context['project']
    model_path = project/'magicnetwork_model.cpp'
    before = model_path.read_text()
    after = round_decay_factor(before, OPTIONS['delay_span'])
    (output/'factor-rounding.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='generated', tofile='f32-factor')))
    for variant in ('generated', 'f32-factor'):
        directory = output/variant
        directory.mkdir()
        if variant == 'f32-factor':
            # Regenerate and compile the exact same model after the sole change.
            model_path.write_text(after)
            context['device'].compile_source(debug=False, directory=str(project), use_GPU=True)
        program = retain_program(project, directory/'program')
        rows = []
        for repeat in range(3):
            if variant == 'generated' and repeat == 0:
                actual = original_result
            else:
                actual, _ = adapter.replay(context, variant+'-'+str(repeat))
            np.savez_compressed(directory/f'result-{repeat}.npz', **actual)
            modes = {}
            for mode in ('f32-factor', 'double-product'):
                predicted = plasticity_from_spikes(*SIZE, actual['ticks'], actual['indices'], mode=mode, **OPTIONS)
                if repeat == 0:
                    np.savez_compressed(directory/(mode+'-prediction.npz'), **predicted)
                modes[mode] = {key: dict(exact=bool(np.array_equal(predicted[key], actual[key])),
                    max_abs=float(np.max(np.abs(predicted[key]-actual[key]), initial=0))) for key in FIELDS}
            rows.append(dict(reference_f32=checks(actual, reference), predictions=modes))
        report['variants'][variant] = dict(program_hashes=program, rows=rows)
        (output/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
    generated, corrected = (report['variants'][key]['rows'] for key in ('generated', 'f32-factor'))
    report['passed'] = (
        all(not row['reference_f32']['fields']['w']['passed'] and
            all(value['exact'] for value in row['predictions']['double-product'].values()) for row in generated)
        and all(row['reference_f32']['passed'] and
            all(value['exact'] for value in row['predictions']['f32-factor'].values()) for row in corrected))
    (output/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], policy=POLICY)), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if not diagnose(args.output)['passed']:
        raise SystemExit(1)
