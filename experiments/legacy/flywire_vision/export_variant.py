"""Run checked viewer trials for the input variant selected on development data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .simulation import SimulationConfig, cut_model, summarize
from .refinement import gain_model, encode_variant
from .stimuli import KINDS
from .run_experiment import save, diagnostic, array64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    read = lambda p: json.loads(p.read_text())
    protocol = read(args.study / 'protocol.json')
    report = read(args.study / 'report.json')
    if report['status'] != 'complete':
        raise ValueError('study incomplete')
    args.output.mkdir(parents=True, exist_ok=False)
    channels = read(args.artifact / 'channels.json')
    groups = {k: np.asarray(v, dtype=int) for k, v in read(args.artifact / 'groups.json').items()}
    cfg = SimulationConfig(**protocol['config'])
    template = gain_model(read(args.artifact / 'model.json'), cfg.input_weight_mv)
    models = {'intact': template, 'cut': cut_model(template, channels)}
    save(args.output / 'model.json', template)
    save(args.output / 'model-cut.json', models['cut'])
    save(args.output / 'channels.json', channels)
    save(args.output / 'groups.json', {k: v.tolist() for k, v in groups.items()})
    # Preserve the exact checked executable; parent artifact is a declared dependency.
    (args.output / 'compile').mkdir()
    (args.output / 'compile/native').symlink_to((args.artifact / 'compile/native').resolve(), target_is_directory=True)
    runners = {c: FrozenCPU(m, args.output / 'compile/native', args.output / c,
                           population='visual_input', threads=4) for c, m in models.items()}
    if runners['intact'].base_hash != protocol['base_sha256'] or runners['intact'].binary_hash != protocol['binary_sha256']:
        raise ValueError('viewer differs from the selected study variant')
    manifest = {**read(args.artifact / 'manifest.json'), 'status': 'running', 'schema': 'flywire-vision-variant-viewer-v1',
                'config': protocol['config'], 'input_mode': protocol['input_mode'],
                'input_encoding': protocol['input_mode'], 'selected_variant': report['selected_variant'],
                'scope': report['scope'], 'source_artifact': str(args.artifact.resolve()),
                'study_selection_sha256': report['selection_sha256'],
                'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    manifest.pop('prepare_seconds', None)
    manifest.pop('total_seconds', None)
    save(args.output / 'manifest.json', manifest)
    sequence = [('intact', k) for k in ('blank', 'flash', 'bright', 'dark', *KINDS, 'right')]
    sequence += [('cut', k) for k in ('blank', 'right', 'looming', 'bright', 'dark')]
    trials = []
    ni = next(i for i, p in enumerate(template['definition']['populations']) if p['name'] == 'flywire_neurons')
    for number, (condition, kind) in enumerate(sequence):
        frames = diagnostic(kind, cfg.seed)
        i, t, strength = encode_variant(frames, channels, cfg, protocol['input_mode'])
        key = f'{number:02d}-{condition}-{kind}'
        result = runners[condition].run(i, t, key)
        summary, counts = summarize(result, models[condition], groups, channels, cfg)
        if np.any(counts > 65535):
            raise ValueError('display count overflow')
        summary['final_state_sha256'] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in result['populations'][ni]['states'].items()}
        trial = {'id': key, 'condition': condition, 'kind': kind, 'seed': cfg.seed, 'summary': summary,
                 'input_spikes': len(i), 'input_sha256': hashlib.sha256(np.stack([t, i], 1).astype('<i8').tobytes()).hexdigest(),
                 'movie_sha256': hashlib.sha256(frames.tobytes()).hexdigest(),
                 'frames_u8': array64(np.rint(frames * 255), 'u1'), 'frame_shape': list(frames.shape),
                 'strength_u8': array64(np.rint(strength * 255), 'u1'),
                 'input_counts_u16': array64(counts, '<u2'), 'count_shape': list(counts.shape)}
        trials.append(trial)
        save(args.output / (key + '.json'), trial)
        print(json.dumps({'trial': key, 'spikes': summary['total_spikes']}), flush=True)
    by_key = {(t['condition'], t['kind']): t for t in trials}
    right = [t for t in trials if t['condition'] == 'intact' and t['kind'] == 'right']
    checks = {'A_B_A_events_and_final_states_exact': all(right[0]['summary'][k] == right[1]['summary'][k] for k in ('neural_event_sha256', 'final_state_sha256')),
              'cut_downstream_equals_cut_blank': {k: by_key['cut', k]['summary']['noninput_event_sha256'] == by_key['cut', 'blank']['summary']['noninput_event_sha256'] for k in ('right', 'looming', 'bright', 'dark')},
              'cut_preserves_external_input': {k: by_key['cut', k]['input_sha256'] == by_key['intact', k]['input_sha256'] for k in ('right', 'looming', 'bright', 'dark')}}
    passed = checks['A_B_A_events_and_final_states_exact'] and all(checks['cut_downstream_equals_cut_blank'].values()) and all(checks['cut_preserves_external_input'].values())
    manifest.update(status='complete' if passed else 'failed_checks', checks=checks, trial_count=len(trials))
    save(args.output / 'checks.json', checks)
    save(args.output / 'manifest.json', manifest)
    save(args.output / 'viewer-data.json', {'manifest': manifest, 'channels': channels, 'trials': trials})
    if not passed:
        raise RuntimeError('variant checks failed')


if __name__ == '__main__':
    main()
