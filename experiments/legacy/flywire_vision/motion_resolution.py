"""Replay development data at 10 ms resolution; never load held-out features."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .refinement import encode_variant
from .simulation import SimulationConfig
from .motion_challenge import movie
from .run_experiment import save


def temporal_counts(indices, ticks, cells, neurons):
    lookup = np.full(neurons, -1, dtype=int)
    lookup[cells] = np.arange(len(cells))
    slots = lookup[np.asarray(indices)]
    ticks = np.asarray(ticks)
    mask = (slots >= 0) & (ticks >= 0) & (ticks < 6000)
    result = np.zeros((60, len(cells)), dtype=np.uint16)
    np.add.at(result, (ticks[mask] // 100, slots[mask]), 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'parent', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    p = read(args.parent/'protocol.json')
    parent_rows = read(args.parent/'rows.json')
    rows = [r for r in parent_rows if r['split'] in ('fit', 'validation')]
    if len(rows) != 192 or any(r['condition'] != 'intact' for r in rows):
        raise ValueError('requires the original development orbits only')
    with np.load(args.parent/'features.npz') as f:
        coarse = f['neural_linear'][:192]
    template = read(args.artifact/'model.json')
    ni = next(i for i, pop in enumerate(template['definition']['populations']) if pop['name'] == 'flywire_neurons')
    neurons = template['definition']['populations'][ni]['count']
    cells = np.array(p['readout_indices']); cfg = SimulationConfig(**p['config'])
    channels = read(args.artifact/'channels.json')
    args.output.mkdir(exist_ok=False, parents=True)
    protocol = {'schema': 'flywire-motion-resolution-v1', 'parent': str(args.parent),
                'parent_protocol_sha256': sha(args.parent/'protocol.json'),
                'source_sha256': sha(Path(__file__)), 'bin_ms': 10, 'bins': 60,
                'fit_groups': p['fit_groups'], 'validation_groups': p['validation_groups'],
                'readout_indices': p['readout_indices'], 'input_indices': p['input_indices'],
                'scope': 'development replay only; same model, input and events; finer bins include the response tail'}
    save(args.output/'protocol.json', protocol)
    runner = FrozenCPU(template, args.artifact/'compile/native', args.output/'execution', population='visual_input', threads=4)
    identity = read(args.parent/'identities.json')['intact']
    if runner.base_hash != identity['base_sha256'] or runner.binary_hash != identity['binary_sha256']:
        raise ValueError('neural model changed')
    records = []; neural = []; inputs = []; start = time.perf_counter()
    for idx, row in enumerate([None]+rows):
        frames = np.full((40,48,48), .5, dtype=np.float32) if row is None else movie(row['kind'], row['group'], p['travel'], tuple(row['phase_index']))
        i, t, _ = encode_variant(frames, channels, cfg, p['input_mode'])
        key = 'blank' if row is None else f'development-{idx-1}'
        result = runner.run(i, t, key); pop = result['populations'][ni]
        counts = temporal_counts(pop['indices'], pop['spike_ticks'], cells, neurons)
        encoded = temporal_counts(i, t, np.arange(len(channels)), len(channels))
        events_hash = hashlib.sha256(np.stack([pop['spike_ticks'],pop['indices']],1).astype('<i8').tobytes()).hexdigest()
        input_hash = hashlib.sha256(np.stack([t,i],1).astype('<i8').tobytes()).hexdigest()
        if row is not None:
            if events_hash != row['events_sha256'] or input_hash != row['input_sha256']:
                raise ValueError('development replay changed actual events')
            np.testing.assert_array_equal(counts[10:50].reshape(8,5,-1).sum(1).ravel(), coarse[idx-1])
            neural.append(counts); inputs.append(encoded)
            records.append({**row, 'fine_events_sha256': events_hash})
        else:
            np.savez_compressed(args.output/'blank.npz', neural=counts, encoded=encoded)
        del result, pop
        shutil.rmtree(runner.directory/key); (runner.directory/(key+'.spikes')).unlink()
        if idx and idx % 16 == 0:
            np.savez_compressed(args.output/'features.npz', neural=np.asarray(neural), encoded=np.asarray(inputs))
            save(args.output/'rows.json', records)
            print(json.dumps({'completed': idx, 'total':192, 'seconds':time.perf_counter()-start}), flush=True)
    save(args.output/'report.json', {'status':'complete', 'native_runs':193, 'events_exact':True,
         'coarse_counts_exact':True, 'test_used':False, 'features_sha256':sha(args.output/'features.npz'),
         'protocol_sha256':sha(args.output/'protocol.json'), 'seconds':time.perf_counter()-start})


if __name__ == '__main__':
    main()
