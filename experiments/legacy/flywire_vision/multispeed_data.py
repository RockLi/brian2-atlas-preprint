"""Full-orbit closed movies and auditable full-graph development execution."""
import argparse
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
from .motion_challenge import movie, centers, render
from .motion_refinement import read, sha, load_npz
from .motion_resolution import temporal_counts
from .refinement import encode_variant
from .simulation import SimulationConfig
from .run_experiment import save


def speed_movie(kind, group, phase, speed):
    if type(speed) is not int or not 1 <= speed <= 4:
        raise ValueError('integer cycles 1 through 4 required')
    if speed == 1:
        return movie(kind, group, 2., tuple(phase))
    if kind in ('left', 'down'):
        return speed_movie({'left': 'right', 'down': 'up'}[kind], group, phase, speed)[::-1].copy()
    start = centers(kind, group, 2., tuple(phase))[0]
    position = np.repeat(start[None], 40, axis=0)
    axis, sign = (0, 1) if kind == 'right' else (1, -1)
    position[:, axis] += sign*2*speed*np.linspace(0, 1, 40)
    position = (position+1) % 2-1
    position[0] = position[-1] = start
    return render(position, group)


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


class NativeRecorder:
    def __init__(self, artifact, directory, parent, template=None, identity='intact'):
        from flywire_mnist.backends.frozen_cpu import FrozenCPU
        model = read(artifact/'model.json') if template is None else template
        self.channels = read(artifact/'channels.json')
        self.protocol = read(parent/'protocol.json')
        self.config = SimulationConfig(**self.protocol['config'])
        self.ni = next(i for i, p in enumerate(model['definition']['populations']) if p['name'] == 'flywire_neurons')
        self.neurons = model['definition']['populations'][self.ni]['count']
        self.cells = np.array(self.protocol['readout_indices'])
        self.runner = FrozenCPU(model, artifact/'compile/native', directory, population='visual_input', threads=4)
        expected = read(parent/identity/'identity.json')
        if self.runner.base_hash != expected['base_sha256'] or self.runner.binary_hash != expected['binary_sha256']:
            raise ValueError('native model changed')
        self.runs = 0

    def run(self, frames, key):
        i, t, _ = encode_variant(frames, self.channels, self.config, self.protocol['input_mode'])
        result = self.runner.run(i, t, key)
        pop = result['populations'][self.ni]
        neural = temporal_counts(pop['indices'], pop['spike_ticks'], self.cells, self.neurons)
        encoded = temporal_counts(i, t, np.arange(len(self.channels)), len(self.channels))
        record = {'movie_sha256': digest(frames), 'input_sha256': digest(np.stack([t, i], 1).astype('<i8')),
                  'events_sha256': digest(np.stack([pop['spike_ticks'], pop['indices']], 1).astype('<i8')),
                  'states_sha256': {k: digest(v) for k, v in pop['states'].items()},
                  'external_spikes': len(i), 'native_key': key}
        del result, pop
        shutil.rmtree(self.runner.directory/key)
        (self.runner.directory/(key+'.spikes')).unlink()
        self.runs += 1
        return neural, encoded, record


def development(args):
    args.output.mkdir(exist_ok=False, parents=True)
    previous = read(args.development/'report.json')
    if previous['status'] != 'complete' or sha(args.development/'features.npz') != previous['features_sha256']:
        raise ValueError('original development provenance changed')
    rows = read(args.development/'rows.json')
    if len(rows) != 192 or any(r['split'] != ('fit' if i < 128 else 'validation') for i, r in enumerate(rows)):
        raise ValueError('development rows only')
    save(args.output/'protocol.json', {'schema': 'flywire-multispeed-development-v1',
         'parent': str(args.parent), 'parent_protocol_sha256': sha(args.parent/'protocol.json'),
         'development': str(args.development), 'development_report_sha256': sha(args.development/'report.json'),
         'speed': 2, 'source_sha256': sha(Path(__file__)), 'test_used': False})
    recorder = NativeRecorder(args.artifact, args.output/'execution', args.parent)
    blank, encoded, record = recorder.run(np.full((40,48,48), .5, dtype=np.float32), 'blank')
    np.testing.assert_array_equal(blank, load_npz(args.development/'blank.npz')['neural'])
    np.savez_compressed(args.output/'blank.npz', neural=blank, encoded=encoded)
    save(args.output/'blank-events.json', record)
    neural, inputs, records = [], [], []
    start = time.perf_counter()
    for idx, row in enumerate(rows):
        frames = speed_movie(row['kind'], row['group'], row['phase_index'], 2)
        nn, ee, rr = recorder.run(frames, f'development-{idx}')
        neural.append(nn); inputs.append(ee)
        records.append({**row, **rr, 'speed': 2})
        if (idx+1) % 16 == 0:
            np.savez_compressed(args.output/'features.npz', neural=np.array(neural), encoded=np.array(inputs))
            save(args.output/'rows.json', records)
            print({'completed': idx+1, 'native_runs': recorder.runs, 'seconds': time.perf_counter()-start}, flush=True)
    save(args.output/'report.json', {'status': 'complete', 'native_runs': recorder.runs, 'test_used': False,
         'protocol_sha256': sha(args.output/'protocol.json'), 'rows_sha256': sha(args.output/'rows.json'),
         'features_sha256': sha(args.output/'features.npz'), 'seconds': time.perf_counter()-start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'parent', 'development', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    development(parser.parse_args())
