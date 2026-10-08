"""Verify complete post-checkpoint spike trajectories and segment diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify_trajectory(original, restored, checkpoint_time_s):
    reports = [json.loads((path/'result.json').read_text()) for path in [original, restored]]
    before, after = reports
    if (before['configuration'] != after['configuration']
            or after['initial_time_seconds'] != checkpoint_time_s
            or before['biological_seconds'] != after['biological_seconds']
            or after['biological_seconds'] <= checkpoint_time_s):
        raise ValueError('matching configuration, restoration clock and final time required')
    selected = [[(index, segment) for index, segment in enumerate(report['segments'], 1)
                 if segment['stop_s'] > checkpoint_time_s]
                for report in reports]
    boundaries = [[(segment['start_s'], segment['stop_s']) for _, segment in rows]
                  for rows in selected]
    if not boundaries[0] or boundaries[0] != boundaries[1]:
        raise ValueError('matching complete post-checkpoint segments required')
    expected_start = checkpoint_time_s
    for start, stop in boundaries[0]:
        if start != expected_start or stop <= start:
            raise ValueError('post-checkpoint segments must be contiguous')
        expected_start = stop
    if expected_start != after['biological_seconds']:
        raise ValueError('post-checkpoint segments do not reach the final time')
    dt = before['configuration']['dt_ms']/1000
    for report in reports:
        window = report['recording_window_steps']
        if window is not None and any(stop-start > window*dt+1e-10 for start, stop in boundaries[0]):
            raise ValueError('retained spike windows do not cover complete segments')
    checks = []
    for (index_a, segment), (index_b, _) in zip(*selected):
        paths = [original/'segments'/f'{index_a:04d}'/'spikes.npz',
                 restored/'segments'/f'{index_b:04d}'/'spikes.npz']
        hashes, counts = {}, {}
        with np.load(paths[0]) as a, np.load(paths[1]) as b:
            fields = {'exc_i', 'exc_t', 'inh_i', 'inh_t'}
            if set(a.files) != fields or set(b.files) != fields:
                raise ValueError('complete excitatory/inhibitory spike arrays required')
            for name in sorted(fields):
                x, y = a[name], b[name]
                if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                    raise AssertionError(f'spike trajectory differs: segment {index_a}/{name}')
                hashes[name] = hashlib.sha256(x.tobytes()).hexdigest()
            for label in ['exc', 'inh']:
                times, ids = a[label+'_t'], a[label+'_i']
                if times.ndim != 1 or ids.shape != times.shape or not np.isfinite(times).all():
                    raise ValueError('consistent finite spike arrays required')
                counts[label] = int(np.count_nonzero(
                    (times >= segment['start_s']-1e-10) & (times < segment['stop_s']-1e-10)))
        diagnostics = [report['trajectory'][index-1]
                       for report, index in zip(reports, [index_a, index_b])]
        if any(row['time_s'] != segment['stop_s'] for row in diagnostics):
            raise ValueError('trajectory diagnostics must align with segment endpoints')
        canonical = [json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)
                     for row in diagnostics]
        if canonical[0] != canonical[1]:
            raise AssertionError(f'weight/spike-total diagnostics differ: segment {index_a}')
        checks.append({'start_s': segment['start_s'], 'stop_s': segment['stop_s'],
                       'spike_array_sha256': hashes, 'spikes_in_segment': counts,
                       'diagnostics_sha256': hashlib.sha256(canonical[0].encode()).hexdigest()})
    return {'checkpoint_time_s': checkpoint_time_s,
            'verified_seconds': after['biological_seconds']-checkpoint_time_s,
            'all_post_checkpoint_spike_arrays_byte_exact': True,
            'all_segment_weight_and_spike_total_diagnostics_exact': True,
            'segments': checks,
            'scope': 'Every newly simulated segment after restoration; copied pre-checkpoint history is excluded. '
                     'All retained E/I spike arrays and every segment weight/spike-total diagnostic match. '
                     'Final dynamic-state verification is recorded separately; unrecorded voltage histories are not asserted.'}


def review(jobs, output, allow_incomplete=False):
    entries = json.loads(jobs.read_text())
    if len(entries) != 3 or len({row['seed'] for row in entries}) != 3:
        raise ValueError('three unique recovery seeds required')
    checks, pending = [], []
    for job in entries:
        original, restored = Path(job['original']), Path(job['destination'])
        if not (restored/'result.json').exists():
            if allow_incomplete:
                pending.append(job['seed'])
                continue
            raise ValueError(f'incomplete recovery: seed {job["seed"]}')
        checkpoint = Path(job['checkpoint'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != job['checkpoint_sha256']:
            raise ValueError('original training checkpoint changed')
        run = json.loads((restored/'result.json').read_text())
        if (run['backend'] != 'rust' or run['threads'] != 8
                or run['configuration']['scale'] != 1 or run['configuration']['mode'] != 'learn'
                or run['biological_seconds'] != 2610.):
            raise ValueError('complete full-scale eight-worker recovery required')
        checks.append({'seed': job['seed'], **verify_trajectory(original, restored, 1610.)})
    report = {'all_three_complete_trajectories_exact': len(checks) == 3 and not pending,
              'pending_seeds': pending, 'checks': checks,
              'jobs_sha256': hashlib.sha256(jobs.read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['jobs', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    review(**vars(parser.parse_args()))
