"""Render fixed-window primary rasters and verify their activity-bin provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from litwin_kumar_figures import csv_rows


def render(raster_sources, primary_sources, output):
    import matplotlib.pyplot as plt
    manifest = json.loads((raster_sources/'manifest.json').read_text())
    runs = manifest['runs']
    keys = [(r['backend'], r['seed']) for r in runs]
    required = {(b, s) for b in ['rust', 'cpp'] for s in [20260906, 20260907, 20260908]}
    if len(keys) != len(set(keys)) or set(keys) != required:
        raise ValueError('complete unique two-backend/three-seed raster matrix required')
    data, rows, source_hashes = {}, [], {}
    for run in runs:
        backend, seed = run['backend'], run['seed']
        if (run['window_start_s'], run['window_stop_s']) != (2600., 2610.):
            raise ValueError('same declared final-10-second window required')
        file = raster_sources/run['output']
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != run['output_sha256']:
            raise ValueError('raster source file digest differs')
        with np.load(file) as source:
            arrays = {key:source[key].copy() for key in source.files}
        if {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()} != run['array_sha256']:
            raise ValueError('raster arrays differ from export manifest')
        suite = 'science' if backend == 'rust' else 'cpp-science'
        result = primary_sources/'results'/suite/f'full-seed-{seed}'/'result.json'
        if hashlib.sha256(result.read_bytes()).hexdigest() != run['result_sha256']:
            raise ValueError('raster and archived scientific run reports differ')
        analysis = 'rust-primary-analysis' if backend == 'rust' else 'cpp-science-analysis'
        activity = primary_sources/'results'/analysis/f'full-seed-{seed}'/'activity_source.npz'
        with np.load(activity) as source:
            selected = (source['time_s'] >= 2600.) & (source['time_s'] < 2610.)
            expected = source['population_rate_hz'][selected]
            if len(expected) != 200 or not np.all(source['coverage'][selected]):
                raise ValueError('complete common 50 ms activity grid required')
        ids, times = arrays['exc_i'], arrays['exc_t']
        bins = np.floor(times/.05+1e-10).astype(int)-52000
        if np.any((bins < 0) | (bins >= 200)):
            raise ValueError('spikes outside fixed raster interval')
        observed = np.bincount(bins, minlength=200)/(4000*.05)
        if not np.array_equal(observed, expected):
            raise ValueError('raster spike counts disagree with archived population rates')
        data[backend, seed] = arrays
        rows.append(dict(backend=backend, seed=seed, start_s=2600., stop_s=2610.,
                         exc_spikes=len(ids), inh_spikes=len(arrays['inh_i']),
                         population_bins=200, population_rates_byte_exact=True))
        source_hashes[str(activity.relative_to(primary_sources))] = hashlib.sha256(activity.read_bytes()).hexdigest()
    for seed in [20260906, 20260907, 20260908]:
        for key in ['membership_e', 'membership_i']:
            a, b = data['rust', seed][key], data['cpp', seed][key]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError('paired backend memberships differ')
    output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(3, 2, figsize=(7.2, 6), layout='constrained', sharex=True, sharey=True)
        for row, seed in enumerate([20260906, 20260907, 20260908]):
            for col, backend in enumerate(['rust', 'cpp']):
                ax = axes[row, col]
                a = data[backend, seed]
                members = a['membership_e']
                primary = np.where(members.any(axis=0), members.argmax(axis=0), len(members))
                order = np.argsort(primary, kind='stable')
                rank = np.empty_like(order)
                rank[order] = np.arange(len(order))
                ax.scatter(a['exc_t']-2600, rank[a['exc_i']], s=.13, color='#222222',
                           linewidths=0, rasterized=True)
                ax.set(xlim=(0, 10), ylim=(0, 4000), title=f"{'Rust' if backend == 'rust' else 'Brian2 C++'} · seed {seed}")
                if col == 0:
                    ax.set_ylabel('E neuron (ordered)')
                if row == 2:
                    ax.set_xlabel('Time from 2,600 s (s)')
        fig.suptitle('Final 10 seconds · full-scale learned networks', fontsize=11)
        fig.savefig(output/'primary_rasters.pdf')
        fig.savefig(output/'primary_rasters.png', dpi=220)
        plt.close(fig)
    csv_rows(output/'raster_counts.csv', rows)
    report = dict(runs=rows, all_1200_population_bins_exact=True,
                  paired_memberships_byte_exact=True, activity_source_sha256=source_hashes,
                  raster_manifest_sha256=hashlib.sha256((raster_sources/'manifest.json').read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'caption.md').write_text(
        'All excitatory spikes in the fixed final 10 seconds [2600,2610), displayed for three network '
        'seeds and both backends. Neurons are ordered by their first stimulus membership, followed by '
        'non-members; memberships overlap and this display order does not change the analysis. '
        'Both backends use byte-identical memberships for a given seed. The 200 population-rate bins '
        'per panel exactly reproduce the archived 50 ms scientific activity data (1,200 bins total). '
        'Different backend random streams are expected. The interval is a display convention chosen '
        'during review, not preregistered. This raster alone does not establish attractors or '
        'stationarity; quantitative activity analysis uses all 1,000 spontaneous seconds. '
        'Raw E/I spike slices and membership arrays, source file hashes and original report hashes '
        'are retained in the associated raster source archive.\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['raster-sources', 'primary-sources', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    render(**vars(parser.parse_args()))
