"""Descriptive six-run rate/PSD comparison, with no equivalence threshold."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import re
import zipfile

CASES = ['native-meta1729', 'native-meta1730', 'native-meta1731',
         'rust-meta1729', 'native-ground1729', 'rust-ground1729']


def compare(artifacts, output, *, cases=None):
    cases=list(CASES if cases is None else cases)
    if not 1<=len(cases)<=6 or len(set(cases))!=len(cases) or any(not re.fullmatch(r'[a-z0-9-]{1,64}',case) for case in cases):
        raise ValueError('requires one to six distinct explicit case names')
    reports, arrays, hashes = {}, {}, {}
    for case in cases:
        path = artifacts / ('mam-paper-series-' + case + '-v1')
        catalog = json.loads((path / 'catalog.json').read_text())
        for name in ['time-series.json', 'time-series.npz']:
            file = path / name
            with file.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert digest == catalog[name]['sha256']
            assert file.stat().st_size == catalog[name]['bytes']
            hashes[str(file)] = digest
        reports[case] = json.loads((path / 'time-series.json').read_text())
        assert (path/'time-series.npz').stat().st_size <= 256*2**20
        with zipfile.ZipFile(path/'time-series.npz') as archive:
            entries=archive.infolist()
            assert len(entries)<=12 and sum(row.file_size for row in entries)<=512*2**20
        with np.load(path / 'time-series.npz', allow_pickle=False) as data:
            arrays[case] = {name: data[name] for name in
                            ['area_rates_hz', 'frequency_hz', 'power_hz2_per_hz']}
    reference = reports[cases[0]]
    areas = reference['area_names']
    frequency = arrays[cases[0]]['frequency_hz']
    nbins=arrays[cases[0]]['area_rates_hz'].shape[1]
    assert nbins in (2000,10000,100000) and len(areas)==32
    duration=nbins/1000
    assert reference['wrapper_window_ms']==f'(500,{int(500+duration*1000)}]'
    for case, report in reports.items():
        assert report['frozen_histogram_exact'] and report['endpoint_count_identity_exact']
        for key in ['area_names', 'population_names', 'welch', 'unrounded_population_sum','wrapper_window_ms','helper_histogram_range_ms','bin_ms','welch_segments','normalization','area_weighting']:
            assert report[key] == reference[key]
        np.testing.assert_array_equal(arrays[case]['frequency_hz'], frequency)
        assert arrays[case]['area_rates_hz'].shape==(32,nbins) and arrays[case]['power_hz2_per_hz'].shape==(32,513)
    output.mkdir(exist_ok=False)
    rows = []
    for i, area in enumerate(areas):
        row = dict(area=area, runs={})
        for case in cases:
            rate = arrays[case]['area_rates_hz'][i]
            power = arrays[case]['power_hz2_per_hz'][i]
            row['runs'][case] = dict(mean_rate_hz=float(rate.mean()),
                **{('first_second_mean_hz' if nbins==2000 else 'first_half_mean_hz'):float(rate[:nbins//2].mean()),
                   ('second_second_mean_hz' if nbins==2000 else 'second_half_mean_hz'):float(rate[nbins//2:].mean())},
                peak_nonzero_frequency_hz=float(frequency[1 + np.argmax(power[1:])]),
                integrated_psd=float(power.sum() * (frequency[1] - frequency[0])))
        rows.append(row)
    report = dict(schema='b2-mam-modern-paper-series-comparison-v1',
        scientific_equivalence=False, source_sha256=hashes, areas=rows,
        scope=('Descriptive independent realizations: three NEST metastable, one Rust metastable, one NEST and one Rust ground. The 2 s observations and overlapping Welch segments are not independent replicates. No fitted thresholds, historical-environment match, or scientific acceptance claim.' if cases==CASES and nbins==2000 else
               f'Descriptive comparison of {len(cases)} explicitly selected realizations over {duration:g} s. Windows and normalization match; condition/input semantics need their separate audits. Time bins and overlapping Welch segments are not independent replicates. No fitted thresholds, historical-environment match, or scientific acceptance claim.'))
    if cases!=CASES or nbins!=2000:
        report.update(observation_seconds=duration,cohort=[dict(case=case,identity=reports[case]['identity']) for case in cases])
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    selected = ['V1', 'V2', 'MT', 'FEF', 'MIP', 'PITd']
    colors = ['#4e79a7', '#76b7b2', '#b07aa1', '#e15759', '#59a14f', '#f28e2b']
    for ax, area in zip(axes.flat, selected, strict=True):
        i = areas.index(area)
        for case, color in zip(cases, colors[:len(cases)], strict=True):
            power = arrays[case]['power_hz2_per_hz'][i]
            ax.semilogy(frequency[1:], np.maximum(power[1:], 1e-20),
                        color=color, label=case, lw=1,
                        ls='--' if 'ground' in case else '-')
        ax.set(title=area, xlim=(0, 150), xlabel='Frequency (Hz)',
               ylabel='PSD ((Hz/neuron)^2 / Hz)')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f'Full MAM: modern rate/PSD convention, {duration:g} s observation\nIndependent realizations; scientific equivalence remains unproven')
    fig.savefig(output / 'spectra-comparison.png', dpi=140)
    for ax in axes.flat:
        ax.set_xlim(0, 500)
    fig.suptitle(f'Full MAM: complete 0–500 Hz PSD, {duration:g} s observation\nIndependent realizations; peaks outside the 0–150 Hz view remain visible')
    fig.savefig(output / 'spectra-full-band.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), layout='constrained')
    for i, case in enumerate(cases):
        ax = axes.flat[i]
        for area in ['V1', 'MIP', 'PITd']:
            rate = arrays[case]['area_rates_hz'][areas.index(area)]
            # Plot 10 ms block means; stored 1 ms arrays remain unchanged.
            ax.plot(.5055 + np.arange(nbins//10) * .01, rate.reshape(nbins//10, 10).mean(axis=1),
                    label=area, lw=1)
        ax.set(title=case, xlabel='Biological time (s)', ylabel='Rate (Hz/neuron)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(cases):]:ax.set_visible(False)
    fig.suptitle('Area-rate patterns: 10 ms display averages of retained 1 ms bins\nModern official unrounded population normalization')
    fig.savefig(output / 'rates-comparison.png', dpi=140)
    plt.close(fig)
    catalog = {}
    for file in output.iterdir():
        with file.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        catalog[file.name] = dict(bytes=file.stat().st_size, sha256=digest)
    (output / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), compared_runs=len(cases), areas=len(areas))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case',action='append',dest='cases',help='Explicit case suffix; repeat up to six times. Defaults to the frozen six-run short cohort.')
    args = parser.parse_args()
    compare(args.artifacts, args.output,cases=args.cases)
