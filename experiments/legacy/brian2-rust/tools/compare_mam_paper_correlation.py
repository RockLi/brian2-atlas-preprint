"""Descriptive old/new sampling comparison; no scientific acceptance test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import re

CASES = ['native-meta1729', 'native-meta1730', 'native-meta1731',
         'rust-meta1729', 'native-ground1729', 'rust-ground1729']


def compare(artifacts, output, *, cases=None):
    cases=list(CASES if cases is None else cases)
    if not 1<=len(cases)<=6 or len(set(cases))!=len(cases) or any(not re.fullmatch(r'[a-z0-9-]{1,64}',case) for case in cases):
        raise ValueError('requires one to six distinct explicit case names')
    reports, sources = {}, {}
    for case in cases:
        folder = artifacts / ('mam-paper-correlation-' + case + '-v1')
        catalog = json.loads((folder / 'catalog.json').read_text())
        path = folder / 'correlation.json'
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == catalog['correlation.json']['sha256'] and len(raw) == catalog['correlation.json']['bytes']
        reports[case] = json.loads(raw)
        sources[str(path)] = digest
    reference=reports[cases[0]]
    assert reference['observation_ms'] in ([500,2500],[500,10500],[500,100500])
    names = [p['name'] for p in reference['populations']]
    rows = []
    for case in cases:
        report = reports[case]
        assert [p['name'] for p in report['populations']] == names
        assert report['frozen_histograms_exact'] and report['scratch_removed']
        for key in ['observation_ms','endpoint','bin_ms','selection','calculation','helper_sha256','wrapper_sha256','toolbox_commit']:
            assert report[key]==reference[key]
        paired = [p for p in report['populations'] if p['available'] and p['frozen_uniform_correlation'] is not None]
        ranked = sorted(paired, key=lambda p: abs(p['mean_pairwise_correlation'] - p['frozen_uniform_correlation']), reverse=True)
        rows.append(dict(case=case, available=report['available_populations'],
            unavailable=[dict(name=p['name'], selected=p['selected_cells'], raw_events=p['raw_events'], frozen_correlation=p['frozen_uniform_correlation']) for p in report['populations'] if not p['available']],
            selected_cells_min=min(p['selected_cells'] for p in report['populations']),
            below_2000_cells=sum(p['selected_cells'] < 2000 for p in report['populations']),
            largest_sampling_view_differences=[dict(name=p['name'], new=p['mean_pairwise_correlation'], frozen=p['frozen_uniform_correlation'], difference=p['mean_pairwise_correlation']-p['frozen_uniform_correlation'], selected=p['selected_cells']) for p in ranked[:10]]))
    output.mkdir(exist_ok=False)
    result = dict(schema='b2-mam-modern-paper-correlation-comparison-v1', scientific_equivalence=False,
        cases=rows, source_sha256=sources,
        scope='Same-run descriptive sampling/endpoint convention differences. Inadequate sample populations are unavailable, never imputed as zero. Cross-simulator independent realizations and paper-duration acceptance remain open.')
    if cases!=CASES or reference['observation_ms']!=[500,2500]:
        result.update(observation_ms=reference['observation_ms'],cohort=[dict(case=case,identity=reports[case]['identity']) for case in cases])
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns = min(2, len(cases))
    plot_rows = (len(cases) + columns - 1) // columns
    fig, axes = plt.subplots(plot_rows, columns, figsize=(6 * columns, 4 * plot_rows),
                             squeeze=False, layout='constrained')
    for ax, case, row in zip(list(axes.flat)[:len(cases)], cases, rows, strict=True):
        pops = [p for p in reports[case]['populations'] if p['available'] and p['frozen_uniform_correlation'] is not None]
        x = np.array([p['frozen_uniform_correlation'] for p in pops])
        y = np.array([p['mean_pairwise_correlation'] for p in pops])
        ax.scatter(x, y, s=10, alpha=.6)
        extent = [min(x.min(), y.min()), max(x.max(), y.max())] if len(x) else [-1,1]
        ax.plot(extent, extent, color='gray', ls='--', lw=1)
        ax.set(title=f"{case}: {row['available']}/254 available", xlabel='Frozen uniform-sample correlation', ylabel='Modern ID-interval correlation')
        ax.grid(alpha=.2)
        if row['largest_sampling_view_differences']:
            largest = row['largest_sampling_view_differences'][0]
            ax.annotate(largest['name'].removeprefix('mam_'), (largest['frozen'], largest['new']), fontsize=8, xytext=(5, 5), textcoords='offset points')
    for ax in axes.flat[len(cases):]:ax.set_visible(False)
    fig.suptitle('Full MAM: sampling/endpoint convention differences within each run\nUnavailable populations excluded; short observations do not establish scientific equivalence' if reference['observation_ms']==[500,2500] else
                 f'Full MAM: sampling/endpoint convention differences within each run\n{(reference["observation_ms"][1]-500)/1000:g} s observations; unavailable populations excluded; scientific equivalence unproven')
    fig.savefig(output / 'sampling-comparison.png', dpi=140)
    plt.close(fig)
    catalog = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in output.iterdir()}
    (output / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case',action='append',dest='cases',help='Explicit case suffix; repeat up to six times. Defaults to the frozen six-run short cohort.')
    args = p.parse_args()
    compare(args.artifacts, args.output,cases=args.cases)
