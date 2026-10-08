"""Compare bounded fixtures directly with hash-pinned toolbox/wrapper code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from mam_paper_correlation import candidate_histogram, summarize_histogram, HELPER_SHA, WRAPPER_SHA, TOOLBOX_COMMIT


def validate(helper, wrapper, output, *, end_tick=25000):
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == HELPER_SHA
    assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == WRAPPER_SHA
    assert end_tick in (25000,105000)
    names = ['sort_gdf_by_id', 'instantaneous_spike_count', 'strip_binned_spiketrains']
    nodes = [n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 3
    env = dict(np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(helper), 'exec'), env)
    rng = np.random.default_rng(20260909)
    fixtures = []
    # Warmup-only lowest ID, inclusive candidate right edge, and final time edge.
    fixtures.append((np.array([0, 4999, 5000, 5001, 5010, end_tick-1, end_tick, end_tick]),
                     np.array([4, 4, 5, 5, 3004, 3005, 3004, 5])))
    # A nonzero constant row must be stripped, not selected as active.
    fixtures.append((np.r_[np.arange(5000, end_tick, 10), [5000, 5010, 5020, end_tick]],
                     np.r_[np.zeros((end_tick-5000)//10, dtype=int), [1, 2, 2, 1]]))
    for cells in [1, 2, 8, 40, 2010]:
        fixtures.append((rng.integers(0, end_tick+1, size=cells * 12),
                         np.repeat(np.arange(cells), 12)))
    rows = []; maximum = 0.
    for ticks, cells in fixtures:
        first = int(cells.min())
        actual = candidate_histogram(ticks, cells, first,end_tick=end_tick)
        spikes = np.column_stack([cells, ticks * .1])
        ids, trains = env['sort_gdf_by_id'](spikes, idmin=first, idmax=first + 3000)
        bins, expected = env['instantaneous_spike_count'](trains, 1., tmin=500., tmax=end_tick/10)
        np.testing.assert_array_equal(actual, expected)
        row, selected_ids, selected = summarize_histogram(actual, first)
        original = env['strip_binned_spiketrains'](expected)[:2000]
        np.testing.assert_array_equal(selected, original)
        np.testing.assert_array_equal(selected_ids, np.asarray(ids)[np.ptp(expected, axis=1) > 0][:2000])
        if len(original) >= 2:
            cc = np.corrcoef(original)
            cc = np.extract(1 - np.eye(cc[0].size), cc)
            cc[np.where(np.isnan(cc))] = 0.
            error = abs(row['mean_pairwise_correlation'] - float(np.mean(cc)))
            assert error <= 1e-12  # Declared arithmetic tolerance, not scientific equivalence.
            maximum = max(maximum, error)
        else:
            assert not row['available'] and row['mean_pairwise_correlation'] is None
        rows.append(dict(input_events=len(ticks), selected_cells=len(selected), constant_nonzero=row['constant_nonzero_candidates'], available=row['available']))
    report = dict(passed=True, fixtures=len(rows), all_histograms_and_selected_ids_exact=True,
        end_tick=end_tick,observation_bins=(end_tick-5000)//10,
        maximum_mean_correlation_error=maximum, predeclared_absolute_arithmetic_tolerance=1e-12,
        helper_sha256=HELPER_SHA, wrapper_sha256=WRAPPER_SHA, toolbox_commit=TOOLBOX_COMMIT,
        numpy=np.__version__, cases=rows,
        scope='Available source fixtures, including 2000-cell selection. Historical dependency pin and paper-level scientific equivalence remain unproven.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['helper', 'wrapper', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--end-tick',type=int,choices=[25000,105000],default=25000)
    a = p.parse_args()
    validate(a.helper, a.wrapper, a.output,end_tick=a.end_tick)
