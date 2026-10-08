"""Separate paper-helper rate/LvR view; does not change frozen diagnostics."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from contextlib import ExitStack

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
DTYPE = np.dtype([('tick', '<u4'), ('cell', '<u4')])


def cell_metrics(ticks, ids, neurons, *, start=5000, end=25000, dt_ms=.1, refractory_ms=2.):
    """Scalar rate uses (start,end); LvR uses [start,end) and all cells.

    Sorting by cell then time makes consecutive triples local to one neuron.
    Per-cell triples are reduced in one pass, avoiding O(neurons * events).
    """
    ticks, ids = np.asarray(ticks), np.asarray(ids)
    if (ticks.ndim != 1 or ids.shape != ticks.shape or ticks.dtype.kind not in 'iu'
            or ids.dtype.kind not in 'iu' or type(neurons) is not int or neurons <= 0
            or not 0 <= start < end or not np.isfinite(dt_ms) or dt_ms <= 0
            or not np.isfinite(refractory_ms) or refractory_ms < 0
            or np.any(ids < 0) or np.any(ids >= neurons)):
        raise ValueError('invalid physical event contract')
    selected = (ticks >= start) & (ticks < end)
    t, c = ticks[selected].astype(np.int64), ids[selected].astype(np.int64)
    counts = np.bincount(c, minlength=neurons)
    boundary = np.bincount(c[t == start], minlength=neurons)
    lvr = np.zeros(neurons)
    if len(t) >= 2:
        order = np.lexsort((t, c)); t, c = t[order], c[order]
        same = c[1:] == c[:-1]
        if np.any(same & (t[1:] <= t[:-1])):
            raise ValueError('duplicate spike timestamp for one neuron')
        triples = (c[2:] == c[:-2])
        left = (t[1:-1] - t[:-2])[triples].astype(float) * dt_ms
        right = (t[2:] - t[1:-1])[triples].astype(float) * dt_ms
        total = left + right
        # Algebraic form avoids cancellation in 1 - 4*a*b/(a+b)^2.
        terms = 3 * ((left-right)/total)**2 * (1 + 4*refractory_ms/total)
        sums = np.bincount(c[:-2][triples], weights=terms, minlength=neurons)
        eligible = counts >= 3
        lvr[eligible] = sums[eligible] / (counts[eligible]-2)
    eligible = counts >= 3
    seconds = (end-start)*dt_ms/1000
    summary = dict(neurons=neurons,half_open_spikes=int(counts.sum()),
        lower_boundary_spikes=int(boundary.sum()),strict_spikes=int((counts-boundary).sum()),
        paper_rate_hz=float((counts-boundary).sum()/neurons/seconds),
        half_open_rate_hz=float(counts.sum()/neurons/seconds),
        paper_lvr_mean=float(lvr.mean()),lvr_eligible_cells=int(eligible.sum()),
        lvr_eligible_mean=float(lvr[eligible].mean()) if eligible.any() else None)
    return summary, dict(half_open_cell_counts=counts,lower_boundary_cell_counts=boundary,cell_lvr=lvr)


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def observation_contract(window):
    end = window['end_tick']
    if (window['start_tick'] != 5000 or type(end) is not int or end not in (25000, 105000, 1005000)
            or window['dt_seconds'] != .0001 or window['endpoint'] != '[start,end)'):
        raise ValueError('requires frozen 2 s, 10 s or 100 s physical observation')
    seconds = (end - 5000) / 10000
    return end, seconds, {25000:3072000000,105000:12288000000,1005000:128*2**30}[end]


def analyze(args):
    with ExitStack() as stack:
        return _analyze(args, stack)


def _analyze(args, stack):
    bounded = getattr(args, 'bounded_memory', False)
    if bounded:
        import mam_correlation_stream as streaming
        from mam_streamed_cell_metrics import CellMetrics
        streaming.require_cache_release()
    digest_file = streaming.file_sha if bounded else file_sha
    baseline_path = args.baseline/'activity.json'
    baseline = json.loads(baseline_path.read_text())
    w = baseline['window']
    end_tick, seconds, native_byte_limit = observation_contract(w)
    if end_tick == 1005000 and not bounded:
        raise ValueError("100 s paper cell analysis requires bounded memory")
    if baseline['neurons'] != 4129924 or len(baseline['populations']) != 254:
        raise ValueError('requires complete population universe')
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {str(baseline_path): digest_file(baseline_path)}
    arrays_path = args.baseline/'activity-arrays.npz'
    catalog = json.loads((args.baseline/'catalog.json').read_text())
    for f in [baseline_path, arrays_path]:
        assert f.stat().st_size == catalog[f.name]['bytes'] and digest_file(f) == catalog[f.name]['sha256']
    sources[str(arrays_path)] = digest_file(arrays_path)
    baseline_arrays = stack.enter_context(np.load(arrays_path,allow_pickle=False))
    if args.native_audit:
        from analyze_native_mam_activity import bucket_events, canonical_population_name, simulation_identity
        audit_path = args.native_audit/'summary.json'
        audit = json.loads(audit_path.read_text())
        if not audit['passed'] or w['spike_tick_offset'] != 0:
            raise ValueError('requires verified native source')
        parameters = list((args.native_audit/'node25').glob('*/parameters.json'))
        assert len(parameters) == 1 and digest_file(parameters[0]) == baseline['parameters_sha256'] == audit['parameters_sha256']
        model = json.loads(parameters[0].read_text()); pops = model['populations']
        files, reports = [], []
        for f in sorted(args.native_audit.glob('node*/runs/*/rank*.events.bin')):
            report = json.loads(f.with_name(f.name.replace('.events.bin','.json')).read_text())
            assert report['parameters_sha256'] == baseline['parameters_sha256']
            assert f.stat().st_size == report['event_bytes'] and digest_file(f) == report['event_sha256']
            sources[str(f)] = report['event_sha256'];files.append(f);reports.append(report)
        assert simulation_identity(reports, duration_ms=end_tick//10) == baseline['simulation']
        assert sum(f.stat().st_size for f in files) <= native_byte_limit
        sources[str(audit_path)] = digest_file(audit_path)
        sources[str(parameters[0])] = digest_file(parameters[0])
        args.output.mkdir()
        paths, totals = bucket_events(pops,files,args.output/'population-events',
                                     end_tick=end_tick,max_bytes=native_byte_limit,release_file_cache=bounded)
        by_name = {canonical_population_name(p):(p,path,int(n)) for p,path,n in zip(pops,paths,totals,strict=True)}
        def events(row):
            p, path, count = by_name[row['name']]
            assert p['count'] == row['neurons'] and count == row['observed_spikes']
            if bounded:
                return lambda: streaming.native_blocks(path, DTYPE), count, path
            if count > 50000000:raise ValueError('population event budget exceeded')
            data = np.fromfile(path,dtype=DTYPE)
            return data['tick'],data['cell']
        identity = dict(simulator='NEST',**baseline['simulation'])
    else:
        spec = importlib.util.spec_from_file_location('paper_results',ROOT/'python/brian2_rust/results.py')
        reader = importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
        load_results = reader.load_results
        if not args.model or not args.results or w['spike_tick_offset'] != 1:
            raise ValueError('requires corrected Rust model and physical offset')
        assert digest_file(args.model) == baseline['model_sha256']
        for name, sha in baseline['result_sha256'].items():
            assert digest_file(args.results/name) == sha;sources[str(args.results/name)] = sha
        assert sum((args.results/n).stat().st_size for n in ['results.bin','events.bin']) < {25000:8,105000:48,1005000:256}[end_tick]*2**30
        if end_tick == 1005000 and args.model.stat().st_size >= 512*2**20:
            raise ValueError('primary model byte budget exceeded')
        model = json.loads(args.model.read_text())
        if end_tick == 1005000 and any(p['steps'] != end_tick for p in model['definition']['populations']):
            raise ValueError('primary analysis requires exact model duration')
        data = load_results(model,args.results,include_times=False,release_file_cache=bounded)
        if bounded:
            release = stack.enter_context(streaming.mapped_cache(data['_dump'], args.results/'results.bin'))
        pops = model['definition']['populations']
        assert [p['name'] for p in pops] == [r['name'] for r in baseline['populations']]
        assert [p['count'] for p in pops] == [r['neurons'] for r in baseline['populations']]
        by_name = {p['name']:v for p,v in zip(pops,data['populations'],strict=True)}
        def events(row):
            v = by_name[row['name']]
            if bounded:
                return lambda: streaming.rust_blocks(v, release), len(v['spike_ticks']), None
            if len(v['spike_ticks']) > 50000000:raise ValueError('population event budget exceeded')
            return v['spike_ticks']+1,v['indices']
        sources[str(args.model)] = digest_file(args.model)
        identity = dict(simulator='Rust',model_sha256=baseline['model_sha256'])
        args.output.mkdir()
    rows, arrays = [], {}
    for i,row in enumerate(baseline['populations']):
        if bounded:
            blocks, expected_raw, scratch = events(row)
            counter = CellMetrics(row['neurons'],end=end_tick)
            for ticks, ids in blocks():
                counter.add(ticks, ids)
            if counter.raw_events != expected_raw:
                raise ValueError('streamed population event count mismatch')
            summary, per_cell = counter.summarize()
        else:
            ticks, ids = events(row)
            summary, per_cell = cell_metrics(ticks,ids,row['neurons'],end=end_tick)
        assert summary['half_open_spikes'] == row['observed_spikes']
        np.testing.assert_allclose(summary['half_open_rate_hz'],row['mean_rate_hz'],rtol=1e-14,atol=0)
        np.testing.assert_array_equal(per_cell['half_open_cell_counts']/seconds,baseline_arrays[f'p{i}_single_cell_rates_hz'])
        sample_ids = baseline_arrays[f'p{i}_lvr_ids']
        np.testing.assert_allclose(per_cell['cell_lvr'][sample_ids],baseline_arrays[f'p{i}_lvr_values'],rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(per_cell['half_open_cell_counts'][sample_ids]>=3,baseline_arrays[f'p{i}_lvr_eligible'])
        rows.append(dict(name=row['name'],area=row['area'],group=row['group'],**summary,
            diagnostic_sampled_lvr_mean=row['lvr_zero_padded_mean'],
            diagnostic_sampled_eligible_lvr_mean=row['lvr_eligible_mean']))
        arrays.update({f'p{i}_{key}':v for key,v in per_cell.items()})
        if bounded and scratch is not None:
            scratch.unlink()  # Only this invocation's validated population scratch.
        print(json.dumps(dict(population=i,name=row['name'],paper_lvr=summary['paper_lvr_mean'])),flush=True)
    assert sum(r['half_open_spikes'] for r in rows) == baseline['observed_spikes']
    if bounded and args.native_audit:
        assert not any(path.exists() for path in paths)
        (args.output/'population-events').rmdir()
    result = dict(schema='b2-mam-paper-cell-metrics-v1',scientific_equivalence=False,
        scope=f'Only pinned scalar population-rate and all-cell LvR helper paths. {seconds:g} s diagnostic window; not paper-duration or other observable acceptance.',
        identity=identity,window=dict(start_tick=5000,end_tick=end_tick,dt_ms=.1,rate_endpoint='(start,end)',lvr_endpoint='[start,end)',refractory_ms=2),
        baseline_regression='All per-cell rates exactly match; sampled LvR matches within 1e-12 absolute/relative and eligibility exactly.',
        lvr_aggregation='All recorded neurons, including zero for <3 spikes. Eligible mean is supplemental.',
        populations=rows,source_sha256=sources,
        total_lower_boundary_spikes=sum(r['lower_boundary_spikes'] for r in rows))
    result['bounded_memory'] = bounded
    if bounded:
        dependencies = [Path(__file__),Path(__file__).with_name('mam_streamed_cell_metrics.py'),
            Path(__file__).with_name('mam_correlation_stream.py'),Path(__file__).with_name('analyze_native_mam_activity.py'),
            ROOT/'python/brian2_rust/results.py']
        result['implementation_sha256'] = {p.name: digest_file(p) for p in dependencies}
        result['scratch_removed'] = not args.native_audit or not (args.output/'population-events').exists()
    np.savez_compressed(args.output/'cell-metrics.npz',**arrays)
    (args.output/'paper-cell-metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    catalog={f.name:dict(bytes=f.stat().st_size,sha256=digest_file(f)) for f in args.output.iterdir() if f.is_file()}
    (args.output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    if args.native_audit and not bounded:
        for path in paths:path.unlink()
        (args.output/'population-events').rmdir()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['native-audit','model','results']:p.add_argument('--'+name,type=Path)
    p.add_argument('--bounded-memory',action='store_true',help='Stream per-neuron ordered events and release Linux file cache under an external resource guard.')
    args=p.parse_args()
    if args.native_audit and (args.model or args.results):p.error('choose one simulator source')
    analyze(args)
