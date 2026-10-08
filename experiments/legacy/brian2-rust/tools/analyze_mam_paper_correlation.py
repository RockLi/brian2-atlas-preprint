"""Apply the pinned modern correlation selection to retained full raw events."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from contextlib import nullcontext
import numpy as np
from mam_paper_correlation import candidate_histogram, summarize_histogram, HELPER_SHA, WRAPPER_SHA, TOOLBOX_COMMIT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def analyze(args):
    start = time.perf_counter()
    bounded = getattr(args, 'bounded_memory', False)
    if bounded:
        import mam_correlation_stream as streaming
        streaming.require_cache_release()
    digest_file = streaming.file_sha if bounded else sha
    base = json.loads((args.baseline / 'activity.json').read_text())
    if base['window']['end_tick'] == 1005000 and not bounded:
        raise ValueError('100 s paper correlation requires bounded memory')
    catalog = json.loads((args.baseline / 'catalog.json').read_text())
    sources = {}
    for name in ['activity.json', 'activity-arrays.npz']:
        path = args.baseline / name
        assert digest_file(path) == catalog[name]['sha256'] and path.stat().st_size == catalog[name]['bytes']
        sources[str(path)] = digest_file(path)
    assert base['neurons'] == 4129924 and len(base['populations']) == 254
    window = base['window']
    end_tick=window['end_tick'];assert end_tick in (25000,105000,1005000)
    if end_tick == 1005000 and window['dt_seconds'] != .0001:
        raise ValueError('primary correlation requires the 0.1 ms physical grid')
    nbins=(end_tick-5000)//10
    assert (window['start_tick'], window['bin_ticks'], window['endpoint']) == (5000, 10, '[start,end)')
    names = [p['name'] for p in base['populations']]
    assert names == sorted(set(names))
    with np.load(args.baseline / 'activity-arrays.npz', allow_pickle=False) as arrays:
        frozen = arrays['population_counts']
    args.output.mkdir(exist_ok=False)
    paths = None
    cache_context = nullcontext(None)
    if args.native_audit:
        from analyze_native_mam_activity import bucket_events, simulation_identity, canonical_population_indices, DTYPE
        assert not args.model and not args.results and window['spike_tick_offset'] == 0
        audit = json.loads((args.native_audit / 'summary.json').read_text())
        assert audit['passed']
        parameters = list((args.native_audit / 'node25').glob('*/parameters.json'))
        assert len(parameters) == 1 and digest_file(parameters[0]) == audit['parameters_sha256'] == base['parameters_sha256']
        params = json.loads(parameters[0].read_text())
        files = sorted(args.native_audit.glob('node*/runs/*/rank*.events.bin'))
        reports = []
        for file in files:
            report = json.loads(file.with_name(file.name.replace('.events.bin', '.json')).read_text())
            assert file.stat().st_size == report['event_bytes'] == base['result_files'][str(file)]['bytes']
            assert digest_file(file) == report['event_sha256'] == base['result_files'][str(file)]['sha256']
            assert report['parameters_sha256'] == base['parameters_sha256']
            sources[str(file)] = report['event_sha256']
            reports.append(report)
        assert simulation_identity(reports,duration_ms=end_tick//10) == base['simulation']
        paths, totals = bucket_events(params['populations'], files, args.output / 'scratch-events', start_tick=0, end_tick=end_tick+1,max_bytes=(128*2**30 if end_tick==1005000 else 3072000000*end_tick//25000), release_file_cache=bounded)
        assert int(totals.sum()) == audit['spikes']
        order = canonical_population_indices(params['populations'])
        assert ['mam_' + params['populations'][j]['name'].replace('-', '_') for j in order] == names
        expected_raw = audit['spikes']
        identity = dict(simulator='NEST', condition=params['state'], **base['simulation'])
        sources[str(parameters[0])] = digest_file(parameters[0])
        sources[str(args.native_audit / 'summary.json')] = digest_file(args.native_audit / 'summary.json')
        def populations():
            for j in order:
                if bounded:
                    yield lambda j=j: streaming.native_blocks(paths[j], DTYPE), int(totals[j])
                    paths[j].unlink()
                    continue
                assert totals[j] <= 50_000_000
                events = np.fromfile(paths[j], dtype=DTYPE)
                assert len(events) == totals[j]
                yield events['tick'], events['cell']
                paths[j].unlink()  # Only this run's reconstructible scratch file.
    else:
        from brian2_rust.results import load_results
        assert args.model and args.results and window['spike_tick_offset'] == 1
        assert args.model.stat().st_size < 512 * 2**20 and digest_file(args.model) == base['model_sha256']
        assert sum((args.results / n).stat().st_size for n in ['results.bin', 'events.bin']) < {25000:8,105000:48,1005000:256}[end_tick] * 2**30
        for name, digest in base['result_sha256'].items():
            assert digest_file(args.results / name) == digest
            sources[str(args.results / name)] = digest
        sources[str(args.model)] = digest_file(args.model)
        model = json.loads(args.model.read_text())
        assert [p['name'] for p in model['definition']['populations']] == names
        assert [p['count'] for p in model['definition']['populations']] == [p['neurons'] for p in base['populations']]
        assert all(p['steps'] == end_tick for p in model['definition']['populations'])
        data = load_results(model, args.results, include_times=False, release_file_cache=bounded)
        if bounded:
            cache_context = streaming.mapped_cache(data['_dump'], args.results/'results.bin')
        expected_raw = data['metadata']['spike_count']
        identity = dict(simulator='Rust', model_sha256=base['model_sha256'], seed=model['instance']['rng_seed'])
        def populations():
            for population in data['populations']:
                if bounded:
                    yield lambda population=population: streaming.rust_blocks(population, release), len(population['spike_ticks'])
                    continue
                assert len(population['spike_ticks']) <= 50_000_000
                yield population['spike_ticks'].astype(np.int64) + 1, population['indices']
    rows, samples = [], {}
    total = 0
    with cache_context as release:
        for i, (ticks, cells) in enumerate(populations()):
            old = base['populations'][i]
            if bounded:
                blocks, raw_count = ticks, cells
                result, sample, original = streaming.summarize_population(blocks,
                    neurons=old['neurons'], end_tick=end_tick, expected_raw=raw_count, frozen=frozen[i])
                samples.update({f'p{i}_{k}': v for k, v in sample.items()})
                total += raw_count
                rows.append(dict(name=names[i], area=old['area'], group=old['group'], raw_events=raw_count,
                    frozen_observation_events=int(original.sum()), frozen_uniform_correlation=old['pairwise_corr_mean'], **result))
                continue
            assert len(ticks) == len(cells) and np.all(ticks >= 0) and np.all(ticks <= end_tick)
            assert np.all(cells >= 0) and np.all(cells < old['neurons'])
            keep = (ticks >= 5000) & (ticks < end_tick)
            original = np.bincount((ticks[keep].astype(np.int64) - 5000) // 10, minlength=nbins)
            np.testing.assert_array_equal(original, frozen[i])
            total += len(ticks)
            if len(cells):
                first = int(cells.min())
                hist = candidate_histogram(ticks, cells, first,end_tick=end_tick)
                result, ids, selected = summarize_histogram(hist, first)
                samples[f'p{i}_selected_ids'] = ids
                samples[f'p{i}_selected_spikes'] = selected.sum(axis=1)
                samples[f'p{i}_selected_histogram_sum'] = selected.sum(axis=0)
                del hist, selected
            else:
                result = dict(available=False, mean_pairwise_correlation=None, selected_cells=0,
                              unavailable_reason='Official wrapper accesses ids[0] on an empty population.')
            rows.append(dict(name=names[i], area=old['area'], group=old['group'], raw_events=len(ticks),
                             frozen_observation_events=int(original.sum()), frozen_uniform_correlation=old['pairwise_corr_mean'], **result))
    assert total == expected_raw and len(rows) == 254
    if paths:
        assert not any(p.exists() for p in paths)
        (args.output / 'scratch-events').rmdir()
    report = dict(schema='b2-mam-modern-paper-correlation-v1', scientific_equivalence=False,
        identity=identity, raw_events=total, frozen_histograms_exact=True, scratch_removed=True,
        observation_ms=[500, end_tick//10], endpoint=f'NumPy histogram [500,{end_tick//10}], last bin includes T', bin_ms=1,
        selection='Lowest recorded ID over the entire run; inclusive 3001-ID interval; remove all constant rows; first 2000 remaining IDs.',
        calculation='Gram-sum identity for mean off-diagonal Pearson; validated against literal corrcoef with absolute arithmetic tolerance 1e-12.',
        helper_sha256=HELPER_SHA, wrapper_sha256=WRAPPER_SHA, toolbox_commit=TOOLBOX_COMMIT,
        populations=rows, available_populations=sum(r['available'] for r in rows), source_sha256=sources,
        implementation_sha256={p.name: digest_file(p) for p in [Path(__file__), Path(__file__).with_name('mam_paper_correlation.py')]},
        analysis_seconds=time.perf_counter() - start,
        scope=f'Available-source correlation convention on full {end_tick/10000:g} s independent realizations. Historical dependency pin and paper-duration scientific equivalence remain unproven. Frozen uniform sampling remains a separate diagnostic.')
    report['bounded_memory'] = bounded
    if bounded:
        dependencies = [Path(__file__).with_name('mam_correlation_stream.py'),
                        Path(__file__).with_name('analyze_native_mam_activity.py'),
                        ROOT/'python/brian2_rust/results.py']
        report['implementation_sha256'].update({p.name: digest_file(p) for p in dependencies})
    (args.output / 'correlation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'selection.npz', **samples)
    catalog = {p.name: dict(bytes=p.stat().st_size, sha256=digest_file(p)) for p in args.output.iterdir()}
    (args.output / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['populations', 'source_sha256']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    for name in ['native-audit', 'model', 'results']:
        p.add_argument('--' + name, type=Path)
    p.add_argument('--bounded-memory', action='store_true', help='Stream events and release Linux file cache under an external resource guard.')
    analyze(p.parse_args())
