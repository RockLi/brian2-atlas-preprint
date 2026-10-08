"""Audit raw before/after full FlyWire MPI outputs; report unselected samples."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from brian2_rust.binary_topology import file_hash
from brian2_rust.results import load_results
from verify_mpi_flywire import peak_rss


def verify(base, before, nodes):
    report = {'schema': 'b2-mpi-optimization-v1', 'complete': False, 'runs': [], 'performance': {}}
    build = json.loads((base/'models-final/build-report.json').read_text())
    assert build['complete'] and len(build['conditions']) == 4
    assert (build['graph']['neurons'], build['graph']['directed_pair_edges']) == (139255, 15091983)
    checks = {'full_graph': True, 'all_outputs_exact': True, 'placement': True, 'project_integrity': True}
    report['checks'] = checks
    identities = set()
    for folder in sorted((base/'runs').iterdir()):
        if not folder.is_dir() or not any(folder.name.startswith(v+'-'+c+'-r') for v in ('before','after') for c in build['conditions']):
            continue
        version, condition, rank_text, repeat_text = folder.name.split('-')
        ranks, repeat = int(rank_text[1:]), int(repeat_text)
        summary = json.loads((folder/'summary.json').read_text())
        runtime = json.loads((folder/'mpi-runtime.json').read_text())
        expected_hosts = nodes[:1] if ranks == 1 else [n for n in nodes for _ in range(ranks//2)]
        assert runtime['ranks'] == ranks and runtime['processor_names'] == expected_hosts
        if version == 'after':
            assert runtime['rank_cpu_ids'] == ([0] if ranks == 1 else [i for _ in nodes for i in range(ranks//2)])
        reference = before/'runs'/condition/'reference'
        hashes = {name: file_hash(folder/name) for name in ('results.bin','events.bin')}
        assert hashes == {name: file_hash(reference/name) for name in hashes}
        assert summary['final_time_seconds'] == 1 and summary['neuron_count'] == 139835
        project = (base/'models-final' if version == 'after' else before/'models')/condition/f'rank-{ranks}'
        if project not in identities:
            manifest = json.loads((project/'manifest.json').read_text())
            assert all(file_hash(project/name) == digest for name,digest in manifest['files'].items())
            assert file_hash(project/'b2-mpi') == json.loads((project/'build.json').read_text())['executable_sha256']
            identities.add(project)
        assert runtime['plan_sha256'] == json.loads((project/'manifest.json').read_text())['plan_sha256']
        peaks = [peak_rss(base/'metrics'/f'{folder.name}-rank{rank}.time') for rank in range(ranks)]
        report['runs'].append({'label': folder.name, 'version': version, 'condition': condition,
                              'ranks': ranks, 'repeat': repeat, 'warmup': condition=='rest' and repeat==0,
                              'summary': summary, 'hashes': hashes, 'rank_peak_rss_bytes': peaks})
    assert len(report['runs']) == 33
    for ranks in (1,2,4):
        rows = {}
        for version in ('before','after'):
            runs = [r for r in report['runs'] if r['condition']=='rest' and r['ranks']==ranks and r['version']==version and r['repeat']>0]
            assert {r['repeat'] for r in runs} == {1,2,3}
            times = [r['summary']['timings']['simulation_and_recording_seconds'] for r in runs]
            median = statistics.median(times)
            rows[version] = {'native_seconds': times, 'median_seconds': median,
                'spread_percent': 100*(max(times)-min(times))/median,
                'max_rank_rss_bytes': max(x for r in runs for x in r['rank_peak_rss_bytes']),
                'max_exchange_seconds': [max(r['summary']['mpi']['spike_exchange_seconds']) for r in runs]}
        rows['speedup'] = rows['before']['median_seconds']/rows['after']['median_seconds']
        report['performance'][str(ranks)] = rows
        print(ranks, rows, flush=True)
    populations = {}
    for c in build['conditions']:
        model = json.loads((base/'models-final'/c/'model.json').read_text())
        brain = next(i for i,p in enumerate(model['definition']['populations']) if p['name']=='flywire_neurons')
        assert model['definition']['populations'][brain]['count'] == 139255
        result = base/'runs'/f'after-{c}-r2-{3 if c=="rest" else 0}'
        populations[c] = load_results(model, result)['populations'][brain]
    checks['finite_states'] = all(np.isfinite(p['states'][s]).all() for p in populations.values() for s in ('v','ge','gi'))
    sensory = np.load(before/'graph/annotations.npz')['sensory']
    cut, quiet = populations['cut'], populations['cut_rest']
    masks = [~np.isin(p['indices'],sensory) for p in (cut,quiet)]
    checks['cut_control'] = bool(np.array_equal(cut['indices'][masks[0]],quiet['indices'][masks[1]]) and np.array_equal(cut['spike_times'][masks[0]],quiet['spike_times'][masks[1]]))
    report['complete'] = all(checks.values())
    (base/'optimization-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--nodes',nargs=2,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if verify(args.base,args.before,args.nodes)['complete'] else 1)
