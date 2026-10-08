"""Audit the six alternating two-node spike-exchange comparison runs."""
import argparse
import json
from pathlib import Path
import re
from statistics import median


def verify(raw, reference, output):
    labels = ['v1v2-scale01-batch', 'spike-baseline-repeat1', 'spike-batch-repeat2',
              'spike-baseline-repeat2', 'spike-batch-repeat3', 'spike-baseline-repeat3']
    hosts = ['hk-prod-model-ae02-23', 'hk-prod-model-ae02-24']
    assert set(raw['launches']) == set(labels) and set(raw['nodes']) == set(hosts)
    rows = []
    for label in labels:
        launch = raw['launches'][label]
        assert launch['error'] is None and set(launch['returncodes']) == {'controller', 'proxy-0', 'proxy-1'}
        assert all(code == 0 for code in launch['returncodes'].values())
        assert launch['nodes'] == hosts and launch['ranks_per_node'] == 1
        assert launch['remote_timeout_seconds'] == 120
        prefix = launch['resource_guard']['unit_prefix']
        kind = 'baseline' if 'baseline' in label else 'batch'
        rss = []
        for rank, host in enumerate(hosts):
            files = raw['nodes'][host]['files']
            guards = [f'{prefix}-proxy-{rank}'] + ([f'{prefix}-controller'] if rank == 0 else [])
            for guard in guards:
                item = json.loads(files[f'guards/{guard}.json'])
                assert item['admitted'] and item['uid'] == 1000 and not item.get('error') and item['returncode'] == 0
                after = item['after']
                assert int(after['memory.max']) == 8*2**30 and after['memory.swap.max'] == '0'
                assert int(after['memory.peak']) <= int(after['memory.max'])
                quota, period = map(int, after['cpu.max'].split())
                assert quota == 4*period
                events = dict(line.split() for line in after['memory.events'].splitlines())
                assert events['oom'] == events['oom_kill'] == '0'
                assert item['wall_seconds'] < 125
            time = files[f'metrics/{label}-rank{rank}.time']
            assert re.search(r'Exit status: 0\s*$', time)
            rss.append(int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', time)[1])*1024)
        first = raw['nodes'][hosts[0]]
        assert first['hashes'][label] == reference['dump_sha256']
        summary = json.loads(first['files'][f'runs/{label}/summary.json'])
        assert summary['neuron_count'] == 35494 and summary['final_time_seconds'] == 0.1
        assert summary['spike_count'] == reference['spikes'] and summary['synaptic_events'] == reference['delivered_edges']
        runtime = summary['mpi']
        assert runtime['processor_names'] == hosts and runtime['rank_cpu_ids'] == [0, 0]
        assert runtime['exchange_calls_per_rank'] == (16000 if kind == 'baseline' else 1000)
        if kind == 'batch':
            assert runtime['spike_exchange_strategy'] == 'consecutive-producers-same-clock'
        rows.append({'label': label, 'kind': kind, 'simulation_seconds': summary['timings']['simulation_and_recording_seconds'],
                     'initialization_seconds': summary['timings']['initialization_seconds'],
                     'rank_peak_rss_bytes': rss, 'rank_exchange_seconds': runtime['spike_exchange_seconds']})
    medians = {kind: median(row['simulation_seconds'] for row in rows if row['kind'] == kind)
               for kind in ['baseline', 'batch']}
    report = {'schema': 'b2-mpi-spike-batch-comparison-v1', 'complete': True,
              'scope': 'Same 10.6-million-edge V1/V2 model, three alternating runs per executable on nodes23/24; no Brian2/NEST benchmark',
              'runs': rows, 'median_simulation_seconds': medians,
              'median_speedup': medians['baseline']/medians['batch'],
              'median_time_reduction_fraction': 1-medians['batch']/medians['baseline'],
              'all_dump_sha256': reference['dump_sha256']}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(json.loads(args.raw.read_text()), json.loads(args.reference.read_text()), args.output)
