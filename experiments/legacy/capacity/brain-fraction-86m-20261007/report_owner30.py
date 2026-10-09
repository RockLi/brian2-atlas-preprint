"""Report the owner-local experiment only after complete output/resource audits."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CASE = 'major-owner30'
def read(name):
    return json.loads((HERE / name).read_text())

def main():
    audit = read(CASE + '-audit.json')
    resources = read(CASE + '-resources.json')
    launch = read(CASE + '-launch-result.json')
    pilot = read('pilot-owner30-validation.json')
    assert audit['passed'] and resources['passed'] and pilot['passed']
    assert read('pilot-owner30-resources.json')['passed']
    assert read(CASE + '-deployment-audit.json')['passed']
    assert read('audit-' + CASE + '-status.json')['guard']['returncode'] == 0
    assert launch['error'] is None and all(v == 0 for v in launch['returncodes'].values())
    assert audit['neurons'] == 86_000_000 and audit['connections'] == 86_000_000_000
    runtime = audit['runtime']
    assert runtime['ranks'] == 30 and len(set(runtime['processor_names'])) == 30
    assert runtime['rank_cpu_ids'] == [1] * 30
    hardware = read('hardware-topology.json')
    assert len(hardware) == 30 and all(any(cpu['cpu'] == 1 for cpu in host['cpus']) for host in hardware)
    assert read('pilot-owner30-placement-proof.json')['passed']
    assert all(runtime['rank_work'][i * 3] > 0 for i in range(30))
    assert {t['construction'] for t in runtime['procedural_topology']} == {'target-owner-local'}
    peaks = []
    for i, row in enumerate(resources['rows']):
        guard = row['guards']['proxy-' + str(i)]
        assert guard['admitted'] and guard['returncode'] == 0
        assert 'oom 0' in guard['after']['memory.events'] and 'oom_kill 0' in guard['after']['memory.events']
        assert guard['after']['memory.swap.max'] == '0'
        assert len(row['rank_times']) == 1
        assert all('Exit status: 0' in s for s in row['rank_times'].values())
        peaks.append(int(guard['after']['memory.peak']) / 2**30)
    construction = np.sum([t['build_seconds'] for t in runtime['procedural_topology']], axis=0)
    csr = np.sum([t['rank_csr_offset_bytes'] for t in runtime['procedural_topology']], axis=0)
    assert construction.shape == csr.shape == (30,)
    history = []
    for case, outcome in [('major-p12', 'timeout'), ('major-b128', 'cancelled'), ('major-r60', 'timeout')]:
        previous = read(case + '-launch-result.json')
        failure = read(case + '-failure-resources.json')
        assert not any(value for row in failure['rows'] for value in row['cgroups_still_present'].values())
        previous_peaks = [int(g['after']['memory.peak']) / 2**30
                          for row in failure['rows'] for role, g in row['guards'].items()
                          if role.startswith('proxy-')]
        history.append({'case': case, 'outcome': outcome, 'completed': False,
                        'launch_wall_seconds': previous['wall_seconds'],
                        'maximum_worker_cgroup_peak_gib': max(previous_peaks),
                        'evidence': case + '-failure-resources.json'})
    summary = {
        'schema': 'atlas-owner30-capacity-final-report-v1', 'passed': True,
        'neurons': audit['neurons'], 'connections': audit['connections'],
        'mean_indegree': audit['mean_indegree'], 'duration_ms': audit['duration_ms'],
        'reference_neuron_count': 86_000_000_000, 'neuron_count_percent': 0.1,
        'hosts': 30, 'ranks': 30, 'physical_worker_cores': 30,
        'populations': 30, 'projections': 900, 'precision': 'reference-f64', 'dt_ms': .1,
        'construction': 'target-owner-local',
        'static_core_array_estimate': read('owner30-core-array-estimate.json'),
        'launch_wall_seconds': launch['wall_seconds'],
        'initialization_max_seconds': audit['stage_max_seconds'][0],
        'simulation_max_seconds': audit['stage_max_seconds'][2],
        'result_collection_max_seconds': audit['stage_max_seconds'][4],
        'spike_exchange_max_seconds': audit['spike_exchange_seconds_minmax'][1],
        'projection_build_per_rank_seconds_minmax': [float(construction.min()), float(construction.max())],
        'csr_offset_bytes_per_rank_minmax': [int(csr.min()), int(csr.max())],
        'worker_cgroup_peak_gib': peaks, 'maximum_worker_cgroup_peak_gib': max(peaks),
        'sum_of_nonsimultaneous_worker_peaks_gib': sum(peaks),
        'controller_cgroup_peak_gib': int(resources['rows'][0]['guards']['controller']['after']['memory.peak']) / 2**30,
        'spikes': audit['spikes'], 'delivered_synaptic_events': audit['delivered_synaptic_events'],
        'mean_firing_rate_hz': audit['spikes'] / (audit['neurons'] * .1),
        'output_bytes': sum(f['bytes'] for f in audit['output_files'].values()),
        'pilot_checks': pilot['checks'], 'pilot_max_state_absolute_difference': pilot['max_state_absolute_difference'],
        'isolated_engine_source_sha256': read('source-b128-identity.json')['archive_sha256'],
        'experiment_script_sha256': hashlib.sha256((HERE / 'prepare-owner30.py').read_bytes()).hexdigest(),
        'batch_constant_used_by_owner_construction': False,
        'prior_attempts': history, 'same_realized_graph_as_p12': False,
        'primary_checkout_modified_by_experiment': False,
        'scientific_human_brain_equivalence_claim': False,
        'full_scale_86b_feasibility_validated': False,
        'scaling_curve_measured': False, 'maximum_capacity_measured': False,
        'other_simulator_speed_comparison': False,
    }
    (HERE / 'final-report.json').write_text(json.dumps(summary, indent=2) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    for excitatory, label, color in [(True, 'Excitatory (80%)', '#3973ac'), (False, 'Inhibitory (20%)', '#c87542')]:
        pops = [p for p in audit['populations'] if (int(p['name'].rsplit('_', 1)[1]) < 24) == excitatory]
        count = sum(p['neurons'] for p in pops)
        assert count == audit['neurons'] * (4 if excitatory else 1) // 5
        rates = np.sum([p['spikes_per_1ms_bin'] for p in pops], axis=0) / (count * .001)
        axes[0].plot(np.arange(20) * 5 + 2.5, rates.reshape(20, 5).mean(1), lw=1.8,
                     linestyle='-' if excitatory else '--', color=color, label=label)
    axes[0].set(xlabel='Biological time (ms)', ylabel='Population firing rate (Hz)', title='a  Measured activity')
    axes[0].legend(fontsize=8, frameon=False)
    times = [summary['initialization_max_seconds'], summary['simulation_max_seconds'], summary['launch_wall_seconds']]
    axes[1].bar(['Initialization', 'Simulation', 'Launch to exit'], times, color=['#6b8bb2', '#428f70', '#d19a49'])
    axes[1].set(ylabel='Wall time (s)', title='b  Measured execution time', ylim=(0, max(times) * 1.18))
    axes[1].tick_params(axis='x', labelsize=8)
    for i, value in enumerate(times):
        axes[1].text(i, value + max(times) * .02, f'{value:.1f}', ha='center', fontsize=9)
    axes[2].bar(np.arange(1, 31), peaks, color='#428f70', width=.8)
    axes[2].axhline(256, color='#a35656', ls='--', lw=1, label='Per-host memory cap')
    axes[2].set(xlabel='Host in launch order', ylabel='Worker cgroup memory peak (GiB)', title='c  Measured resource use', ylim=(0, 280), xticks=[1, 5, 10, 15, 20, 25, 30])
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle('86 million neurons · 86 billion connections · 30 MPI ranks on 30 hosts', fontsize=13)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(HERE / ('capacity-result.' + suffix), dpi=180)
    plt.close(fig)
    s = summary
    report = f'''# brian2-atlas: 30-host connected capacity experiment

Completed and audited: **86 million neurons and 86 billion recurrent connections**, 100 ms biological time, dt 0.1 ms, reference-f64, 30 MPI ranks on 30 physical hosts. Every rank owns a populated neuron pool and uses one physical CPU core.

| Quantity | Measured result |
| --- | ---: |
| Launch to exit | {s['launch_wall_seconds']:.3f} s |
| Maximum local initialization | {s['initialization_max_seconds']:.3f} s |
| Maximum local simulation | {s['simulation_max_seconds']:.3f} s |
| Maximum rank spike exchange, included in simulation | {s['spike_exchange_max_seconds']:.3f} s |
| Maximum local collection/reporting | {s['result_collection_max_seconds']:.3f} s |
| Maximum per-host worker cgroup peak | {max(peaks):.3f} GiB |
| Spikes retained | {s['spikes']:,} |
| Delivered synaptic events | {s['delivered_synaptic_events']:,} |
| Complete output size | {s['output_bytes']:,} bytes |

![Measured activity, time and resource use](capacity-result.png)

## Workload and verification

This synthetic recurrent E/I network has 24 nearly equal excitatory pools (80% of neurons) and six inhibitory pools (20%), assigned one whole pool per rank. All 900 directed pool pairs use fixed-total uniform random draws with replacement; autapses and multapses are allowed. Projection counts use integer floors of source-count × target-count × 1,000 / total-neuron-count, with largest-remainder allocation to retain exactly 86 billion edges. Mean indegree is 1,000; it is not a fixed indegree for every neuron. The largest projection is below the frontend signed-int32 Synapses.N limit.

Euler LIF dynamics use membrane time constant 20 ms, current time constant 5 ms, drive 1.05, threshold 1, reset 0 and refractory time 2 ms. Initial voltage cycles through 1,000 values. E weights are uniform in [0.0027, 0.0033], I weights in [-0.0132, -0.0108]. Delays are clipped normal (mean 1.5 ms, standard deviation 0.25 ms, bounds 0.1–3 ms). Model seed is 20261007; projection seeds are 2026100700 + source×30 + target in numeric pool-creation order. Canonical population order sorts names lexicographically; ownership follows that canonical order.

All spikes and v/current traces from two neurons per pool are retained. A 24,000-neuron / 24-million-connection pilot with the same 30-pool layout and 30 hosts passed {pilot['checks']} independent Rust-reference checks: exact spikes and event totals, zero state/trace difference. All pilot ranks were populated, pinned to CPU 1, and used target-owner-local construction. The large run passed complete binary reading, finite-state, refractory, spike/count, rank-placement, and global connection-total checks. Ownership-local topology checks require each projection's target owner to construct all its edges and every other rank to construct zero. These are engineering checks; they do not establish anatomical human-brain equivalence or numerical equivalence to NEST.

All 30 workers and 31 MPI guards exited zero. Each host had a 256 GiB memory cap, one CPU, no swap, 128 GiB disk reserve, 8 GiB per output file and a 1,800 s timeout. Two hosts used data mounts; root disks were explicitly allowed on the other 28. Source/deployment hashes, physical core mappings, terminal cgroup records and per-rank timing files are retained. Nominal bond speed is 50,000 Mb/s, not measured usable throughput. These are shared hosts.

The prospective static array estimate was approximately 54.04 GiB per rank for target indices, f64 weights, 64-bit delays and source CSR offsets. It excludes states, replicas, queues, recording, scratch, result buffers, MPI, allocator and charged cache; it is not a peak-memory prediction or a maximum-capacity certificate. This run uses 30 worker cores and does not saturate all CPU cores available on these hosts.

## Implementation and prior attempts

The experiment uses an isolated source snapshot. Python/Rust opt-in neuron ceilings were raised from 16 million to 128 million; the default one-million ceiling and remaining guards were retained. Boundary checks passed (16 Python tests and the Rust budget test) with byte-identical frozen reference output. Original failed checks and narrow fixture/diagnostic corrections remain recorded. The primary engine checkout was not changed.

The source snapshot also contains a construction-batch increase from 16,384 to 131,072, tested separately across batch boundaries against the independent reference. **The final target-owner-local constructor does not use this batch loop.** The present result uses the existing ownership constructor, not a new algorithm attributable to that constant change. Engine archive SHA-256: `{s['isolated_engine_source_sha256']}`. The separately staged experiment script SHA-256 is `{s['experiment_script_sha256']}`; it is outside that archive and recorded separately.

An eight-pool preparation overflowed Brian2's signed-int32 Synapses.N before simulation. Two 12-pool distributed-draw-ranges attempts (240 and 60 ranks) timed out at 1,800 s without complete output; the larger-batch 240-rank attempt was cancelled after 410.90 s. Their launch, guard, memory and cancellation records are retained. No completed numerical/performance result is attributed to them. The owner-local protocol was frozen before this pilot and large run.

The final 30-pool model retains size, density and dynamics but changes the pool boundaries and seed matrix, hence the realized graph. It is **not byte-identical to the failed 12-pool model**, and these attempts are not a controlled speedup comparison. The earlier 8.6-million-neuron / 8.6-billion-connection workload completed 100 ms and 1 s on four hosts. Together these are tested capacity configurations, not a strong/weak scaling curve or maximum-capacity measurement.

## Interpretation for the preprint

The neuron count is **0.1% of an 86-billion-neuron reference count**. This is a synthetic connected capacity workload, not a reconstruction of a fraction of a human brain. This study was self-funded; execution at the full 86-billion-neuron scale was not evaluated, and feasibility at that scale remains unvalidated. Replicated source CSR offsets, frontend count limits, IR initialization data, communication and output require further validation at larger scales. Adding machines alone is not demonstrated to guarantee full-scale execution. The present configuration is tested for 100 ms only.

Suggested manuscript sentence: In a connected synthetic excitatory–inhibitory workload, brian2-atlas simulated 86 million neurons with 86 billion recurrent connections for 100 ms of biological time using 30 MPI ranks on 30 physical hosts. Execution completed in {s['launch_wall_seconds']:.2f} s including distributed initialization and output, with a maximum per-host worker cgroup memory peak of {max(peaks):.2f} GiB. Complete spike histories and sampled state traces were retained and audited. This corresponds to 0.1% of an 86-billion reference neuron count and does not imply human-brain equivalence or validated full-scale feasibility.

Preparation/export/compilation and deployment are excluded from launch-to-exit. Initialization includes connection construction/loading. Simulation includes exchange and recording; exchange time must not be added again. Local-stage maxima may occur on different ranks and must not be summed as end-to-end time. Worker cgroup peaks include each proxy, rank and charged file cache; summed individual peaks are not simultaneous aggregate RAM. Activity is shown in 5 ms bins.

Evidence: protocol-owner30.json, source-b128-identity.json, owner30-script-staging.json, hardware-topology.json, pilot-owner30-validation.json, pilot-owner30-resources.json, major-owner30-deployment-audit.json, major-owner30-launch-result.json, major-owner30-resources.json, major-owner30-audit.json, audit-major-owner30-status.json and final-report.json. Earlier attempts retain their own evidence. Remote raw artifacts remain under `/atlas-home/0003/atlas-capacity-86m-20261007/major-owner30/`; the leader resolves to its recorded data mount.
'''
    (HERE / 'RESULTS.md').write_text(report)
    manifest = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in HERE.iterdir() if p.is_file() and p.name != 'report-manifest.json'}
    (HERE / 'report-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
