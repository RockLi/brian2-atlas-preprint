"""Build a paper-ready report only after the full 30-host output audit passes."""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
def read(name): return json.loads((HERE / name).read_text())

def main():
    case = 'major-r60'
    audit = read(case + '-audit.json')
    resources = read(case + '-resources.json')
    launch = read(case + '-launch-result.json')
    pilot = read('pilot-r60-validation.json')
    assert read('batch-boundary-proof.json')['passed']
    assert audit['passed'] and resources['passed'] and pilot['passed']
    assert all(v == 0 for v in launch['returncodes'].values()) and launch['error'] is None
    assert read(case + '-deployment-audit.json')['passed']
    assert read('audit-' + case + '-status.json')['guard']['returncode'] == 0
    hardware = read('hardware-topology.json')
    assert len(hardware) == 30 and all(len({(r['physical_package_id'],r['core_id']) for r in h['cpus'] if r['cpu'] in [1,49]}) == 2 for h in hardware)
    peaks = []
    for i, row in enumerate(resources['rows']):
        guard = row['guards']['proxy-' + str(i)]
        assert guard['admitted'] and guard['returncode'] == 0
        assert 'oom 0' in guard['after']['memory.events'] and 'oom_kill 0' in guard['after']['memory.events']
        assert guard['after']['memory.swap.max'] == '0'
        assert all('Exit status: 0' in s for s in row['rank_times'].values())
        peaks.append(int(guard['after']['memory.peak']) / 2**30)
    old = json.loads((HERE.parent / 'brain-fraction-20261007/final-report.json').read_text())
    topology = audit['runtime']['procedural_topology']
    construction = np.sum([t['build_seconds'] for t in topology], axis=0)
    csr_bytes = np.sum([t['rank_csr_offset_bytes'] for t in topology], axis=0)
    assert construction.shape == (60,) and csr_bytes.shape == (60,)
    summary = {
        'schema': 'atlas-86m-capacity-final-report-v1', 'passed': True,
        'neurons': audit['neurons'], 'connections': audit['connections'],
        'mean_indegree': audit['mean_indegree'], 'duration_ms': audit['duration_ms'],
        'reference_neuron_count': 86_000_000_000, 'neuron_count_percent': 0.1,
        'hosts': 30, 'ranks': 60, 'physical_worker_cores': 60, 'populations': 12, 'projections': 144,
        'precision': 'reference-f64', 'dt_ms': 0.1,
        'launch_wall_seconds': launch['wall_seconds'],
        'initialization_max_seconds': audit['stage_max_seconds'][0],
        'simulation_max_seconds': audit['stage_max_seconds'][2],
        'result_collection_max_seconds': audit['stage_max_seconds'][4],
        'spike_exchange_max_seconds': audit['spike_exchange_seconds_minmax'][1],
        'projection_build_total_per_rank_seconds_minmax': [float(construction.min()), float(construction.max())],
        'csr_offset_bytes_per_rank_minmax': [int(csr_bytes.min()), int(csr_bytes.max())],
        'worker_cgroup_peak_gib': peaks, 'maximum_worker_cgroup_peak_gib': max(peaks),
        'sum_of_nonsimultaneous_worker_peaks_gib': sum(peaks),
        'controller_cgroup_peak_gib': int(resources['rows'][0]['guards']['controller']['after']['memory.peak']) / 2**30,
        'spikes': audit['spikes'], 'delivered_synaptic_events': audit['delivered_synaptic_events'],
        'mean_firing_rate_hz_over_100ms': audit['spikes'] / (audit['neurons'] * .1),
        'output_bytes': sum(f['bytes'] for f in audit['output_files'].values()),
        'pilot_checks': pilot['checks'], 'pilot_max_state_absolute_difference': pilot['max_state_absolute_difference'],
        'isolated_engine_source_sha256': read('source-b128-identity.json')['archive_sha256'],
        'construction_batch': 131072,
        'prior_attempt': {'case': 'major-p12', 'completed': False, 'timeout_seconds': 1800,
                          'launch_wall_seconds': read('major-p12-launch-result.json')['wall_seconds']},
        'primary_checkout_modified_by_experiment': False,
        'earlier_8_6m_result': {'neurons': old['neurons'], 'connections': old['connections'], 'hosts': 4, 'ranks': 32,
                              'populations': 4, 'durations_ms': [r['duration_ms'] for r in old['rows']]},
        'scientific_human_brain_equivalence_claim': False,
        'full_scale_86b_feasibility_validated': False, 'scaling_curve_measured': False,
        'maximum_capacity_measured': False, 'other_simulator_speed_comparison': False,
    }
    (HERE / 'final-report.json').write_text(json.dumps(summary, indent=2) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    excitatory = [p for p in audit['populations'] if int(p['name'].rsplit('_',1)[1]) < 8]
    inhibitory = [p for p in audit['populations'] if int(p['name'].rsplit('_',1)[1]) >= 8]
    assert sum(p['neurons'] for p in excitatory)==audit['neurons']*4//5
    assert sum(p['neurons'] for p in inhibitory)==audit['neurons']//5
    for pops, label, color in [(excitatory, 'Excitatory (80%)', '#3973ac'),
                               (inhibitory, 'Inhibitory (20%)', '#c87542')]:
        counts = np.sum([p['spikes_per_1ms_bin'] for p in pops], axis=0)
        rates = counts / (sum(p['neurons'] for p in pops) * .001)
        axes[0].plot(np.arange(20) * 5 + 2.5, rates.reshape(20, 5).mean(1), lw=1.8, color=color, label=label)
    axes[0].set(xlabel='Biological time (ms)', ylabel='Population firing rate (Hz)', title='a  Measured activity')
    axes[0].legend(fontsize=8, frameon=False)
    timing = [summary['initialization_max_seconds'], summary['simulation_max_seconds'], summary['launch_wall_seconds']]
    axes[1].bar(['Initialization', 'Simulation', 'Launch to exit'], timing, color=['#6b8bb2', '#428f70', '#d19a49'])
    axes[1].set(ylabel='Wall time (s)', title='b  Measured execution time', ylim=(0, max(timing) * 1.18))
    axes[1].tick_params(axis='x', labelsize=8)
    for i, value in enumerate(timing): axes[1].text(i, value + max(timing) * .02, f'{value:.1f}', ha='center', fontsize=9)
    axes[2].bar(np.arange(1, 31), peaks, color='#428f70', width=.8)
    axes[2].axhline(256, color='#a35656', ls='--', lw=1, label='Per-host memory cap')
    axes[2].set(xlabel='Host in launch order', ylabel='Worker cgroup memory peak (GiB)',
                title='c  Measured resource use', ylim=(0, 280), xticks=[1, 5, 10, 15, 20, 25, 30])
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle('86 million neurons · 86 billion connections · 60 MPI ranks on 30 hosts', fontsize=13)
    for suffix in ['png', 'svg', 'pdf']: fig.savefig(HERE / ('capacity-result.' + suffix), dpi=180)
    plt.close(fig)
    s = summary
    report = f'''# brian2-atlas: 30-host connected capacity experiment

**Completed and audited:** {s['neurons']:,} neurons, {s['connections']:,} recurrent connections, expected mean indegree 1,000, 100 ms biological time, dt 0.1 ms, reference-f64. Execution used 60 MPI ranks across 30 physical hosts, with one worker thread per rank on 60 distinct physical cores.

| Quantity | Measured result |
| --- | ---: |
| Launch to exit | {s['launch_wall_seconds']:.3f} s |
| Maximum local rank initialization | {s['initialization_max_seconds']:.3f} s |
| Maximum local rank simulation | {s['simulation_max_seconds']:.3f} s |
| Maximum rank spike exchange (included in simulation) | {s['spike_exchange_max_seconds']:.3f} s |
| Maximum local collection/reporting | {s['result_collection_max_seconds']:.3f} s |
| Maximum per-host worker cgroup peak | {max(peaks):.3f} GiB |
| Spikes retained | {s['spikes']:,} |
| Delivered synaptic events | {s['delivered_synaptic_events']:,} |
| Complete output size | {s['output_bytes']:,} bytes |

![Activity, execution time and memory on all 30 hosts](capacity-result.png)

## Workload and verification

The network consists of eight equal excitatory populations (80% of neurons) and four equal inhibitory populations (20%), with all 144 directed population pairs connected by fixed-total uniform random draws with replacement. Autapses and multapses are allowed. Positive weights are uniform in [0.0027, 0.0033]; inhibitory weights are uniform in [-0.0132, -0.0108]. Delays are clipped normal with mean 1.5 ms, standard deviation 0.25 ms and limits 0.1–3 ms. Euler LIF dynamics use membrane time constant 20 ms, current time constant 5 ms, drive 1.05, threshold 1, reset 0 and 2 ms refractory time. Initial voltage cycles through 1,000 deterministic values. The model seed is 20261007; projection seeds are 2026100700 + source×12 + target. Each projection contains at most 860 million connections.

All spikes and v/current traces from two neurons per population are retained. The 24,000-neuron / 24-million-connection pilot used the same 12-population configuration and all 30 hosts. It passed {pilot['checks']} checks against the independent single-process Rust reference: exact spikes and event totals, and zero state/trace difference. The large run passed complete binary reading, finite-state, refractory, count and global connection-total checks; all 60 ranks and 30 distinct host names are present. These checks establish engineering consistency at the tested size. They do not establish equivalence to an anatomical human-brain model, scientific equivalence to NEST, or a simulator speed advantage.

All 60 workers and all 31 MPI guards exited zero. Worker limits were 256 GiB and two CPUs per host, no swap, 8 GiB per output file, 128 GiB disk reserve, and a 1,800 s timeout. Root disks were explicitly allowed on 28 hosts; hosts 23 and 24 used `/data/brick2`. Source and output hashes, final cgroup evidence, per-rank timing files and physical core mappings are retained in the evidence records. The nominal bond speed is 50,000 Mb/s on all hosts; this is not a measured usable throughput. These are shared hosts with the recorded CPU placement.

## Implementation boundary and preparation failure

This experiment used an isolated source snapshot with an explicit neuron-limit extension: the opt-in ceiling was raised from 16 million to 128 million in the Python and Rust validators, while the default one-million limit and other semantic, work, array and execution guards were retained. Connection-construction batches were increased from 16,384 to 131,072, reducing the statically calculated routing batches from 87,536 to 11,008 at the final 60-rank configuration. Graph parameters, draw ranges, edge identities, f64 arithmetic and simulation scheduling were unchanged. A 48,000-neuron / 48-million-connection two-rank test exercised multiple batches at both sizes: both runs matched an independent Rust reference exactly in 123 checks. A separate 60-rank / 30-host pilot verified the final configuration. The isolated engine archive SHA-256 is `{s['isolated_engine_source_sha256']}`. The primary engine checkout was not changed by this experiment. Corrected boundary tests passed (16 Python tests plus the Rust budget test), with byte-identical frozen reference output. The initial Python test run exposed stale diagnostic wording and missing fixtures; the failed run, fixture provenance and narrowly corrected assertions remain recorded.

The first 12-population large attempt, using the 16,384 construction batch, did not complete within its 1,800 s timeout. It produced no completed runtime/output record; its maximum worker cgroup peak was 104.022 GiB and no OOM was recorded. All 31 terminal guard records were collected and all associated cgroups were gone before subsequent tests. Its evidence is retained under `major-p12-*`. A subsequent 240-rank attempt with the larger batch was actively cancelled after 410.90 s; it had no complete runtime/output record and is retained under `major-b128-*`. The final configuration used two ranks per host (60 total), reducing per-host source-count/CSR replication and collective participants. Its local-edge guard was set prospectively to 1.6 billion per rank, above the expected 1.433 billion; the 256 GiB memory guard and 1,800 s timeout were retained. No numerical or completed performance result is attributed to the cancelled attempt. The final batch-size protocol was frozen after that outcome and before the new numeric tests and large run. The incomplete attempt is not used as a completed performance baseline or as a measured initialization duration.

The first eight-population large preparation failed before simulation because Brian2 stores Synapses.N as signed int32: a 3.44-billion-connection projection overflowed. The final 12-population partition reduces the largest projection to 860 million while retaining 86 million neurons and 86 billion connections. The failed preparation and the frozen revised protocol are both retained. No successful large result is attributed to the failed configuration.

## Interpretation for the preprint

The current distributed construction retains source-index CSR offsets per process. This replicated component grows with global neuron count and projection partition, and per-host replication depends on rank density. The experiment does not demonstrate that adding machines alone enables the full reference scale; larger scales require further layout, frontend-count, communication and capacity validation.

The neuron count is **0.1% of an 86-billion-neuron reference scale**. This is a synthetic connected capacity workload, not a simulation of an anatomical or functional 0.1% of a human brain. This study was self-funded; the full 86-billion-neuron scale was not evaluated with the available resources, and feasibility at that scale remains unvalidated.

The earlier 8.6-million-neuron / 8.6-billion-connection experiment completed both 100 ms and 1 s on four hosts. The present result uses a different population partition and 30 hosts. Together they demonstrate two tested capacity configurations; they do not form a strong/weak scaling curve or measure maximum capacity. The present 86-million configuration has been tested for 100 ms only.

Suggested manuscript sentence: In a connected synthetic excitatory–inhibitory workload, brian2-atlas simulated 86 million neurons with 86 billion recurrent connections for 100 ms of biological time using 60 MPI ranks across 30 physical hosts. Execution completed in {s['launch_wall_seconds']:.2f} s including distributed initialization and output, with a maximum per-host worker cgroup memory peak of {max(peaks):.2f} GiB. Complete spike histories and sampled state traces were retained and audited. This corresponds to 0.1% of an 86-billion reference neuron count and does not imply human-brain equivalence or validated full-scale feasibility.

Timing scopes: preparation/export/compilation and artifact deployment occur before launch and are excluded from launch-to-exit. Initialization includes distributed connection construction/loading. Simulation includes spike exchange and recording; exchange time must not be added to it again. Local-stage maxima need not occur on the same rank and must not be summed to infer end-to-end duration. Cgroup peaks cover each host's proxy and two workers, including charged file cache; summed individual peaks are not a simultaneous aggregate RAM measurement. Activity uses 5 ms bins.

Evidence: protocol-p12.json, protocol-b128.json, protocol-r60.json, source-b128-identity.json, hardware-topology.json, batch-boundary-proof.json, pilot-r60-validation.json, pilot-r60-resources.json, major-r60-deployment-audit.json, major-r60-launch-result.json, major-r60-resources.json, major-r60-audit.json, audit-major-r60-status.json and final-report.json. The failed attempt remains in major-p12-failure-resources.json and its launch transcripts. Raw model/output artifacts remain in isolated remote directories under `/atlas-home/0003/atlas-capacity-86m-20261007/major-r60/`; the leader resolves to the recorded data mount.
'''
    (HERE / 'RESULTS.md').write_text(report)
    manifest = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in HERE.iterdir() if p.is_file() and p.name not in ['report-manifest.json']}
    (HERE / 'report-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
