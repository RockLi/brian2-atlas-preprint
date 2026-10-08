"""Source-bound conditional arithmetic; no simulation, regression or remote job."""
from pathlib import Path
from evidence_paths import legacy_repo, external_directory
import base64
import csv
import hashlib
import json
import math
import os
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPO = legacy_repo()
EXPERIMENT = REPO / 'brian2-rust/mpi-evidence/brain-count-scaling-1pct-20261007'
OUT = ROOT / 'data/full_scale'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
weak_path = ROOT / 'data/capacity/weak-evidence.json'
weak = json.loads(weak_path.read_text())
rows = weak['rows']
assert [r['hosts'] for r in rows] == [3, 6, 12, 24, 30]
assert all(r['passed'] and r['duration_ms'] == 100 for r in rows)
identity = json.loads((EXPERIMENT / 'source-identity.json').read_text())
assert identity['archive_sha256'] == weak['source_archive_sha256']
names = ['mpi_codegen.py', 'slot_codegen.py', 'mpi_partition.py',
         'mpi_compact.py', 'native.py', 'mpi_spike_history.py',
         'mpi_spike_output.py', 'mpi_queue_compact.py',
         'mpi_report_compact.py', 'mpi_runtime/topology.rs',
         'mpi_runtime/bridge.c', 'mpi_runtime/additive.rs']
sources = []
for name in names:
    relative = 'python/brian2_rust/' + name
    path = EXPERIMENT / 'source' / relative
    assert sha(path) == identity['files'][relative]['sha256'], relative
    retained = 'source-' + name.replace('/', '-')
    shutil.copyfile(path, OUT / retained)
    sources.append(dict(source=relative, retained=retained, sha256=sha(path)))
shutil.copyfile(EXPERIMENT / 'source-identity.json', OUT / 'source-identity.json')
code = (OUT / 'source-slot_codegen.py').read_text()
assert 'capacity + d["populations"][event[0]]["count"] > 2**31-1' in code
assert 'flush_events()' in code
runtime = (OUT / 'source-mpi_codegen.py').read_text()
assert 'if receive.len() < capacity { receive.resize(capacity, 0); }' in runtime
assert 'let capacity_i32 = i32::try_from(capacity)?;' in runtime
assert 'let (send, receive) = &mut *buffers;' in runtime
topology = (OUT / 'source-mpi_runtime-topology.rs').read_text()
assert 'let mut offsets=vec![0usize;sources+1];' in topology
assert 'let mut cursor=offsets[..sources].to_vec();' in topology
compact = (OUT / 'source-mpi_compact.py').read_text()
assert 'if not random_edges(syn):' in compact and 'drop(original_edges)' in compact
bridge = (OUT / 'source-mpi_runtime-bridge.c').read_text()
assert 'MPI_Allgatherv' in bridge
report_code = (OUT / 'source-mpi_report_compact.py').read_text()
assert 'let mut topology_report = String::from("[");' in report_code
assert 'rank_stats' in report_code and 'rank_csr_offset_bytes' in report_code

GIB = 2**30
FULL = 86_000_000_000
K = 1000
BLOCK = 86_000_000
LIMIT = 2**31 - 1
endpoint = rows[-1]
activity = endpoint['spikes'] / endpoint['neurons']
e_sizes = [68_800_000 // 19 + int(i < 68_800_000 % 19) for i in range(19)]
i_sizes = [3_440_000] * 5
host_sizes = [sum(e_sizes[:6]) + sum(i_sizes[:2]),
              sum(e_sizes[6:12]) + sum(i_sizes[2:4]),
              sum(e_sizes[12:]) + i_sizes[4]]
assert sum(host_sizes) == BLOCK
max_local_n = max(e_sizes + i_sizes)
max_host_n = max(host_sizes)

def quantities(n):
    blocks = n / BLOCK
    hosts = 3 * blocks
    ranks = 24 * blocks
    p = ranks
    # Consecutive populations are split before the checked MPI count limit.
    # This upper bound does not assert a particular ungenerated 86B schedule.
    receive_capacity = min(n, LIMIT)
    csr_rank_bytes = 8 * (n + p)
    local_edge_host_bytes = 20 * max_host_n * K
    return dict(neurons=n, count_percent=n / FULL * 100,
        workload_hosts=hosts, workload_physical_cores=ranks, populations=p,
        connections=K*n, projections=p*p,
        immutable_synapse_cluster_bytes=20*K*n,
        local_synapse_host_bytes=local_edge_host_bytes,
        local_neuron_host_bytes=33*max_host_n,
        dense_source_csr_rank_bytes=csr_rank_bytes,
        dense_source_csr_host_bytes=8*csr_rank_bytes,
        spike_receive_rank_bytes_bound=8*receive_capacity,
        spike_receive_host_bytes_bound=64*receive_capacity,
        selected_worker_arrays_host_bytes=local_edge_host_bytes + 8*csr_rank_bytes + 64*receive_capacity,
        root_spike_count_bytes=8*n,
        root_final_neuron_arrays_bytes=25*n,
        root_logical_history_bytes_if_endpoint_activity=8*activity*n,
        root_selected_record_bytes_if_endpoint_activity=(33+8*activity)*n,
        binary_scientific_payload_bytes_if_endpoint_activity=(33+16*activity)*n,
        dense_report_scalar_entries_per_copy=4*p**3,
        dense_report_json_bytes_lower_bound_per_copy=8*p**3,
        projection_report_collectives=math.ceil(p*p/128),
        projection_report_gather_scratch_bytes=640*8*ranks,
        population_instance_initial_values=4*n,
        binary_neuron_input_bytes=25*n,
        per_rank_topology_recipe_payload_bytes=24*p*p,
        topology_recipe_payload_all_rank_files_bytes=24*p**3)

measured = []
for row in rows:
    path = ROOT / f"data/capacity/weak-{row['case']}-audit.json"
    audit = json.loads(path.read_text())
    assert audit['passed']
    q = quantities(row['neurons'])
    # Preserve realised activity/output rather than substituting an estimated rate.
    q.update(case=row['case'], measured_host_peak_gib=row['maximum_worker_cgroup_peak_gib'],
        measured_other_hosts_peak_gib=max(row['worker_cgroup_peak_gib'][1:]),
        measured_root_host_peak_gib=row['worker_cgroup_peak_gib'][0],
        measured_binary_output_bytes=sum(audit['output_files'][f]['bytes'] for f in ['results.bin','events.bin']),
        measured_root_logical_record_bytes=33*row['neurons'] + 8*row['spikes'],
        measured_history_capacity_bytes=audit['runtime']['spike_history_capacity_bytes'],
        measured_max_rank_queue_capacity_bytes=max(audit['runtime']['rank_queue_capacity_bytes']),
        measured_spikes=row['spikes'], launch_wall_seconds=row['launch_wall_seconds'],
        measured_full_output_bytes=row['output_bytes'])
    assert audit['runtime']['exchange_calls_per_rank'] == 1000
    assert audit['runtime']['spike_exchange_strategy'] == 'consecutive-producers-same-clock'
    measured.append(q)
full = quantities(FULL)
assert full['workload_hosts'] == 3000 and full['workload_physical_cores'] == 24000
assert full['connections'] == 86_000_000_000_000
assert full['projections'] == 576_000_000
assert full['dense_report_scalar_entries_per_copy'] == 55_296_000_000_000
assert full['spike_receive_rank_bytes_bound'] < 16*GIB
assert K*max_local_n + full['populations'] < 4_290_000_000
assert math.ceil(K*max_local_n**2/FULL) < LIMIT
assert math.isclose(full['dense_source_csr_rank_bytes']/GIB, 640.7501101493835)
qrate = endpoint['spikes'] / (endpoint['neurons']*.1)
queue_scenarios = [dict(rate_hz=f, mean_delay_seconds=.0015,
    logical_queue_host_bytes=4*max_host_n*K*f*.0015,
    global_fired_and_last_fired_bytes_per_rank=16*FULL*f*.0001,
    logical_history_root_bytes=8*FULL*f*.1,
    binary_scientific_payload_bytes=33*FULL + 16*FULL*f*.1)
    for f in [1, qrate, 40]]
analysis = dict(schema='atlas-full-scale-conditional-resource-analysis-v1',
    status='passed', source_archive_sha256=identity['archive_sha256'],
    evidence_sha256=sha(weak_path), sources=sources,
    source_identity_sha256=sha(OUT/'source-identity.json'),
    observed=measured, conditional_full_scale=full, queue_and_activity_scenarios=queue_scenarios,
    assumptions=dict(neuron_count_reference=FULL, mean_indegree=K, model_time_ms=100,
        ranks_per_host=8, neuron_state_bytes=33, edge_payload_bytes=20,
        csr_offset_bytes=8, spike_receive_element_bytes=8, queue_item_bytes=4,
        stored_history_pair_bytes=8, public_spike_pair_bytes_per_file=8,
        endpoint_spikes_per_neuron_per_100ms=activity,
        maximum_local_population_neurons=max_local_n,
        local_incoming_edges_per_rank_approx=max_local_n*K,
        maximum_local_host_neurons=max_host_n,
        signed_mpi_count_max=LIMIT,
        receive_policy='Reuse maximum batch buffer; generator splits consecutive producers before signed-i32 capacity. No ungenerated exact batch-count assertion.',
        report_policy='Every process builds full projection-by-rank report strings; four scalar entries per projection/rank per report copy.',
        units='GiB = 2^30 bytes; decimal TB = 10^12 bytes; TiB = 2^40 bytes.'),
    unquantified=['Vec capacity/growth and allocator overhead','fired-vector transient peaks',
        'exact full-scale projection/rank metadata allocation','frontend/compiler peak',
        'charged cache and simultaneous host peaks','network fabric and runtime',
        'changed graph activity, scientific anatomy and full-scale correctness'],
    scope='Conditional synthetic-workload arithmetic bound to frozen source. Not a peak-memory fit, current-version host requirement, execution feasibility, wall-time prediction or new experiment.')
(OUT/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
with (OUT/'resource_scenarios.csv').open('w', newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(full));w.writeheader()
    for n in [86_000_000,172_000_000,344_000_000,688_000_000,860_000_000,8_600_000_000,FULL]:w.writerow(quantities(n))

os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parent / 'tmp/matplotlib'))
os.environ.setdefault('MPL_IGNORE_SYSTEM_FONTS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
blue,orange,green='#245c85','#ba6d2f','#31826e'
fig,axes=plt.subplots(2,2,figsize=(11.4,8),layout='constrained')
x=np.array([r['count_percent'] for r in measured])
future=np.geomspace(1,100,100); future_q=[quantities(float(FULL*f/100)) for f in future]
ax=axes[0,0]
ax.plot(x,[r['workload_hosts'] for r in measured],'o-',color=blue,label='Tested layouts')
ax.plot(future,[r['workload_hosts'] for r in future_q],'--',color=blue,label='Conditional local-work mapping')
ax.annotate('3,000 hosts / 24,000 cores',xy=(100,3000),xytext=(.6,2000),fontsize=10,
    arrowprops={'arrowstyle':'->','color':blue})
ax.set(title='a  Fixed local-work host allocation',ylabel='Hosts (8 worker cores per host)',yscale='log')
ax=axes[0,1]
ax.plot(x,[r['measured_other_hosts_peak_gib'] for r in measured],'o-',color=green,label='Measured other-host peak')
ax.plot(x,[r['selected_worker_arrays_host_bytes']/GIB for r in measured],'-',color=orange,label='Selected arrays (dense CSR)')
ax.plot(future,[r['selected_worker_arrays_host_bytes']/GIB for r in future_q],'--',color=orange)
ax.axhline(768,color='#777777',ls=':',label='Tested coordinator cap (768 GiB)')
ax.set(title='b  Selected worker arrays',ylabel='Per-host GiB (not a peak forecast)',yscale='log')
ax=axes[1,0]
ax.plot(x,[r['measured_binary_output_bytes']/GIB for r in measured],'o-',color=blue,label='Observed scientific binary files')
ax.plot(future,[r['binary_scientific_payload_bytes_if_endpoint_activity']/GIB for r in future_q],'--',color=blue,label='Conditional binary payload')
ax.plot(x,[r['measured_root_logical_record_bytes']/GIB for r in measured],'-',color=orange,label='Root logical record arrays')
ax.plot(future,[r['root_selected_record_bytes_if_endpoint_activity']/GIB for r in future_q],'--',color=orange)
ax.set(title='c  Scientific recording (100 ms)',ylabel='GiB (excludes provenance reports)',yscale='log')
ax=axes[1,1]
ax.plot(x,[r['dense_report_scalar_entries_per_copy'] for r in measured],'-',color=orange,label='Derived at tested sizes')
ax.plot(future,[r['dense_report_scalar_entries_per_copy'] for r in future_q],'--',color=orange,label='Conditional dense reporting')
ax.set(title='d  Dense report growth',ylabel='Scalar entries per report copy',yscale='log')
for ax in axes.ravel():
    ax.set_xscale('log');ax.set_xlim(.08,140);ax.set_xlabel('Reference neuron count (%)')
    ax.set_xticks([.1,1,10,100],['0.1','1','10','100'])
    ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=11,frameon=False,loc='best')
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    ax.tick_params(labelsize=11)
fig.savefig(ROOT/'figures/figS2_full_scale_resources.svg')
fig.savefig(ROOT/'figures/figS2_full_scale_resources.png',dpi=180)
plt.close(fig)

section = f'''## S21. Conditional resource analysis at an 86-billion-neuron reference count

### S21.1 Scope and workload mapping

This analysis concerns the synthetic recurrent E/I workload in S20, with reference-f64, expected mean indegree K = 1,000, dt = 0.1 ms and 100 ms model time. N = 86 billion therefore gives E = KN = 86 trillion connections. It does not represent reconstructed human anatomy, a validated whole-brain dynamical model, or a new execution. The [machine-readable analysis](data/full_scale/analysis.json), [scenario table](data/full_scale/resource_scenarios.csv) and [analysis script](scripts/analyze_full_scale_resources.py) bind arithmetic and assumptions to the accepted observations and frozen source archive. The twelve inspected source files are retained and hash-verified against the experiment catalogue.

Preserving the three-host/86-million-neuron block gives H = 3N/(86 million) hosts and R = 8H ranks/physical worker cores. The endpoint extended by a factor of 100 maps to 3,000 hosts and 24,000 physical worker cores, with 24,000 populations and 576 million population-pair projections. This is a conditional local-work allocation, requiring bounded indices, buffers, reporting and recording. It is not the host requirement of the current implementation, a guarantee that the tested host memory suffices, or a prediction of runtime. Graph realizations and activity may change with scale. The measured 1% launch time is not extrapolated.

### S21.2 Source-derived memory accounting

For one rank owning n neurons and e incoming edges, the retained immutable synapse arrays use 20e bytes: target indices u32 (4), weights f64 (8) and per-edge delay ticks usize (8) on the tested 64-bit target. Local e is approximated by Kn in the conditional table; largest-remainder projection allocation can slightly perturb individual local counts, and no full-scale edge-count matrix is constructed here. Temporary original-edge identities are used during initialization, then dropped for this deterministic additive pathway. Runtime neuron state uses 33n bytes: v/current (16), lastspike (8), refractory-until (8) and a byte flag (1). These exclude allocation capacity, queues and metadata.

There is one dense source-offset array per incoming projection. Summing source populations gives 8(N + P) bytes per rank, where P = R is the population count; the eight ranks on a host therefore retain 64(N + P) bytes. At the full reference count this is {full['dense_source_csr_rank_bytes']/GIB:.3f} GiB per rank and {full['dense_source_csr_host_bytes']/2**40:.3f} TiB per host before other arrays. This replication survives adding hosts. Dense index construction also uses a temporary 8n_source-byte cursor for one projection, and temporary 8e_projection-byte original-edge identities coexist with weight/delay construction. They are lifetime-specific temporaries and are not summed across all projections as simultaneous allocations.

Spike-receive capacity requires a separate qualification. `slot_codegen.py` flushes consecutive producers before their summed population count would exceed 2^31 - 1; `exchange_spike_batch` then checks the batch capacity and resizes one reusable Vec<u64>. For this workload every individual population remains below the limit. At tested N the single batch has capacity N; above that range, the existing generator splits batches. Thus 8N is not the full-scale receive-buffer formula. The element payload has an upper bound of 8(2^31 - 1) bytes, just below 16 GiB per rank and 128 GiB per eight-rank host, excluding Vec over-allocation and MPI scratch. Every proposed schedule and count/displacement still needs qualification. Batching limits the buffer; it does not remove global spike dissemination or establish efficient communication at 24,000 ranks.

**Table S4 | Full-reference conditional resource components.** Array sizes are logical element payloads on a 64-bit target. They are not a predicted process RSS/cgroup peak, and rows belonging to different lifetimes must not be summed as a simultaneous requirement. Root recording and output estimates assume the observed 1% activity per neuron persists for 100 ms. “Unchanged dense” terms identify costs to redesign, not a feasible execution configuration.

| Component / condition | Full-reference quantity | Interpretation |
|---|---:|---|
| Neurons / connections | 86 billion / 86 trillion | Count-normalized synthetic workload |
| Fixed local-work allocation | 3,000 hosts / 24,000 cores | Conditional allocation, not current host requirement |
| Largest local population / incoming edges | {max_local_n:,} / approximately {max_local_n*K:,} | Largest-remainder allocation can perturb edge totals |
| Largest host's immutable synapse arrays | {full['local_synapse_host_bytes']/GIB:.3f} GiB | 20 bytes per edge; other memory excluded |
| Immutable synapse arrays across ranks | {full['immutable_synapse_cluster_bytes']/1e15:.3f} decimal PB | 20E bytes; not total cluster memory |
| Largest host's local neuron arrays | {full['local_neuron_host_bytes']/GIB:.3f} GiB | 33 bytes per local neuron |
| Unchanged dense source CSR | {full['dense_source_csr_rank_bytes']/GIB:.3f} GiB/rank; {full['dense_source_csr_host_bytes']/2**40:.3f} TiB/host | Global source rows replicated per rank |
| Batched spike-receive element bound | <16 GiB/rank; <128 GiB/host | Existing signed-count-aware generator; capacity overhead excluded |
| Root counts / collected final neuron arrays | {full['root_spike_count_bytes']/GIB:.3f} / {full['root_final_neuron_arrays_bytes']/GIB:.3f} GiB | Concentrated recording and post-run collection |
| Root logical spike history | {full['root_logical_history_bytes_if_endpoint_activity']/GIB:.3f} GiB | 8 bytes per spike; Vec capacity can exceed this |
| Scientific binary payload | {full['binary_scientific_payload_bytes_if_endpoint_activity']/1e12:.3f} decimal TB | Final arrays and two spike streams; excludes reports/headers/final-fired/traces |
| Population-pair projections / dense report entries | 576 million / {full['dense_report_scalar_entries_per_copy']:,.0f} | Four scalar entries per projection/rank in each report copy |
| Binary topology recipes in all rank shards | {full['topology_recipe_payload_all_rank_files_bytes']/1e12:.3f} decimal TB | 24 bytes/projection/rank, before neuron input/headers |

### S21.3 Activity, queues and recording

For roughly stationary activity f Hz and mean delay d = 1.5 ms, a logical pending-edge estimate is 4efd bytes per rank, because this experiment stores u32 pending edge indices. For the largest host, f = 1, {qrate:.3f} and 40 Hz give approximately {queue_scenarios[0]['logical_queue_host_bytes']/GIB:.3f}, {queue_scenarios[1]['logical_queue_host_bytes']/GIB:.3f} and {queue_scenarios[2]['logical_queue_host_bytes']/GIB:.3f} GiB of live queue entries. These are activity scenarios, not capacity reservations or hard upper bounds. Transients, heterogeneous delays, ring-slot capacities and allocator growth matter: the accepted 1% run retained up to {measured[-1]['measured_max_rank_queue_capacity_bytes']/GIB:.3f} GiB of queue capacity per rank. Averages cannot establish a transient peak.

All ranks receive the spikes and retain population fired/last-fired vectors for canonical event delivery. Their logical global storage depends on activity; a stationary scenario is approximately 16Nf dt bytes per rank for two usize vectors, before retained peak capacities. Communication includes global allgather/allgatherv, validation, sorting and scattering. At endpoint activity S/N = {activity:.9f} spikes per neuron per 100 ms, full-reference payload delivery is 8S = {8*activity*FULL/1e9:.3f} decimal GB per rank over the run; aggregate recipient payload R(8S) is {24000*8*activity*FULL/1e15:.3f} decimal PB. This is logical recipient payload, not measured fabric traffic or a bandwidth/runtime forecast; MPI algorithms, same-host sharing and network topology alter physical traffic.

Rank zero retains counts (8N bytes) and one compact spike history (8S logical bytes). Post-run collection adds v/current/lastspike/flag arrays (25N bytes) on that rank, with temporary conversion/receive buffers during each population's collection. For unchanged activity, these selected root arrays total {(33*FULL+8*activity*FULL)/2**40:.3f} TiB, excluding its synapses, CSR, queues, capacity overhead and report strings. The observed 1% history allocation was {measured[-1]['measured_history_capacity_bytes']/GIB:.3f} GiB versus {8*endpoint['spikes']/GIB:.3f} GiB of logical records, illustrating why payload is not peak memory. Distributed/streamed recording and collection are conditions for avoiding this central concentration.

The two scientific binary files store final arrays and both logical spike streams. A selected payload formula is 33N + 16S bytes. At unchanged endpoint activity it gives {full['binary_scientific_payload_bytes_if_endpoint_activity']/1e12:.3f} decimal TB for 100 ms. Trace samples, final-fired vectors, headers and provenance reports are additional. Files of this size exceed the tested 64 GiB per-file guard; output sharding/streaming or newly qualified finite limits would be required. This is not a forecast of total output, and it is not extrapolated to longer model times or different activity.

### S21.4 Projection metadata, construction and output provenance

The unchanged population-pair representation has P² projection recipes/objects on every rank. Binary shards serialize edge-count/seed/pending-count words for each recipe, even on non-owner ranks: at least 24P² bytes per shard, giving {full['per_rank_topology_recipe_payload_bytes']/1e9:.3f} decimal GB per shard and {full['topology_recipe_payload_all_rank_files_bytes']/1e12:.3f} decimal TB across all shards, before population inputs. The four per-neuron instance columns alone contribute 4N initial values and 25N binary bytes. They exceed the experiment's four-billion-value admission policy at full reference count. The frozen experiment driver admits only the five tested host counts and its opt-in 860-million-neuron ceiling; these and the 64 GiB IR limit are finite tested policies, not full-reference qualification. The endpoint model JSON was 53,438,046,766 bytes; neither its frontend peak nor compilation/preparation time is fitted or linearly forecast. Lazy/shared recipes, streamed instances and newly qualified finite admission limits would be needed.

Projection reporting already batches 128 projections: the gathered packet is at most 640R u64 values, {full['projection_report_gather_scratch_bytes']/GIB:.3f} GiB at R = 24,000. However, the reporting function appends the complete dense report string on every process. It retains two integer rank statistics, one build-time value and one CSR-byte value for every projection/rank pair: 4P²R = 4P³ scalar entries per report copy. At full reference count there are {full['dense_report_scalar_entries_per_copy']:,.0f} entries; even a conservative two bytes per entry corresponds to at least {full['dense_report_json_bytes_lower_bound_per_copy']/1e12:.3f} decimal TB ({full['dense_report_json_bytes_lower_bound_per_copy']/2**40:.3f} TiB) per serialized copy, before other fields. The current formatting is larger. This is a combinatorial lower-bound warning for an unchanged representation, not a proposed memory allocation. Root writes this provenance into both runtime and summary JSON. Sparse owner records, aggregated counters and bounded streaming would need to replace this path. Simply multiplying the observed complete output by 100 would miss this growth.

### S21.5 Measured range and conditional extension

![Supplementary Figure S2. Full-reference conditional resources](figures/figS2_full_scale_resources.svg)

**Supplementary Figure S2 | Source-derived conditional resource analysis.** Solid segments describe tested layouts, observed memory/output, or source-derived quantities at those layouts; dashed segments beyond 1% are conditional estimates/bounds from arithmetic, not executions. (a) Fixed local-work host allocation. (b) Other-host measured peaks and selected arrays retaining dense CSR and the batched receive bound. Selected arrays exclude state, queues, recording, report metadata, allocator and cache; they do not predict peak memory. (c) Scientific binary output and root logical record arrays under fixed endpoint activity and 100 ms, excluding provenance reports. (d) Projection-by-rank report entries under unchanged dense reporting. Axes are logarithmic. No runtime fit, uncertainty band, full-scale host admission or actual full-scale capability is implied.

### S21.6 Conditions for further extension

The tested local-work distribution provides a basis for further distributed capacity studies. Extending it requires reducing global source-index replication, preserving count-safe batched communication while qualifying thousands of ranks, bounding projection metadata/reporting and frontend construction, and distributing recording/result output. New index/layout choices must preserve source/edge identities and event ordering, pass independent numerical gates, and be measured for memory and communication at intermediate scales. The selected array payloads and buffer bounds here are not sufficient evidence that any proposed redesign fits a given host configuration.

This study was self-funded. Execution at the full 86-billion-neuron count was not evaluated within the available computational resources. Resource arithmetic identifies a conditional allocation and specific implementation changes for further investigation; it does not establish full-scale execution feasibility or imply that additional machines alone suffice.
'''
supp = ROOT/'SUPPLEMENTARY.md'
text = supp.read_text()
text = re.sub(r'\n## S21\. Conditional resource analysis at an 86-billion-neuron reference count\n.*\Z','',text,flags=re.S)
supp.write_text(text.rstrip()+'\n\n'+section)
try:
    import mistune
    html = mistune.create_markdown(escape=False,plugins=['table'])(section)
except ImportError:
    from markdown_it import MarkdownIt
    html = MarkdownIt('commonmark',{'html':True}).enable('table').render(section)
html=html.replace('<h2>', '<h2 class="conditional-resource-section">',1)
html=re.sub(r'<p>(<img[^>]+>)</p>\s*<p>(<strong>Supplementary Figure S2.*?</p>)',
    r'<figure>\1<figcaption>\2</figcaption></figure>',html,flags=re.S)
html=html.replace('</p></figcaption>','</figcaption>')
figure=ROOT/'figures/figS2_full_scale_resources.svg'
html=html.replace('src="figures/figS2_full_scale_resources.svg"','src="data:image/svg+xml;base64,'+base64.b64encode(figure.read_bytes()).decode()+'"')
html=html.replace('<table>', '<table class="resource-component-table">')
(OUT/'section.html').write_text(html)
validation=dict(status='passed',analysis_sha256=sha(OUT/'analysis.json'),
    source_files=len(sources),conditional_hosts=3000,conditional_cores=24000,
    corrected_receive_policy='Generator pre-splits signed-i32 batches; 641 GiB is dense CSR per rank, not a full-scale receive buffer.',
    section_sha256=sha(OUT/'section.html'),figure_sha256=sha(figure),
    script_sha256=sha(Path(__file__)),simulations_started=0,remote_jobs_started=0,
    scope=analysis['scope'])
(ROOT/'validation/full_scale_resource_analysis.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation))
