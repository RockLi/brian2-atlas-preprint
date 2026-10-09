"""Generate a bounded capacity report only after output and resource acceptance."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'brain-fraction-86m-20261007'
def read(name):return json.loads((HERE/name).read_text())

def main():
    protocol=read('protocol-v2.json');case=protocol['case']
    audit=read(case+'-audit.json');resources=read(case+'-resources.json');launch=read(case+'-launch-result.json')
    old=json.loads((OLD/'final-report.json').read_text())
    pilot=json.loads((OLD/'pilot-owner30-validation.json').read_text())
    assert all(x['passed'] for x in [audit,resources,old,pilot,read(case+'-deployment-audit.json'),read('final-cleanup.json')])
    assert read('engine-source-readback.json')['passed']
    assert read('audit-'+case+'-status.json')['guard']['returncode']==0
    assert launch['error'] is None and all(x==0 for x in launch['returncodes'].values())
    assert audit['neurons']==protocol['neurons'] and audit['connections']==protocol['connections']
    runtime=audit['runtime'];assert runtime['ranks']==30 and len(set(runtime['processor_names']))==30
    assert runtime['rank_cpu_ids']==[1]*30 and all(runtime['rank_work'][3*i]>0 for i in range(30))
    assert len(runtime['procedural_topology'])==900
    assert {x['construction'] for x in runtime['procedural_topology']}=={'target-owner-local'}
    peaks=[]
    for i,row in enumerate(resources['rows']):
        g=row['guards']['proxy-'+str(i)]
        assert g['admitted'] and g['returncode']==0 and g['after']['memory.swap.max']=='0'
        assert 'oom 0' in g['after']['memory.events'] and 'oom_kill 0' in g['after']['memory.events']
        assert len(row['rank_times'])==1 and all('Exit status: 0' in x for x in row['rank_times'].values())
        peaks.append(int(g['after']['memory.peak'])/2**30)
    prepared=resources['rows'][0]['prepared']
    summary={'schema':'atlas-128m-capacity-final-report-v1','passed':True,'neurons':audit['neurons'],
             'connections':audit['connections'],'mean_indegree':audit['mean_indegree'],'duration_ms':audit['duration_ms'],
             'dt_ms':.1,'precision':'reference-f64','hosts':30,'ranks':30,'physical_worker_cores':30,
             'populations':30,'projections':900,'reference_neuron_count':86_000_000_000,
             'neuron_count_percent':audit['neurons']/86_000_000_000*100,
             'launch_wall_seconds':launch['wall_seconds'],'initialization_max_seconds':audit['stage_max_seconds'][0],
             'simulation_max_seconds':audit['stage_max_seconds'][2],
             'collection_max_seconds':audit['stage_max_seconds'][4],
             'spike_exchange_max_seconds':audit['spike_exchange_seconds_minmax'][1],
             'prepare_wall_seconds':prepared['wall_seconds'],
             'worker_cgroup_peak_gib':peaks,'maximum_worker_cgroup_peak_gib':max(peaks),
             'spikes':audit['spikes'],'delivered_synaptic_events':audit['delivered_synaptic_events'],
             'output_bytes':sum(x['bytes'] for x in audit['output_files'].values()),
             'model_sha256':prepared['files']['model.json']['sha256'],'plan_sha256':audit['plan_sha256'],
             'engine_archive_sha256':protocol['source_archive_sha256'],
             'experiment_script_sha256':protocol['script_sha256'],'initial_value_budget':600_000_000,
             'prior_reference_pilot_checks':pilot['checks'],'prior_reference_pilot_max_difference':pilot['max_state_absolute_difference'],
             'prior_preparation_failed':True,'engine_changed_from_86m':False,
             'scientific_human_brain_equivalence_claim':False,'full_scale_86b_feasibility_validated':False,
             'maximum_capacity_measured':False,'strong_or_weak_scaling_claim':False,
             'competing_simulator_speed_claim':False}
    assert hashlib.sha256((HERE/'prepare-owner30-128m-b600m.py').read_bytes()).hexdigest()==summary['experiment_script_sha256']
    (HERE/'final-report.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    for excitatory,label,color in [(True,'Excitatory (80%)','#3973ac'),(False,'Inhibitory (20%)','#c87542')]:
        pops=[p for p in audit['populations'] if (int(p['name'].rsplit('_',1)[1])<24)==excitatory]
        count=sum(p['neurons'] for p in pops);assert count==audit['neurons']*(4 if excitatory else 1)//5
        rates=np.sum([p['spikes_per_1ms_bin'] for p in pops],axis=0)/(count*.001)
        axes[0].plot(np.arange(20)*5+2.5,rates.reshape(20,5).mean(1),lw=1.8,
                     linestyle='-' if excitatory else '--',label=label,color=color)
    axes[0].set(xlabel='Biological time (ms)',ylabel='Population firing rate (Hz)',title='a  Measured activity')
    axes[0].legend(fontsize=8,frameon=False)
    times=[summary['initialization_max_seconds'],summary['simulation_max_seconds'],summary['launch_wall_seconds']]
    axes[1].bar(['Initialization','Simulation','Launch to exit'],times,color=['#6b8bb2','#428f70','#d19a49'])
    axes[1].set(ylabel='Wall time (s)',title='b  Measured execution time',ylim=(0,max(times)*1.18))
    axes[1].tick_params(axis='x',labelsize=8)
    for i,t in enumerate(times):axes[1].text(i,t+max(times)*.02,f'{t:.1f}',ha='center',fontsize=9)
    axes[2].bar(np.arange(1,31),peaks,color='#428f70',width=.8)
    axes[2].axhline(256,color='#a35656',ls='--',lw=1,label='Per-host memory cap')
    axes[2].set(xlabel='Host in launch order',ylabel='Worker cgroup memory peak (GiB)',title='c  Measured resource use',ylim=(0,280),xticks=[1,5,10,15,20,25,30])
    axes[2].legend(fontsize=8,frameon=False)
    fig.suptitle('128 million neurons · 128 billion connections · 30 MPI ranks on 30 hosts',fontsize=13)
    for ext in ['png','svg','pdf']:fig.savefig(HERE/('capacity-result.'+ext),dpi=180)
    plt.close(fig)
    s=summary
    report=f'''# brian2-atlas: 128-million-neuron capacity experiment

**Completed and audited:** {s['neurons']:,} neurons and {s['connections']:,} recurrent connections, 100 ms biological time, dt 0.1 ms, reference-f64. Thirty populated MPI ranks ran on thirty physical hosts, one physical CPU core per host.

| Measured quantity | Previous 86M configuration | Current 128M configuration |
| --- | ---: | ---: |
| Neurons | 86,000,000 | 128,000,000 |
| Connections | 86,000,000,000 | 128,000,000,000 |
| Biological duration | 100 ms | 100 ms |
| Launch to exit | {old['launch_wall_seconds']:.3f} s | {s['launch_wall_seconds']:.3f} s |
| Maximum local initialization | {old['initialization_max_seconds']:.3f} s | {s['initialization_max_seconds']:.3f} s |
| Maximum local simulation | {old['simulation_max_seconds']:.3f} s | {s['simulation_max_seconds']:.3f} s |
| Maximum per-host worker cgroup peak | {old['maximum_worker_cgroup_peak_gib']:.3f} GiB | {max(peaks):.3f} GiB |
| Recorded spikes | {old['spikes']:,} | {s['spikes']:,} |
| Delivered synaptic events | {old['delivered_synaptic_events']:,} | {s['delivered_synaptic_events']:,} |

![Activity, execution time and node memory](capacity-result.png)

## Configuration and acceptance

The exact engine snapshot and whole-population target-owner-local construction from the passed 86M experiment were reused. The isolated engine archive SHA-256 is `{s['engine_archive_sha256']}`. No new engine/default-cap or numerical-kernel changes were made. The experiment script SHA-256 is `{s['experiment_script_sha256']}`. The 30-pool allocation, 80/20 E/I split, mean indegree 1,000, model/projection seeds, weight/delay distributions, Euler dynamics and monitoring policy follow the prior [workload description](../brain-fraction-86m-20261007/RESULTS.md). Population sizes and draw counts change with N, so the realized graphs differ. This is a pair of tested capacity configurations, not a strong/weak scaling curve or speedup against another simulator.

Each rank owns approximately 4.267 billion connections, while the largest projection has {protocol['largest_projection_edges']:,} edges. The explicit cumulative local-edge guard is 4.29 billion, below the controller's u32 bound; individual projection indices also fit u32 and Brian2's signed-int32 Synapses.N. Rank ownership and exact projection counts are checked in the complete output audit. The engine's existing opt-in neuron ceiling is 128 million; the default remains one million.

The first preparation was rejected before simulation because the explicit 400-million initial-value budget was below the approximately 512-million neuron-state/refractory values needed. The failure is retained under `owner30-128m` and `protocol.json`. A new prospective protocol (`protocol-v2.json`) set this existing configurable budget to 600 million and used a separate output directory. The main engine and 64 GiB preparation memory cap were unchanged. The preparation completed in {s['prepare_wall_seconds']:.3f} s. Initial-value budget is a validation ceiling, not a physical memory reservation.

The earlier identical-algorithm 24,000-neuron / 24-million-connection, 30-host pilot passed {pilot['checks']} independent single-process Rust-reference checks, with exact spikes/events and zero state/trace difference. It is reused as the small-model numerical gate; it is not a new reference execution at 128M. The present full-output audit checked finite states/traces, refractory behavior, spike counts, total neurons/edges/events, all 30 populated ranks, all 30 host identities, CPU placement and exclusive target-owner construction. All workers and 31 launch guards exited zero, without OOM or swap. Final cgroup cleanup passed.

Worker limits were 256 GiB and one CPU per host, an 8 GiB per-file ceiling, 128 GiB disk reserve and an 1,800 s timeout. Preparation used 64 GiB/four CPUs; the separate output audit used 128 GiB/four CPUs. The static necessary-array estimate was 80.43 GiB per rank, excluding states, queues, scratch, recording, result buffers, MPI and charged cache. The prospective 95.65 GiB linear peak estimate was illustrative, not a guaranteed peak. Actual output size is {s['output_bytes']:,} bytes. Complete output, input/output hashes, deployment readbacks, per-rank timing and resource records are retained. Raw artifacts remain in the shared experiment base's `{case}/` directory; leader storage uses the recorded data mount.

## Scientific scope and timing

This neuron count corresponds to **{s['neuron_count_percent']:.5f}% of an 86-billion-neuron reference count**. It is a synthetic recurrent capacity workload, not anatomical or functional human-brain reconstruction. The study is self-funded; full 86-billion-neuron execution remains unevaluated and its feasibility unvalidated. This result does not establish maximum capacity or that adding machines alone suffices. Going beyond 128M requires a separately validated budget/index/output and layout plan. Only 100 ms was tested at this size, and only 30 worker cores were used; the full CPU resources of these hosts were not saturated.

Launch-to-exit includes initialization, simulation and output, excluding prior export/compile/deployment. Local stage maxima occur on potentially different ranks and must not be summed as end-to-end duration. Maximum rank spike-exchange time was {s['spike_exchange_max_seconds']:.3f} s and is already included in simulation. Cgroup peaks include proxy/rank and charged cache; they are not pure allocator RAM. Activity is displayed in 5 ms bins.

Suggested manuscript sentence: In a connected synthetic excitatory-inhibitory workload, brian2-atlas simulated 128 million neurons with 128 billion recurrent connections for 100 ms using 30 MPI ranks on 30 physical hosts. Execution completed in {s['launch_wall_seconds']:.2f} s including initialization and output, with a maximum per-host worker cgroup memory peak of {max(peaks):.2f} GiB. Full spike histories and sampled state traces were retained and audited. This corresponds to approximately 0.149% of an 86-billion reference neuron count, without implying human-brain equivalence or validated full-scale feasibility.

Evidence: protocol.json, protocol-v2.json, script-staging.json, script-staging-v2.json, {case}-deployment-audit.json, {case}-launch-result.json, {case}-resources.json, {case}-audit.json, audit-{case}-status.json, final-cleanup.json and final-report.json. The engine/numerical gates link to the unchanged preceding experiment's records.
'''
    (HERE/'RESULTS.md').write_text(report)
    manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in HERE.iterdir() if p.is_file() and p.name!='report-manifest.json'}
    (HERE/'report-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
