"""Build a comparison exclusively from accepted, audited observations."""
from pathlib import Path
import csv, hashlib, json, os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/atlas-owner30-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    rows=[]
    case_map=json.loads((HERE/'sweep-state.json').read_text()).get('case_map',{})
    for h in [3,6,12,24,30]:
        path=HERE/(case_map.get(str(h),f'weak1pct-h{h:02d}-v1')+'-accepted.json')
        if not path.exists():
            continue
        row=json.loads(path.read_text())
        assert row['passed'] and row['cleanup_passed'] and row['hosts']==h
        audit=json.loads((HERE/(row['case']+'-audit.json')).read_text())
        assert audit['passed'] and audit['neurons']==row['neurons'] and audit['connections']==row['connections']
        assert row['duration_ms']==100 and row['mean_indegree']==1000 and row['physical_worker_cores']==h*8
        row['accepted_record_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        row.setdefault('worker_memory_caps_gib',[768]+[665]*(h-1))
        row.setdefault('host_memory_reserve_gib',64)
        row['coordinator_cgroup_peak_gib']=row['worker_cgroup_peak_gib'][0]
        row['maximum_noncoordinator_peak_gib']=max(row['worker_cgroup_peak_gib'][1:])
        rows.append(row)
    assert rows,'No accepted observations yet'
    first=rows[0]
    for row in rows:
        row['simulation_weak_efficiency']=first['simulation_max_seconds']/row['simulation_max_seconds']
        row['launch_weak_efficiency']=first['launch_wall_seconds']/row['launch_wall_seconds']
    fields=['hosts','ranks','physical_worker_cores','neuron_count_percent','neurons','connections','host_memory_reserve_gib','prepare_wall_seconds','compile_seconds','launch_wall_seconds','initialization_max_seconds','simulation_max_seconds','collection_max_seconds','spike_exchange_max_seconds','maximum_worker_cgroup_peak_gib','coordinator_cgroup_peak_gib','maximum_noncoordinator_peak_gib','worker_cpu_seconds_sum','spikes','delivered_synaptic_events','output_bytes','simulation_weak_efficiency','launch_weak_efficiency','accepted_record_sha256']
    with (HERE/'accepted-weak-scaling.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    report=dict(passed=True,endpoint_complete=rows[-1]['neurons']==860000000,observations=rows,
        scope='Synthetic recurrent E/I; count normalized to 86B neurons. Mean indegree 1000, reference-f64, 100ms, 8 distinct physical worker cores per host. One observation per size. Different graph realizations; stage maxima may occur on different ranks. No anatomical human-brain reconstruction or full-86B feasibility claim.',
        timing_scope='Launch-to-exit excludes prior export, compilation, deployment and independent post-run audit. Simulation is maximum local simulation interval, including communication. Exchange includes synchronization and packing, not just network transfer.',
        memory_scope='Worker cgroup peak includes charged cache. Peaks are per host and occur at different times; their sum is not a simultaneous cluster peak.',
        scaling_limit='Global receive buffers and CSR offsets scale with global neuron count on each rank, so constant useful local work does not imply constant memory. Weak efficiency here is descriptive; repeated timing and strong scaling remain separate qualifications.')
    (HERE/'accepted-weak-scaling.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(12.8,3.9),layout='constrained');x=[r['hosts'] for r in rows]
    for key,label in [('launch_wall_seconds','Launch to exit'),('initialization_max_seconds','Initialization max'),('simulation_max_seconds','Simulation max')]:
        axes[0].plot(x,[r[key] for r in rows],'o-',label=label)
    axes[0].set_ylabel('Seconds');axes[0].legend(fontsize=8);axes[0].set_title('(a) Measured intervals')
    axes[1].plot(x,[r['maximum_noncoordinator_peak_gib'] for r in rows],'o-',color='#267770',label='Other hosts max');axes[1].plot(x,[r['coordinator_cgroup_peak_gib'] for r in rows],'s-',color='#c57932',label='Coordinator host');axes[1].fill_between(x,[min(r['worker_memory_caps_gib'][1:]) for r in rows],[max(r['worker_memory_caps_gib'][1:]) for r in rows],color='#777777',alpha=.12,label='Other-host cap range');axes[1].plot(x,[max(r['worker_memory_caps_gib'][1:]) for r in rows],'--',color='#777777');axes[1].axhline(768,ls=':',color='#777777',label='Coordinator cap');axes[1].set_ylabel('GiB');axes[1].legend(fontsize=8);axes[1].set_title('(b) Maximum host cgroup peak')
    axes[2].plot(x,[r['simulation_weak_efficiency'] for r in rows],'o-',label='Simulation');axes[2].plot(x,[r['launch_weak_efficiency'] for r in rows],'s-',label='Launch');axes[2].axhline(1,ls='--',color='#777777');axes[2].set_ylabel('Baseline time / measured time');axes[2].legend(fontsize=8);axes[2].set_title('(c) Descriptive weak efficiency')
    for ax in axes:
        ax.set_xlabel('Hosts (8 worker cores per host)');ax.set_xticks(x);ax.grid(alpha=.2)
    fig.savefig(HERE/'accepted-weak-scaling.png',dpi=180);fig.savefig(HERE/'accepted-weak-scaling.svg');plt.close(fig)
    print(json.dumps({'accepted_points':len(rows),'endpoint_complete':report['endpoint_complete']}))

if __name__=='__main__':main()
