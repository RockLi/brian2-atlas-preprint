"""Report measured capacity after numerical, resource and output acceptance."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CASE = 'capacity-256m-owner60'


def read(name):
    return json.loads((HERE / name).read_text())


def cpu_seconds(guard):
    def usage(values):
        return int(dict(line.split() for line in values['cpu.stat'].splitlines())['usage_usec'])
    return (usage(guard['after']) - usage(guard['before'])) / 1e6


def main():
    protocol = read('protocol.json')
    audit = read(CASE + '-audit.json')
    resources = read(CASE + '-resources.json')
    launch = read(CASE + '-launch-result.json')
    for name in ['engine-source-readback.json', 'final-cleanup.json',
                 CASE + '-deployment-audit.json', 'pilot-owner60-n256-placement-proof.json',
                 'pilot-owner60-n256-resources.json', 'pilot-owner60-n256-validation.json']:
        assert read(name)['passed'], name
    assert audit['passed'] and resources['passed']
    assert read('audit-' + CASE + '-status.json')['guard']['returncode'] == 0
    assert launch['error'] is None and all(x == 0 for x in launch['returncodes'].values())
    assert audit['neurons'] == protocol['neurons'] and audit['connections'] == protocol['connections']
    runtime = audit['runtime']
    assert runtime['ranks'] == 60 and len(set(runtime['processor_names'])) == 30
    assert runtime['rank_cpu_ids'] == [v for _ in range(30) for v in [1, 49]]
    assert all(runtime['rank_work'][3*i] > 0 for i in range(60))
    assert len(runtime['procedural_topology']) == 3600
    assert {t['construction'] for t in runtime['procedural_topology']} == {'target-owner-local'}
    peaks, cpus = [], []
    for i, row in enumerate(resources['rows']):
        g = row['guards']['proxy-' + str(i)]
        assert g['admitted'] and g['returncode'] == 0 and g['after']['memory.swap.max'] == '0'
        events = dict(line.split() for line in g['after']['memory.events'].splitlines())
        assert events['oom'] == events['oom_kill'] == '0'
        assert len(row['rank_times']) == 2 and all('Exit status: 0' in t for t in row['rank_times'].values())
        peaks.append(int(g['after']['memory.peak']) / 2**30)
        cpus.append(cpu_seconds(g))
    prepared = resources['rows'][0]['prepared']
    pilot = read('pilot-owner60-n256-validation.json')
    summary = dict(
        schema='atlas-256m-capacity-final-report-v1', passed=True,
        neurons=audit['neurons'], connections=audit['connections'], mean_indegree=1000,
        duration_ms=100, dt_ms=0.1, precision='reference-f64', hosts=30, ranks=60,
        physical_worker_cores=60, populations=60, projections=3600,
        reference_neuron_count=86_000_000_000,
        neuron_count_percent=audit['neurons']/86_000_000_000*100,
        launch_wall_seconds=launch['wall_seconds'],
        initialization_max_seconds=audit['stage_max_seconds'][0],
        simulation_max_seconds=audit['stage_max_seconds'][2],
        collection_max_seconds=audit['stage_max_seconds'][4],
        spike_exchange_max_seconds=audit['spike_exchange_seconds_minmax'][1],
        prepare_wall_seconds=prepared['wall_seconds'], compile_seconds=prepared['compile_seconds'],
        worker_cgroup_peak_gib=peaks, maximum_worker_cgroup_peak_gib=max(peaks),
        sum_host_peak_gib_non_simultaneous=sum(peaks),
        worker_cpu_seconds=cpus, worker_cpu_seconds_sum=sum(cpus),
        spikes=audit['spikes'], delivered_synaptic_events=audit['delivered_synaptic_events'],
        output_bytes=sum(f['bytes'] for f in audit['output_files'].values()),
        model_sha256=prepared['files']['model.json']['sha256'], plan_sha256=audit['plan_sha256'],
        engine_archive_sha256=protocol['source_archive_sha256'],
        test_fixture_archive_sha256=read('test-fixture-stage.json')['archive_sha256'],
        reference_binary_sha256=read('engine-source-readback.json')['reference_binary_sha256'],
        experiment_script_sha256=protocol['script_sha256'], initial_value_budget=1_200_000_000,
        numerical_pilot_checks=pilot['checks'], numerical_pilot_max_difference=pilot['max_state_absolute_difference'],
        observations_per_configuration=1, strong_or_weak_scaling_claim=False,
        scientific_human_brain_equivalence_claim=False, maximum_capacity_measured=False,
        full_scale_86b_feasibility_validated=False, competing_simulator_speed_claim=False,
        source_change_scope='Isolated opt-in neuron ceiling 128M to 256M; default 1M unchanged. Numerical kernels unchanged. New 60-pool workload and bounded preparation budgets.',
        timing_scope='Launch excludes export/compile/deployment. Stage maxima can occur on different ranks and must not be summed. Exchange is included in simulation.',
        memory_scope='Worker cgroup peaks include proxy, ranks and charged cache. Sum of individual host peaks is not a simultaneous aggregate peak.',
        cpu_scope='Sum of per-host worker cgroup CPU usage deltas; excludes preparation, controller and output audit.'
    )
    assert hashlib.sha256((HERE/'prepare-owner60.py').read_bytes()).hexdigest() == protocol['script_sha256']
    (HERE/'final-report.json').write_text(json.dumps(summary, indent=2)+'\n')

    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    for excitatory, label, color in [(True,'Excitatory (80%)','#3973ac'), (False,'Inhibitory (20%)','#c87542')]:
        pops = [p for p in audit['populations'] if (int(p['name'].rsplit('_',1)[1]) < 48) == excitatory]
        count = sum(p['neurons'] for p in pops)
        assert count == audit['neurons']*(4 if excitatory else 1)//5
        rates = np.sum([p['spikes_per_1ms_bin'] for p in pops], axis=0)/(count*.001)
        axes[0].plot(np.arange(20)*5+2.5, rates.reshape(20,5).mean(1), color=color,
                     linestyle='-' if excitatory else '--', lw=1.8, label=label)
    axes[0].set(xlabel='Biological time (ms)', ylabel='Population firing rate (Hz)', title='a  Measured activity')
    axes[0].legend(fontsize=8, frameon=False)
    times = [summary['initialization_max_seconds'], summary['simulation_max_seconds'], summary['launch_wall_seconds']]
    axes[1].bar(['Initialization','Simulation','Launch to exit'], times, color=['#6b8bb2','#428f70','#d19a49'])
    axes[1].set(ylabel='Wall time (s)', title='b  Measured execution time', ylim=(0,max(times)*1.18))
    axes[1].tick_params(axis='x', labelsize=8)
    for i, t in enumerate(times):
        axes[1].text(i, t+max(times)*.02, f'{t:.1f}', ha='center', fontsize=9)
    axes[2].bar(np.arange(1,31), peaks, color='#428f70', width=.8)
    axes[2].axhline(256, color='#a35656', ls='--', lw=1, label='Per-host memory cap')
    axes[2].set(xlabel='Host in launch order', ylabel='Worker cgroup memory peak (GiB)',
                title='c  Measured resource use', ylim=(0,280), xticks=[1,5,10,15,20,25,30])
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle('256 million neurons · 256 billion connections · 60 MPI ranks on 30 hosts', fontsize=13)
    for ext in ['png','svg']:
        fig.savefig(HERE/('capacity-result.'+ext), dpi=180)
    plt.close(fig)

    rows = []
    cases = [('brain-fraction-86m-20261007','major-owner30'),
             ('brain-fraction-128m-20261007','owner30-128m-b600m'), (HERE.name,CASE)]
    for directory, case in cases:
        root = HERE.parent/directory
        s = json.loads((root/'final-report.json').read_text())
        r = json.loads((root/(case+'-resources.json')).read_text())
        usage = sum(cpu_seconds(row['guards']['proxy-'+str(i)]) for i,row in enumerate(r['rows']))
        rows.append({k:s[k] for k in ['neurons','connections','neuron_count_percent','hosts','ranks',
                                     'physical_worker_cores','duration_ms',
                                     'launch_wall_seconds','initialization_max_seconds','simulation_max_seconds',
                                     'spike_exchange_max_seconds','maximum_worker_cgroup_peak_gib','output_bytes']}
                    | {'prepare_wall_seconds':r['rows'][0]['prepared']['wall_seconds'],
                       'worker_cpu_seconds_sum':usage, 'observations':1, 'controlled_scaling':False})
    with (HERE/'accepted-capacity-comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (HERE/'accepted-capacity-comparison.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
