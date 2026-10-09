"""Serial accepted capacity observations through 1%; bounded stages, no retries."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
STATE={'schema':'atlas-1pct-sweep-state-v1','started_epoch':time.time(),'accepted_hosts':[],
       'endpoint_neurons':860000000,'status':'running'}


def save():
    (HERE/'sweep-state.json').write_text(json.dumps(STATE,indent=2)+'\n')


def read(name):
    return json.loads((HERE/name).read_text())


def run(action,case=None,hosts=30):
    STATE.update(action=action,case=case,hosts=hosts,last_update_epoch=time.time())
    save()
    print(json.dumps({k:STATE[k] for k in ['action','case','hosts','last_update_epoch']}),flush=True)
    argv=[sys.executable,str(HERE/'control.py'),action]
    if case:
        argv.append(case)
    argv.extend(['--hosts',str(hosts)])
    subprocess.run(argv,check=True,stdout=subprocess.DEVNULL)


def wait(stage,hosts,seconds):
    deadline=time.monotonic()+seconds
    while True:
        run('status',stage,hosts)
        record=read(stage+'-status.json');guard=record['guard']
        if guard and 'returncode' in guard:
            assert guard['returncode']==0,record['log'][-3000:]
            return
        assert time.monotonic()<deadline,stage+' observer deadline'
        time.sleep(35)


def usage_seconds(g):
    def get(phase):
        return int(dict(line.split() for line in g[phase]['cpu.stat'].splitlines())['usage_usec'])
    return (get('after')-get('before'))/1e6


def accept(case,hosts):
    resources=read(case+'-resources.json');audit=read(case+'-audit.json')
    launched=read(case+'-launch-result.json');pilot=read(f'pilot-weak1pct-h{hosts:02d}-v1-validation.json')
    g=read('audit-'+case+'-status.json')['guard']
    assert all(x['passed'] for x in [resources,audit,pilot]) and g['returncode']==0
    expected=next(x for x in P['weak_grid'] if x['hosts']==hosts)
    assert audit['neurons']==expected['neurons'] and audit['connections']==expected['edges']
    assert audit['duration_ms']==100 and audit['mean_indegree']==1000
    assert audit['output_files']['mpi-runtime.json']['sha256']==resources['rows'][0]['runtime_sha256']
    assert len(resources['rows'])==hosts and set(audit['runtime']['processor_names'])==set(r['host'] for r in resources['rows'])
    peaks=[int(row['guards']['proxy-'+str(i)]['after']['memory.peak'])/2**30
           for i,row in enumerate(resources['rows'])]
    cpus=[usage_seconds(row['guards']['proxy-'+str(i)]) for i,row in enumerate(resources['rows'])]
    prep=resources['rows'][0]['prepared']
    post=read(case+'-post-inventory.json')
    assert all(not row['active_mpi_services'] for row in post['rows'])
    caps=[int(row['guards']['proxy-'+str(i)]['before']['memory.max'])//2**30 for i,row in enumerate(resources['rows'])]
    summary=dict(worker_memory_caps_gib=caps,protocol_revision='v2' if case.endswith('-v2') else 'v1',schema='atlas-weak-scaling-accepted-v1',passed=True,case=case,
        hosts=hosts,ranks=hosts*8,physical_worker_cores=hosts*8,
        neurons=audit['neurons'],connections=audit['connections'],mean_indegree=1000,
        duration_ms=100,dt_ms=.1,precision='reference-f64',
        neuron_count_percent=audit['neurons']/86e9*100,
        launch_wall_seconds=launched['wall_seconds'],
        initialization_max_seconds=audit['stage_max_seconds'][0],
        initialization_wait_max_seconds=audit['stage_max_seconds'][1],
        simulation_max_seconds=audit['stage_max_seconds'][2],
        collection_max_seconds=audit['stage_max_seconds'][4],
        spike_exchange_max_seconds=audit['spike_exchange_seconds_minmax'][1],
        prepare_wall_seconds=prep['wall_seconds'],compile_seconds=prep['compile_seconds'],
        prepare_guard_wall_seconds=read('prepare-'+case+'-status.json')['guard']['wall_seconds'],
        worker_cgroup_peak_gib=peaks,maximum_worker_cgroup_peak_gib=max(peaks),
        nonsimultaneous_sum_host_peaks_gib=sum(peaks),worker_cpu_seconds=cpus,
        worker_cpu_seconds_sum=sum(cpus),spikes=audit['spikes'],
        delivered_synaptic_events=audit['delivered_synaptic_events'],
        output_bytes=sum(f['bytes'] for f in audit['output_files'].values()),
        source_archive_sha256=P['source_archive_sha256'],
        prepare_script_sha256=P['prepare_script_sha256'],launcher_sha256=P['launcher_sha256'],
        model_sha256=prep['files']['model.json']['sha256'],plan_sha256=audit['plan_sha256'],
        full_audit_file_sha256=audit['full_audit_file_sha256'],
        numerical_pilot_checks=pilot['checks'],numerical_pilot_max_difference=pilot['max_state_absolute_difference'],
        timing_observations=1,cleanup_passed=True,
        scope='Count-normalized synthetic recurrent E/I weak scaling; not anatomical brain reconstruction, maximum capacity or full 86B feasibility.')
    (HERE/(case+'-accepted.json')).write_text(json.dumps(summary,indent=2)+'\n')
    STATE['accepted_hosts'].append(hosts)
    STATE['last_accepted']=summary
    save()
    print(json.dumps({'accepted':True,'hosts':hosts,'neurons':summary['neurons'],
                      'launch_seconds':summary['launch_wall_seconds'],
                      'maximum_host_peak_gib':max(peaks)}),flush=True)


def main():
    old=read('sweep-state.json')
    assert old['status']=='stopped_on_failure' and old['action']=='collect-audit' and old['case']=='weak1pct-h24-v2' and old['accepted_hosts']==[3,6,12]
    assert read('audit-weak1pct-h24-v2-status.json')['guard']['returncode']==0
    STATE.update(old,status='running',resumed_after='Observer-summary compaction only; full remote audit and all numerical outputs retained unchanged')
    STATE.pop('error',None);save()
    case='weak1pct-h24-v2'
    run('collect-audit',case,24);run('inventory',case+'-post',24);accept(case,24)
    hosts=30;pilot='pilot-weak1pct-h30-v1';case='weak1pct-h30-v2'
    run('prepare',pilot,hosts);wait('prepare-'+pilot,hosts,7350)
    run('prepared',pilot,hosts)
    run('reference',pilot,hosts);wait('reference-'+pilot,hosts,1250)
    run('deploy',pilot,hosts);run('launch',pilot,hosts)
    run('validate',pilot,hosts);run('collect',pilot,hosts)
    run('prepare',case,hosts);wait('prepare-'+case,hosts,7350)
    run('prepared',case,hosts);run('readback',hosts=hosts)
    run('deploy',case,hosts);run('launch',case,hosts)
    run('collect',case,hosts);run('audit',case,hosts)
    wait('audit-'+case,hosts,1850);run('collect-audit',case,hosts)
    run('inventory',case+'-post',hosts);accept(case,hosts)
    assert STATE['last_accepted']['neurons']==860000000
    STATE.update(status='endpoint_complete',completed_epoch=time.time());save()


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        STATE.update(status='stopped_on_failure',error=str(exc)[-4000:],last_update_epoch=time.time())
        save();traceback.print_exc();raise
