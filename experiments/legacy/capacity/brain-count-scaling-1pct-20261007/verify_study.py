"""Verify terminal evidence for all five accepted layouts without remote work."""
from pathlib import Path
import hashlib,json,math
HERE=Path(__file__).resolve().parent

def read(name):return json.loads((HERE/name).read_text())

def main():
    state=read('sweep-state.json');p=read('protocol.json')
    assert state['status']=='endpoint_complete' and state['accepted_hosts']==[3,6,12,24,30]
    assert hashlib.sha256((HERE/'source.tar.gz').read_bytes()).hexdigest()==p['source_archive_sha256']
    assert read('engine-source-readback.json')['passed'] and read('engine-source-readback.json')['files_checked']==156
    assert read('launcher-tests.json')['passed']
    for name in ['build-scale1pct-v1','budget-scale1pct-v1']:
        assert read(name+'-status.json')['guard']['returncode']==0
    checked=[]
    for h in [3,6,12,24,30]:
        case=state.get('case_map',{}).get(str(h),f'weak1pct-h{h:02d}-v1')
        accepted=read(case+'-accepted.json');pilot=accepted.get('numerical_pilot_case',f'pilot-weak1pct-h{h:02d}-v1')
        audit=read(case+'-audit.json');resources=read(case+'-resources.json');launch=read(case+'-launch-result.json');validation=read(pilot+'-validation.json')
        assert all(r['passed'] for r in [accepted,audit,resources,validation])
        assert accepted['cleanup_passed'] and launch['error'] is None and all(v==0 for v in launch['returncodes'].values())
        assert accepted['neurons']==h//3*86_000_000 and accepted['connections']==accepted['neurons']*1000
        assert accepted['duration_ms']==100 and accepted['dt_ms']==.1 and accepted['precision']=='reference-f64'
        assert accepted['ranks']==accepted['physical_worker_cores']==h*8
        assert validation['checks']==h*80+3 and validation['max_state_absolute_difference']==0
        assert accepted['full_audit_file_sha256']==audit['full_audit_file_sha256']
        assert audit['output_files']['mpi-runtime.json']['sha256']==resources['rows'][0]['runtime_sha256']
        assert accepted['output_bytes']==sum(f['bytes'] for f in audit['output_files'].values())
        assert accepted['spikes']==audit['spikes']>0 and accepted['delivered_synaptic_events']==audit['delivered_synaptic_events']>0
        assert accepted['launch_wall_seconds']==launch['wall_seconds']<p['simulation_timeout_seconds']
        assert all(not r['active_mpi_services'] for r in read(case+'-post-inventory.json')['rows'])
        runtime=resources['rows'][0]['runtime']
        assert runtime['exchange_calls_per_rank']==1000
        assert len(resources['rows'])==h and len(runtime['rank_cpu_ids'])==h*8
        all_rank_ids=set();peaks=[]
        for i,row in enumerate(resources['rows']):
            g=row['guards']['proxy-'+str(i)]
            cap=accepted.get('worker_memory_caps_gib',[768]+[665]*(h-1))[i]
            assert g['admitted'] and g['returncode']==0
            assert int(g['before']['memory.max'])==cap*2**30
            assert all(role['reserved_host_memory_bytes']==accepted.get('host_memory_reserve_gib',64)*2**30 for role in row['guards'].values())
            assert g['before']['memory.swap.max']==g['after']['memory.swap.max']=='0'
            events=dict(line.split() for line in g['after']['memory.events'].splitlines())
            assert all(events[k]=='0' for k in ['oom','oom_kill','oom_group_kill'])
            assert int(g['after']['memory.peak'])<=cap*2**30
            assert runtime['processor_names'][8*i:8*i+8]==[row['host']]*8
            assert sorted(runtime['rank_cpu_ids'][8*i:8*i+8])==p['cpu_ids']
            ids={int(name.removeprefix('rank-').removesuffix('.time')) for name in row['rank_times']}
            assert ids==set(range(8*i,8*i+8)) and not ids & all_rank_ids
            assert all('Exit status: 0' in text and 'Swaps: 0' in text for text in row['rank_times'].values())
            all_rank_ids|=ids;peaks.append(int(g['after']['memory.peak'])/2**30)
        assert peaks==accepted['worker_cgroup_peak_gib']
        assert all(math.isfinite(t) and t>0 for t in accepted['worker_cpu_seconds'])
        for stage in ['prepare-'+case,'audit-'+case]:
            guard=read(stage+'-status.json')['guard']
            assert guard['returncode']==0
            assert int(guard['before']['memory.max'])==320*2**30
            assert int(guard['after']['memory.peak'])<=320*2**30
        checked.append(dict(case=case,hosts=h,neurons=accepted['neurons'],passed=True))
        if accepted.get('protocol_revision')=='v3':
            policy=read('protocol-revision-v3.json');staged=read('policy48-stage.json')
            assert policy['status']=='activated' and policy['reserve_gib']==48 and h==30
            assert staged['passed'] and staged['tests']==11 and len(staged['rows'])==30
            assert read('policy48-kernel-smoke.json')['passed']
            assert accepted['admission_guard_sha256']==staged['guard_sha256']==policy['guard_sha256']
            assert hashlib.sha256((HERE/'guard-host-reserve48-v3.py').read_bytes()).hexdigest()==policy['guard_sha256']
            assert hashlib.sha256((HERE/'control_policy48.py').read_bytes()).hexdigest()==read('policy48-controller-identity.json')['policy48_controller_sha256']
            assert read(case+'-input-reuse.json')['passed'] and read(case+'-input-reuse.json')['numerical_input_changes'] is False
    assert checked[-1]['neurons']==860_000_000
    report=dict(schema='atlas-1pct-terminal-qa-v1',passed=True,endpoint_complete=True,cases=checked,
        evidence_scope='Five single-observation CPU weak-scaling layouts; all per-layout numerical pilots and complete-output audits accepted; no anatomical/full-86B feasibility claim.')
    (HERE/'qa-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
