"""Retain a fully accepted five-point weak-scaling study; no remote work."""
from pathlib import Path
from evidence_paths import legacy_repo, external_directory
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
REPO=legacy_repo()
SOURCE=REPO/'brian2-rust/mpi-evidence/brain-count-scaling-1pct-20261007'
OUT=ROOT/'data/capacity'

def main():
    p=json.loads((SOURCE/'protocol.json').read_text())
    state=json.loads((SOURCE/'sweep-state.json').read_text())
    assert state['status']=='endpoint_complete' and state['accepted_hosts']==[3,6,12,24,30]
    qa=json.loads((SOURCE/'qa-report.json').read_text())
    assert qa['passed'] and qa['endpoint_complete']
    retained=[];rows=[];pilots=[]
    def retain(name):
        original=(SOURCE/name).read_bytes();raw=original;target='weak-'+name
        compacted=False
        if name.endswith(('-resources.json','-audit.json')):
            data=json.loads(original)
            def compact(value):
                nonlocal compacted
                if isinstance(value,dict):
                    topology=value.pop('procedural_topology',None)
                    if topology is not None:
                        assert isinstance(topology,list)
                        value['procedural_topology_summary']=dict(projections=len(topology),construction=sorted({t['construction'] for t in topology}))
                        compacted=True
                    for child in value.values():compact(child)
                elif isinstance(value,list):
                    for child in value:compact(child)
            compact(data)
            if compacted:
                data['retained_summary_source_sha256']=hashlib.sha256(original).hexdigest()
                data['retained_summary_scope']='Numerical metrics retained; duplicate full topology replaced by projection count/construction summary. Complete source records remain hash-bound in the experiment archive.'
                raw=(json.dumps(data,indent=2)+'\n').encode()
        (OUT/target).write_bytes(raw)
        retained.append(dict(source=str((SOURCE/name).resolve()),retained=target,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),source_bytes=len(original),source_sha256=hashlib.sha256(original).hexdigest(),summary_compacted=compacted))
    for h in [3,6,12,24,30]:
        case=state.get('case_map',{}).get(str(h),f'weak1pct-h{h:02d}-v1')
        row=json.loads((SOURCE/(case+'-accepted.json')).read_text())
        pilot=row.get('numerical_pilot_case',f'pilot-weak1pct-h{h:02d}-v1')
        audit=json.loads((SOURCE/(case+'-audit.json')).read_text())
        validation=json.loads((SOURCE/(pilot+'-validation.json')).read_text())
        assert row['passed'] and row['cleanup_passed'] and audit['passed'] and validation['passed']
        assert row['hosts']==h and row['ranks']==h*8 and row['physical_worker_cores']==h*8
        assert row['neurons']==h//3*86_000_000 and row['connections']==row['neurons']*1000
        assert row['duration_ms']==100 and row['precision']=='reference-f64'
        assert validation['checks']==h*80+3 and validation['max_state_absolute_difference']==0
        assert row['full_audit_file_sha256']==audit['full_audit_file_sha256']
        for key in ['source_archive_sha256','prepare_script_sha256','launcher_sha256']:
            assert row[key]==p[key]
        row['populations']=audit['populations']
        row['excitatory_populations']=19*(h//3)
        row.setdefault('host_memory_reserve_gib',64)
        rows.append(row);pilots.append(validation)
        for name in [case+'-accepted.json',case+'-audit.json',case+'-resources.json',case+'-launch-result.json',case+'-deployment-audit.json',case+'-post-inventory.json',pilot+'-validation.json',pilot+'-resources.json','prepare-'+case+'-status.json','audit-'+case+'-status.json']:
            retain(name)
    for name in ['protocol.json','source-identity.json','engine-source-readback.json','launcher-tests.json','build-scale1pct-v1-status.json','budget-scale1pct-v1-status.json','affinity-validation-correction.json','audit-script-stage.json','sweep-state.json','qa-report.json','accepted-weak-scaling.json','protocol-revision-v2.json','weak1pct-h24-v2-input-reuse.json','observer-compaction-identity.json']:
        retain(name)
    if rows[-1].get('protocol_revision')=='v3':
        for name in ['protocol-revision-v3.json','policy48-stage.json','policy48-tests.xml','policy48-kernel-smoke.json','policy48-controller-identity.json','guard-host-reserve48-v3.py','control_policy48.py','overlay_inputs.py','resume_policy48.py',rows[-1]['case']+'-input-reuse.json',rows[-1]['numerical_pilot_case']+'-input-reuse.json']:
            retain(name)
    evidence=dict(schema='atlas-preprint-weak-scaling-v1',rows=rows,pilots=pilots,sources=retained,
        source_archive_sha256=p['source_archive_sha256'],reference_neuron_count=86_000_000_000,
        endpoint_neurons=860_000_000,scope='Five accepted synthetic E/I weak-scaling capacity observations. One timing observation per size. All complete outputs audited; all per-layout small-model numerical gates passed. Count normalization only, not anatomical human-brain reconstruction or full-86B feasibility. Strong scaling remains separate.')
    (OUT/'weak-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    old=SOURCE.parent/'brain-fraction-256m-20261007'
    old_case='capacity-256m-owner60'
    old_report=json.loads((old/'final-report.json').read_text())
    old_audit=json.loads((old/(old_case+'-audit.json')).read_text())
    old_pilot=json.loads((old/'pilot-owner60-n256-validation.json').read_text())
    old_qa=json.loads((old/'qa-report.json').read_text())
    assert all(r['passed'] for r in [old_report,old_audit,old_pilot,old_qa])
    assert old_report['neurons']==256_000_000 and old_report['ranks']==60 and old_report['hosts']==30
    assert old_pilot['checks']==603 and old_pilot['max_state_absolute_difference']==0
    previous=[]
    for name in ['final-report.json','qa-report.json','protocol.json','source-identity.json','engine-source-readback.json',old_case+'-audit.json',old_case+'-resources.json',old_case+'-launch-result.json','pilot-owner60-n256-validation.json']:
        raw=(old/name).read_bytes();target='256m-'+name
        (OUT/target).write_bytes(raw)
        previous.append(dict(source=str((old/name).resolve()),retained=target,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    (OUT/'historical-256m-evidence.json').write_text(json.dumps(dict(schema='atlas-preprint-historical-256m-v1',row=old_report,sources=previous,scope='Single accepted fixed-30-host capacity observation with 60 physical worker cores and a separate engine snapshot. Not part of the five-point weak-scaling curve.'),indent=2)+'\n')
    print(json.dumps({'passed':True,'rows':len(rows),'retained_sources':len(retained)}))

if __name__=='__main__':main()
