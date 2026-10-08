"""Retain only accepted capacity records; launch no simulation or remote work."""
from pathlib import Path
from evidence_paths import legacy_repo, external_directory
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
REPO=legacy_repo()
OUT=ROOT/'data/capacity'
OUT.mkdir(parents=True,exist_ok=True)
rows=[];sources=[];engines=[]
for size,directory,case in [(86,'brain-fraction-86m-20261007','major-owner30'),
                          (128,'brain-fraction-128m-20261007','owner30-128m-b600m')]:
    source=REPO/'brian2-rust/mpi-evidence'/directory
    summary=json.loads((source/'final-report.json').read_text())
    audit=json.loads((source/(case+'-audit.json')).read_text())
    qa=json.loads((source/'qa-report.json').read_text())
    resources=json.loads((source/(case+'-resources.json')).read_text())
    launch=json.loads((source/(case+'-launch-result.json')).read_text())
    assert all(j['passed'] for j in [summary,audit,qa,resources])
    assert launch['error'] is None and all(v==0 for v in launch['returncodes'].values())
    assert audit['neurons']==size*1_000_000 and audit['connections']==size*1_000_000_000
    assert audit['runtime']['ranks']==30 and len(set(audit['runtime']['processor_names']))==30
    assert audit['runtime']['rank_cpu_ids']==[1]*30
    assert {t['construction'] for t in audit['runtime']['procedural_topology']}=={'target-owner-local'}
    engines.append(summary.get('isolated_engine_source_sha256',summary.get('engine_archive_sha256')))
    for name in ['final-report.json',case+'-audit.json',case+'-resources.json',case+'-launch-result.json',case+'-deployment-audit.json','qa-report.json']:
        raw=(source/name).read_bytes();retained=f'{size}m-{name}'
        (OUT/retained).write_bytes(raw)
        sources.append({'source':str((source/name).resolve()),'retained':retained,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    rows.append({'neurons':audit['neurons'],'connections':audit['connections'],'hosts':30,'ranks':30,
                 'duration_ms':audit['duration_ms'],'dt_ms':.1,'precision':'reference-f64',
                 'launch_wall_seconds':launch['wall_seconds'],
                 'initialization_max_seconds':audit['stage_max_seconds'][0],
                 'simulation_max_seconds':audit['stage_max_seconds'][2],
                 'spike_exchange_max_seconds':audit['spike_exchange_seconds_minmax'][1],
                 'maximum_worker_cgroup_peak_gib':summary['maximum_worker_cgroup_peak_gib'],
                 'spikes':audit['spikes'],'delivered_synaptic_events':audit['delivered_synaptic_events'],
                 'populations':audit['populations'],'model_plan_sha256':audit['plan_sha256']})
old=REPO/'brian2-rust/mpi-evidence/brain-fraction-86m-20261007'
new=REPO/'brian2-rust/mpi-evidence/brain-fraction-128m-20261007'
for source in [old/'protocol-owner30.json',old/'source-b128-identity.json',old/'pilot-owner30-placement-proof.json',
               old/'pilot-owner30-validation.json',old/'pilot-owner30-resources.json',
               old/'owner30-core-array-estimate.json',new/'protocol.json',new/'protocol-v2.json',
               new/'engine-source-readback.json',new/'prepare-owner30-128m-status.json']:
    raw=source.read_bytes();retained=source.name
    (OUT/retained).write_bytes(raw)
    sources.append({'source':str(source.resolve()),'retained':retained,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
pilot=json.loads((old/'pilot-owner30-validation.json').read_text())
assert pilot['passed'] and pilot['checks']==303 and pilot['max_state_absolute_difference']==0
assert len(set(engines))==1 and engines[0] is not None
evidence={'schema':'atlas-preprint-connected-capacity-v1','rows':rows,'sources':sources,
          'pilot':{k:pilot[k] for k in ['passed','checks','max_state_absolute_difference','spikes','synaptic_events']},
          'reference_neuron_count':86_000_000_000,'same_engine_snapshot':True,
          'engine_archive_sha256':json.loads((old/'source-b128-identity.json').read_text())['archive_sha256'],
          'scope':'Accepted single observations; complete-output engineering audit. No anatomical human-brain, maximum-capacity, strong/weak scaling, NEST speed or full-scale feasibility claim.'}
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps({'passed':True,'rows':len(rows),'retained_sources':len(sources)}))
