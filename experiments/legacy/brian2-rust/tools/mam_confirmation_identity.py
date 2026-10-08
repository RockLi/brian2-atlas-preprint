"""Pinned artifact identity and rank-owned deployment for completed replicates.

No simulation, scientific acceptance, or relaxation of legacy1729 gates.
"""
import hashlib
import json
from pathlib import Path
from mam_replicate_seeds import allocation,registry

CASE='confirmation-artifact-v1-seed1750'
COMPLETION='b8c07d5039ce8e7e815d82dd865bc526079afcc4998210cfce70b125226ef050'
ARTIFACT_COMPLETIONS={1750:COMPLETION,1751:'03ea219c55db3d3c686358ac501f7e5587fa9f75edcc91563bbf8910c2308dd9'}
CONTRACT='1768e58fa402f26de3fd927aa65cb0349a68734a78366360e993bc3b77ba23a6'
NODES=['hk-prod-model-ae02-23','hk-prod-model-ae08-81','hk-prod-model-ae08-83','hk-prod-model-ae07-71']
IPS=['192.168.20.23','192.168.30.81','192.168.30.83','192.168.30.71']
HOME='/atlas-home/0003/workspace/brian2-mpi-primary-20260909'
BRICK='/data/brick2/brian2-mpi-region-20260907/primary-host-v1'
BUILD='/data/brick2/brian2-mpi-region-20260907'
SOURCE=BUILD+'/mam-confirmation-artifact-v1-seed1750/artifact'
PROJECT=HOME+'/confirmation-v1/seed1750'
LABEL='rust-mam-confirmation-v1-replicate1750-100500ms'
SHARED=('main.rs','mpi_bridge.c','instance.bin','execution-plan.json','instance-identities.rs','manifest.json','build.json','b2-mpi')


def need(ok,msg):
    if not ok:raise ValueError(msg)


def read(path,digest=None):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<2**20,'bounded control file required')
    raw=path.read_bytes()
    if digest is not None:need(hashlib.sha256(raw).hexdigest()==digest,'control digest changed: '+str(path))
    return json.loads(raw)


def identity(evidence,replicate=1750):
    need(type(replicate) is int and replicate in ARTIFACT_COMPLETIONS,'artifact completion is not pinned')
    completion=ARTIFACT_COMPLETIONS[replicate]
    source=BUILD+f'/mam-confirmation-artifact-v1-seed{replicate}/artifact'
    project=HOME+f'/confirmation-v1/seed{replicate}'
    label=f'rust-mam-confirmation-v1-replicate{replicate}-100500ms'
    p=evidence/f'confirmation-artifact-v1-seed{replicate}';c=read(p/'complete.json',completion)
    expected={'intent.json','admission.json','command.json','stage.json','build.log','guard.json',
        'prepared.json','random-input-audit.json','pending.json','terminal-state.json'}
    if replicate==1750:expected.add('snapshot-1.json')
    need(set(c['evidence_sha256'])==expected,'artifact evidence coverage differs')
    for n,h in c['evidence_sha256'].items():
        raw=(p/n).read_bytes();need(len(raw)<2**20 and hashlib.sha256(raw).hexdigest()==h,'artifact evidence changed')
    need(c['complete'] is True and c['artifact_prepared'] is True and c['replicate']==replicate
         and c['neural_runs']==0 and c['launch_admitted'] is False,'artifact stage identity')
    original=read(evidence/'confirmation-random-input-contract-v1/report.json',CONTRACT)
    need({k:original[k] for k in registry()}==registry(),'random allocation contract differs')
    r=read(p/'random-input-audit.json');need(r['keys']==allocation(replicate) and r['registry']==registry(),'actual random keys differ')
    need(r['legacy_voltage_reproduced_exactly'] is True and r['delta']==dict(only_declared_random_inputs_changed=True,
        definition_exact=True,run_exact=True,initial_voltage_neurons=4129924,projection_keys=8344,
        runtime_key=allocation(replicate)['runtime_input']),'random input audit incomplete')
    w=read(p/'pending.json');need(w==c['worker'] and w['artifact']==source and w['neural_runs']==0
        and w['confirmation_outcomes_observed']==0 and not w['launch_admitted'],'worker identity')
    prepared=read(p/'prepared.json',w['prepared_sha256']);read(p/'random-input-audit.json',w['random_input_audit_sha256'])
    need(prepared['definition_and_run_exact'] and prepared['plan_only_instance_identity_changed']
        and prepared['all_32_shards_changed'],'full model/plan identity check incomplete')
    from mam_benchmark_terminal import guard
    from mam_prepare_confirmation_artifact import LIMIT
    tag=f'mam-confirmation-artifact-v1-seed{replicate}'
    g=read(p/'guard.json');command=read(p/'command.json')['command']
    need(g['command']==command[command.index('--')+1:],'artifact guard command differs')
    spec=dict(host=NODES[0],role='build',memory_bytes=16*2**30,pids_max=64,cpu_ids=[8,9,10,11],cpu_quota_cores=4,
        volume='/data/brick2',allow_root_volume=False,file_limit_bytes=1024*2**20,
        minimum_free_bytes=1280*2**30,reserved_host_memory_bytes=64*2**30)
    need(guard(g,spec,'b2mpi-'+tag,LIMIT)==c['guard_accounting'],'artifact guard decision differs')
    need(read(p/'terminal-state.json')==dict(returncode=0,active_services=''),'build still live or failed')
    files=w['files']
    need(set(files)=={'model.json','parameters.json','placement.json','mpi/mpi_bridge.o'}|
        {'mpi/'+n for n in SHARED}|{'mpi/instance.rank-'+str(i)+'.bin' for i in range(32)},'full artifact inventory')
    return dict(schema='b2-mam-confirmation-identity-v1',replicate=replicate,label=label,
        artifact_completion_sha256=completion,random_contract_sha256=CONTRACT,
        model_sha256=prepared['model_sha256'],instance_sha256=prepared['layers']['instance'],
        definition_sha256=prepared['layers']['definition'],run_sha256=prepared['layers']['run'],
        plan_sha256=prepared['plan_sha256'],executable_sha256=w['build']['executable_sha256'],
        random_keys=r['keys'],files=files,source_artifact=source,project=project,
        output=BRICK+'/runs/'+label,receipts=HOME+'/confirmation-receipts/'+label,
        metrics=HOME+'/confirmation-metrics/'+label,analysis=BUILD+'/confirmation-analysis/'+label,
        raw_audit=BUILD+'/confirmation-raw-audit/'+label,
        neural_launch_admitted=False,scientific_acceptance=False)


def deployment_catalog(value,index):
    from mam_collection_transfer import SCHEMA,validate
    need(type(index) is int and 0<=index<4,'deployment host index')
    wanted={'mpi/'+n for n in SHARED}|{'mpi/instance.rank-'+str(i)+'.bin' for i in range(index*8,(index+1)*8)}
    catalog=dict(schema=SCHEMA,files=[dict(path=n,**value['files'][n]) for n in sorted(wanted)])
    total,_=validate(catalog,64*2**20,128*2**20)
    need(total>0,'empty deployment')
    return catalog
