"""Global arithmetic and delayed publication using explicit synthetic controls."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_native_primary_output_audit as m


@pytest.fixture
def inputs():
    ranks=[]
    for rank in range(48):
        prefix=m.PRIOR_SPIKES//48+(rank<m.PRIOR_SPIKES%48)
        bins=[0]*2010;bins[0]=prefix;bins[-1]=1
        ranks.append(dict(rank=rank,rank_event_stream_verified=True,raw_byte_prefix_exact=True,
            construction_ledger_exact=True,duration_ms=100500,chunks=2010,prefix_duration_ms=10500,
            linux_cache_release=True,physical_50ms_bin_counts=bins,terminal_tick_events=1,spikes=prefix+2,
            event_bytes=8*(prefix+2),prefix_bytes=8*prefix,prefix_sha256='a'*64))
    counts=np.zeros(2010,dtype=np.int64);counts[0]=m.PRIOR_SPIKES;counts[-1]=96
    resource=dict(reported_spikes=m.PRIOR_SPIKES+96,reported_event_bytes=8*(m.PRIOR_SPIKES+96),
        terminal_resource_audit_passed=True,construction_ledger_passed=True,neurons=m.NEURONS,recurrent_edges=m.EDGES,projections=8344)
    return ranks,resource,counts,np.zeros(2010),np.zeros(2010,dtype=np.int64)


def test_full_rank_summary_keeps_physical_and_drain_counts_separate(inputs):
    r=m.summarize(*inputs)
    assert not r['passed'] and not r['audit_guard_passed'] and r['raw_output_audit_passed']
    assert r['terminal_tick_events']==48 and r['physical_50ms_bin_counts'][-1]==48
    assert r['chunks'][-1]['spikes']==96 and len(r['chunks'])==2010
    assert r['retained_prefix_spikes']==527018677 and len(r['prefix_checks'])==48
    assert not r['scientific_equivalence'] and not r['performance_cost_acceptance']


@pytest.mark.parametrize('fault',['missing-rank','duplicate-rank','missing-prefix','short-window','no-cache-release',
    'physical-count','negative-count','prefix-total','resource-total','chunk-total','missing-resource'])
def test_global_audit_cannot_pass_partial_or_inconsistent_evidence(inputs,fault):
    ranks,resource,counts,sim,rss=inputs
    if fault=='missing-rank':ranks.pop()
    elif fault=='duplicate-rank':ranks[-1]['rank']=46
    elif fault=='missing-prefix':ranks[-1]['raw_byte_prefix_exact']=False
    elif fault=='short-window':ranks[-1]['duration_ms']=10500
    elif fault=='no-cache-release':ranks[-1]['linux_cache_release']=False
    elif fault=='physical-count':ranks[-1]['physical_50ms_bin_counts'][-1]+=1
    elif fault=='negative-count':ranks[-1]['terminal_tick_events']=-1
    elif fault=='prefix-total':ranks[-1]['prefix_bytes']-=8
    elif fault=='resource-total':resource['reported_spikes']+=1
    elif fault=='chunk-total':counts[-1]+=1
    else:resource['terminal_resource_audit_passed']=False
    with pytest.raises(ValueError):m.summarize(ranks,resource,counts,sim,rss)


@pytest.fixture
def pending(inputs,tmp_path,monkeypatch):
    r=m.summarize(*inputs);r['audit_seconds']=9.
    raw=(json.dumps(r)+'\n').encode();(tmp_path/m.PENDING).write_bytes(raw)
    monkeypatch.setattr(m,'DESTINATION',str(tmp_path))
    facts={'memory.max':str(16*2**30),'memory.swap.max':'0','pids.max':'64',
        'cpuset.cpus.effective':'8-9','cpu.max':'200000 100000','memory.peak':str(2**20),
        'memory.events':'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0'}
    guard=dict(schema='b2-mpi-resource-guard-v1',host='hk-prod-model-ae02-23',uid=1000,admitted=True,returncode=0,
        data_volume='/data/brick2',data_device=2,root_device=1,minimum_free_bytes=1280*2**30,
        minimum_observed_free_bytes=1290*2**30,wall_seconds=10.,before=copy.deepcopy(facts),after=copy.deepcopy(facts),
        disk_check_interval_seconds=5,file_limit_bytes=64*2**20,
        command=['python3','/fixture/mam_native_primary_output_audit.py','audit','--root',str(tmp_path)])
    path=tmp_path/'guard.json';path.write_text(json.dumps(guard))
    return tmp_path,path,hashlib.sha256(raw).hexdigest(),guard


def test_publish_requires_terminal_guard_then_publishes_without_overwrite(pending):
    root,gp,digest,g=pending
    result=m.publish(root,gp,digest)
    r=json.loads((root/m.SUMMARY).read_text())
    assert result['passed'] and r['passed'] and r['audit_guard_passed']
    assert r['analysis_wall_seconds_consumed']==10. and r['shared_analysis_budget_seconds']==10800
    assert (root/m.PENDING).exists() and not (root/'summary.publishing.json').exists()
    with pytest.raises(FileExistsError):m.publish(root,gp,digest)
    assert hashlib.sha256((root/m.SUMMARY).read_bytes()).hexdigest()==result['summary_sha256']


@pytest.mark.parametrize('fault',['guard-live','guard-failed','memory-pressure','memory-cap','wrong-volume',
    'budget','wrong-command','disk-drop','pending-hash'])
def test_failed_audit_never_publishes_scientific_input(pending,fault):
    root,gp,digest,g=pending
    if fault=='guard-live':g.pop('returncode')
    elif fault=='guard-failed':g['returncode']=1
    elif fault=='memory-pressure':g['after']['memory.events']=g['after']['memory.events'].replace('max 0','max 1')
    elif fault=='memory-cap':g['after']['memory.max']=str(32*2**30)
    elif fault=='wrong-volume':g['data_volume']='/'
    elif fault=='budget':g['wall_seconds']=10801.
    elif fault=='wrong-command':g['command'][-1]=str(root/'different')
    elif fault=='disk-drop':g['minimum_observed_free_bytes']=1279*2**30
    else:digest='0'*64
    gp.write_text(json.dumps(g))
    with pytest.raises(ValueError):m.publish(root,gp,digest)
    assert not (root/m.SUMMARY).exists() and not (root/'summary.publishing.json').exists()


def test_exact_inventory_rejects_unlisted_files_and_directories(tmp_path):
    p=tmp_path/'runs/rank0.json';p.parent.mkdir();p.write_bytes(b'{}')
    catalog=dict(schema='mam-direct-collection-v1',files=[dict(path='runs/rank0.json',bytes=2,sha256=hashlib.sha256(b'{}').hexdigest())])
    m.verify_inventory(tmp_path,catalog)
    extra=tmp_path/'extra';extra.mkdir()
    with pytest.raises(ValueError,match='directory set'):m.verify_inventory(tmp_path,catalog)
    extra.rmdir();p.write_bytes(b'[]')
    with pytest.raises(ValueError,match='checksum'):m.verify_inventory(tmp_path,catalog)


@pytest.mark.parametrize('fault',[None,'receipt-hash','transfer-pressure','shared-budget'])
@pytest.mark.parametrize('reference_seed',[None,1730])
def test_collection_receipts_and_guards_are_linked_before_raw_reads(tmp_path,fault,reference_seed):
    from test_mam_native_primary_collection import host,progress
    spec=m.audit_spec(reference_seed);label=spec['label'];tag=spec['tag'];destination=spec['destination']
    hosts=[host(i) for i in range(6)];rows=[];total=0
    if reference_seed is not None:
        for h in hosts:
            h['source_files']={n.replace('/'+m.LABEL+'/', '/'+label+'/'):v for n,v in h['source_files'].items()}
    outer=dict(returncode=0,stop_reason=None,wall_seconds=4.,wall_limit_seconds=180,
               sampled_peak_rss_bytes=100*2**20,sampled_rss_limit_bytes=1536*2**20)
    if reference_seed is not None:
        outer['command']=['mam_collect_native_primary_resources.py','--reference-seed',str(reference_seed)]
    gp=tmp_path/'outer.json';gp.write_text(json.dumps(outer))
    for i,h in enumerate(hosts):
        name='node'+m.NODES[i].rsplit('-',1)[-1]
        p={n.replace('/'+m.LABEL+'/', '/'+label+'/'):v for n,v in progress(i).items()}
        catalog=m.make_catalog(i,h,p,run_label=label);size,digest=m.validate(catalog,m.MAX_FILE,m.MAX_HOST);total+=size
        (tmp_path/(name+'-catalog.json')).write_text(json.dumps(catalog))
        receipt=dict(schema='mam-direct-collection-v1',complete=True,catalog_sha256=digest,bytes=size,files=len(catalog['files']),
            peer=m.IPS[i],output=destination+'/'+name,max_file_bytes=m.MAX_FILE,max_total_bytes=m.MAX_HOST,
            reserve_bytes=1280*2**30,file_cache_release_supported=True,source_archive_created=False)
        if fault=='receipt-hash' and i==2:receipt['catalog_sha256']='f'*64
        (tmp_path/(name+'-receipt.json')).write_text(json.dumps(receipt))
        rows.append(dict(host=m.NODES[i],catalog_sha256=digest,bytes=size,files=len(catalog['files']),send_wall_seconds=10.,receive_wall_seconds=10.))
        for receiver,phase in [(True,'receive'),(False,'send')]:
            facts={'memory.max':str(4*2**30),'memory.swap.max':'0','pids.max':'64','cpuset.cpus.effective':'8-9',
                'cpu.max':'200000 100000','memory.peak':str(2**20),
                'memory.events':'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0'}
            g=dict(schema='b2-mpi-resource-guard-v1',host='hk-prod-model-ae02-23' if receiver else m.NODES[i],uid=1000,
                cgroup='/system.slice/b2mpi-'+tag+'-'+name+'-'+phase+'.service',admitted=True,returncode=0,wall_seconds=10.,
                before=copy.deepcopy(facts),after=copy.deepcopy(facts),data_volume='/data/brick2' if receiver else '/',
                data_device=2 if receiver else 1,root_device=1,minimum_free_bytes=(1280 if receiver else 128)*2**30,
                minimum_observed_free_bytes=2000*2**30,file_limit_bytes=3*2**30)
            if fault=='transfer-pressure' and i==5 and receiver:g['after']['memory.events']=g['after']['memory.events'].replace('max 0','max 1')
            (tmp_path/(tag+'-'+name+'-'+phase+'-guard.json')).write_text(json.dumps(g))
    report=dict(schema='b2-mam-native-primary-collection-v1',ready=True,complete=True,resource_report_sha256='a'*64,
        destination=destination,source_archive_created=False,automatic_retry=False,shared_collection_budget_seconds=7200,
        control_outer_guard_sha256=hashlib.sha256(gp.read_bytes()).hexdigest(),previous_collection_seconds=5.,
        bulk_collection_seconds=7200. if fault=='shared-budget' else 10.,nodes=rows,bytes=total)
    if reference_seed is not None:
        from mam_launch_native_full_reference import PROTOCOL_SHA
        report.update(label=label,seed=reference_seed,protocol_sha256=PROTOCOL_SHA,transfer_tag=tag,source_project=m.PROJECT)
    (tmp_path/'report.json').write_text(json.dumps(report))
    if fault is None:
        catalogs,result=m.collection_gate(tmp_path,hosts,'a'*64,gp,reference_seed=reference_seed)
        assert len(catalogs)==6 and result['bytes']==total
    else:
        with pytest.raises(ValueError):m.collection_gate(tmp_path,hosts,'a'*64,gp,reference_seed=reference_seed)


@pytest.fixture(params=[1730,1731])
def reference_inputs(inputs,request):
    seed=request.param;spec=m.audit_spec(seed)
    ranks,resource,counts,sim,rss=inputs
    for rank,r in enumerate(ranks):
        prefix=spec['prior_spikes']//48+(rank<spec['prior_spikes']%48)
        r.update(seed=seed,prefix_duration_ms=2500,prefix_bytes=8*prefix,spikes=prefix+2,event_bytes=8*(prefix+2))
        r['physical_50ms_bin_counts'][0]=prefix
    counts[0]=spec['prior_spikes']
    resource.update(label=spec['label'],reported_spikes=spec['prior_spikes']+96,
                    reported_event_bytes=8*(spec['prior_spikes']+96))
    return seed,inputs


def test_reference_summary_reports_its_actual_prefix_without_primary_claim(reference_inputs):
    seed,args=reference_inputs;result=m.summarize(*args,reference_seed=seed)
    assert result['seed']==seed and result['label']==m.audit_spec(seed)['label']
    assert result['all_first_2500ms_event_prefixes_exact']
    assert 'all_first_10500ms_event_prefixes_exact' not in result
    assert result['retained_prefix_spikes']==m.audit_spec(seed)['prior_spikes']
    assert not result['passed'] and not result['scientific_equivalence']
    with pytest.raises(ValueError):m.summarize(*args)
    args[0][-1]['seed']=1729
    with pytest.raises(ValueError,match='seed identity'):m.summarize(*args,reference_seed=seed)


@pytest.fixture
def reference_pending(pending,reference_inputs,monkeypatch):
    root,gp,_,g=pending;seed,args=reference_inputs
    spec=m.audit_spec
    def fixture_spec(value):
        return {**spec(value),'destination':str(root)}
    monkeypatch.setattr(m,'audit_spec',fixture_spec)
    r=m.summarize(*args,reference_seed=seed);r['audit_seconds']=9.
    raw=(json.dumps(r)+'\n').encode();(root/m.PENDING).write_bytes(raw)
    g['command']+=['--reference-seed',str(seed)];gp.write_text(json.dumps(g))
    return root,gp,hashlib.sha256(raw).hexdigest(),g,seed


def test_reference_publication_requires_the_explicit_seed_and_guard(reference_pending):
    root,gp,digest,g,seed=reference_pending
    with pytest.raises(ValueError):m.publish(root,gp,digest)
    result=m.publish(root,gp,digest,reference_seed=seed)
    r=json.loads((root/m.SUMMARY).read_text())
    assert result['passed'] and r['seed']==seed and r['all_first_2500ms_event_prefixes_exact']
    assert not r['scientific_equivalence']


@pytest.mark.parametrize('fault',['missing-seed','wrong-seed','failed-guard'])
def test_reference_guard_mismatch_never_publishes(reference_pending,fault):
    root,gp,digest,g,seed=reference_pending
    if fault=='missing-seed':g['command']=g['command'][:-2]
    elif fault=='wrong-seed':g['command'][-1]='1729'
    else:g['returncode']=1
    gp.write_text(json.dumps(g))
    with pytest.raises(ValueError):m.publish(root,gp,digest,reference_seed=seed)
    assert not (root/m.SUMMARY).exists()
