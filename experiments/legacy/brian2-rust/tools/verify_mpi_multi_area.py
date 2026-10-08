"""Audit the bounded MAM adapter pilots, including their guarded builds."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from brian2_rust.distributed import _verify_artifact
from brian2_rust.results import load_results
from brian2_rust.protocol import verify_protocol, layer_hashes
import numpy as np


def verify(root,output,*,owner_local=False,batched=False,profile=None,opt_level=3,compile_label=None,project_label=None):
    profiles = {"n02-k04": (84835660, 158, 64000000, 8),
                "n02-k08": (169671396, 160, 128000000, 16),
                "four-n02-k08": (373706149, 418, 128000000, 32),
                "four-n1-k1": (2335664581, 427, 768000000, 128)}
    if profile is not None and profile not in profiles:
        raise ValueError("unknown pilot profile")
    large = profile is not None
    batched = batched or large
    owner_local = owner_local or batched
    full_subset = profile == 'four-n1-k1'
    four = profile in ('four-n02-k08', 'four-n1-k1')
    hosts = (['hk-prod-model-ae02-25','hk-prod-model-ae08-81','hk-prod-model-ae08-83','hk-prod-model-ae07-71']
             if four else ['hk-prod-model-ae02-23','hk-prod-model-ae02-24'])
    nodes = [root/('node'+host.rsplit('-',1)[1]) for host in hosts]
    ranks = len(nodes)
    a = nodes[0]
    reference_node = root/'node23'
    project='v1v2-scale01-batch' if batched else ('v1v2-scale01' if owner_local else 'v1v2-kernels')
    label='v1v2-scale01-batch' if batched else ('v1v2-scale01-owner-local' if owner_local else 'v1v2-kernels-8g')
    if large:
        project=label=profile if four else 'v1v2-'+profile
    if project_label is not None:
        project=label=project_label
    expected_neurons=663405 if full_subset else (132670 if four else (70997 if large else (35494 if owner_local else 3540)))
    expected_edges=profiles[profile][0] if large else (10604386 if owner_local else 105973)
    expected_projections=profiles[profile][1] if large else (150 if owner_local else 137)
    model=json.loads((a/project/'model.json').read_text())
    verify_protocol(model)
    assert all((a/project/'model.json').read_bytes()==(node/project/'model.json').read_bytes() for node in nodes)
    manifests=[];builds=[]
    for node in nodes:
        p=node/project/'mpi';manifests.append(_verify_artifact(p));build=json.loads((p/'build.json').read_text())
        assert hashlib.sha256((p/'b2-mpi').read_bytes()).hexdigest()==build['executable_sha256']
        assert build['opt_level']==opt_level
        builds.append(build)
    assert all(item==manifests[0] for item in manifests) and all(item==builds[0] for item in builds)
    run=a/'runs'/label
    ref=reference_node/('runs/reference-'+profile if large else ('runs/reference-scale01' if owner_local else 'runs/reference-v1v2'))
    hashes={}
    for name in ['results.bin','events.bin']:
        data=(run/name).read_bytes();assert data==(ref/name).read_bytes(),name+' differs'
        hashes[name]=hashlib.sha256(data).hexdigest()
    loaded=load_results(model,run)
    for pop in loaded['populations']:
        assert all(np.isfinite(v).all() for v in pop['states'].values())
    summary=json.loads((run/'summary.json').read_text());runtime=json.loads((run/'mpi-runtime.json').read_text())
    assert summary['neuron_count']==expected_neurons and summary['spike_count']>0 and summary['synaptic_events']>0
    assert summary['final_time_seconds']==(0.1 if owner_local else 0.01)
    assert runtime['processor_names']==hosts
    assert runtime['rank_cpu_ids']==[0]*ranks and runtime['ranks']==ranks
    assert runtime['plan_sha256']==manifests[0]['plan_sha256']
    assert runtime['exchange_calls_per_rank']==(1000 if batched else (16000 if owner_local else 1600))
    if batched:
        assert runtime['spike_exchange_strategy']=='consecutive-producers-same-clock'
    plan=json.loads((a/project/'mpi/execution-plan.json').read_text())
    for layer, digest in layer_hashes(model).items():
        assert plan[layer+'_sha256']==digest
    if large:
        assert all(type(value) is int and value >= 0 for value in runtime['rank_queue_capacity_bytes'])
        assert len(runtime['rank_queue_capacity_bytes'])==ranks
        launch=json.loads((root/'launch.json').read_text())
        assert launch['error'] is None and all(code==0 for code in launch['returncodes'].values())
        controller=next(j for j in (json.loads(path.read_text()) for path in (a/'guards').glob('*.json'))
                        if 'b2-mpi' in ' '.join(j['command']) and 'mpiexec.hydra' in ' '.join(j['command']))
        assert 'B2_MPI_MAX_LOCAL_EDGES='+str(profiles[profile][2]) in ' '.join(controller['command'])
    local=[0]*ranks;offsets=[0]*ranks
    assert len(runtime['procedural_topology'])==expected_projections
    for item,syn,inst in zip(runtime['procedural_topology'],model['definition']['synapses'],model['instance']['synapses'],strict=True):
        owner=plan['population_owners'][syn['target_population']];count=inst['topology']['edge_count']
        stats=np.array(item['rank_stats']).reshape(ranks,2)
        others=[rank for rank in range(ranks) if rank!=owner]
        assert stats[:,0].sum()==stats[:,1].sum()==count and stats[owner,0]==count and np.all(stats[others,0]==0)
        if owner_local:
            assert item['construction']=='target-owner-local'
            assert stats[owner,1]==count and np.all(stats[others,1]==0)
        assert all(item['rank_csr_offset_bytes'][rank]==0 for rank in others)
        assert item['rank_csr_offset_bytes'][owner]==(syn['source_count']+1)*8
        local[owner]+=count;offsets[owner]+=item['rank_csr_offset_bytes'][owner]
    assert sum(local)==expected_edges
    resources=[];failures=[]
    for node in nodes+([reference_node] if four else []):
        for path in sorted((node/'guards').glob('*.json')):
            j=json.loads(path.read_text());assert j['admitted'] and j['uid']==1000 and 'after' in j
            assert j['after']['memory.swap.max']=='0'
            events=dict(line.split() for line in j['after']['memory.events'].splitlines())
            assert events['oom']==events['oom_kill']=='0'
            peak=int(j['after']['memory.peak']);cap=int(j['after']['memory.max']);assert peak<=cap
            quota,period=map(int,j['after']['cpu.max'].split())
            if j.get('error'):
                assert not owner_local
                assert path.stem in ['build-v1v2','build-v1v2-o1','build-v1v2-shared']
                assert 'timed out after 120 seconds' in j['error'] and 120<=j['wall_seconds']<126
                assert cap==2*2**30 and quota==period*2
                failures.append(path.stem)
            else:
                is_reference=large and path.stem=='reference-'+profile
                is_four_build=four and path.stem==(compile_label or 'build-'+profile)
                is_preparation=full_subset and path.stem=='prepare-'+profile
                expected_cap=(profiles[profile][3] if is_reference else (16 if is_four_build else (8 if is_preparation else (32 if full_subset else 8))))*2**30
                assert j['returncode']==0 and cap==expected_cap and quota==period*(8 if is_four_build else 4)
                assert j['host_memory_bytes']['MemAvailable'] >= cap+j['reserved_host_memory_bytes']
                assert j['data_free_bytes'] >= 100*2**30
                if four and node != reference_node:
                    assert j['root_volume_allowed'] and j['data_volume']=='/' and j['data_device']==j['root_device']
            resources.append({'node':node.name,'name':path.stem,'peak_bytes':peak,'cap_bytes':cap,'wall_seconds':j['wall_seconds'],'success':not bool(j.get('error'))})
    assert len(resources)==(8 if full_subset else (7 if four else (5 if owner_local else 8))) and len(failures)==(0 if owner_local else 3)
    rss=[]
    for rank,node in enumerate(nodes):
        log=(node/f'metrics/{label}-rank{rank}.time').read_text()
        assert re.search(r'Exit status: 0\s*$',log)
        rss.append(int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',log)[1])*1024)
    report={'schema':'b2-mam-adapter-audit-v1','complete':True,'scope':'Area-subset adapter pilot only; not NEST scientific validation or full 32-area MAM capacity proof',
            'hosts':hosts,
            'neurons':expected_neurons,'recurrent_edges':sum(local),'owned_edges':local,'csr_offset_bytes':offsets,
            'rank_peak_rss_bytes':rss,'spikes':summary['spike_count'],'delivered_edges':summary['synaptic_events'],
            'timings':summary['timings'],'exchange_calls_per_rank':runtime['exchange_calls_per_rank'],
            'rank_exchange_seconds':runtime['spike_exchange_seconds'],
            'rank_queue_capacity_bytes':runtime.get('rank_queue_capacity_bytes'), 'dump_sha256':hashes,'build':builds[0],'resources':resources}
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--owner-local',action='store_true',help='Audit the N=K=0.1 owner-local V1/V2 run')
    p.add_argument('--batched',action='store_true',help='Audit the same N=K=0.1 run with batched spike exchange')
    p.add_argument('--profile',choices=['n02-k04','n02-k08','four-n02-k08','four-n1-k1'],help='Audit a larger bounded scale profile')
    p.add_argument('--opt-level',type=int,choices=range(4),default=3,help='Expected compilation optimization level')
    p.add_argument('--compile-label',help='Exact guard label for an explicitly selected compiler experiment')
    p.add_argument('--project-label',help='Explicit project/run label for the same model under a new generated implementation')
    a=p.parse_args();verify(a.root,a.output,owner_local=a.owner_local,batched=a.batched,profile=a.profile,opt_level=a.opt_level,compile_label=a.compile_label,project_label=a.project_label)
