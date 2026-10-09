"""Read complete outputs and check finite state, counts, rank coverage and hashes."""
from pathlib import Path
import argparse, hashlib, json, resource, sys, time

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--case',default='major');a=p.parse_args()
    started=time.monotonic();b=a.base;c=b/a.case
    sys.path.insert(0,str(b/'source/python'))
    import numpy as np
    from brian2_rust.results import load_results
    prepared=json.loads((c/'prepared.json').read_text())
    with (c/'model.json').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==prepared['files']['model.json']['sha256']
    model=json.loads((c/'model.json').read_text())
    result=load_results(model,c/'result',include_times=False,release_file_cache=True)
    runtime=json.loads((c/'result/mpi-runtime.json').read_text())
    ranks=prepared['ranks']
    work=np.asarray(runtime['rank_work'],dtype=np.int64).reshape(ranks,3)
    assert runtime['ranks']==ranks and runtime['plan_sha256']==prepared['plan_sha256']
    assert runtime['numeric_profile']=='reference-f64'
    assert int(work[:,0].sum())==prepared['neurons']
    assert int(work[:,1].sum())==result['metadata']['spike_count']>0
    assert int(work[:,2].sum())==result['synaptic_events']>0
    assert runtime['exchange_calls_per_rank']==1000
    assert len(set(runtime['processor_names']))==30
    topology=runtime['procedural_topology']
    assert len(topology)==prepared['projections']
    assert sum(t['global_edges'] for t in topology)==prepared['connections']
    for t in topology:
        expected='target-owner-local' if prepared.get('population_owners') else 'distributed-draw-ranges'
        assert t['construction']==expected
        assert len(t['rank_stats'])==ranks*2 and sum(t['rank_stats'][::2])==t['global_edges']
        assert sum(t['rank_stats'][1::2])==t['global_edges']
        if expected=='target-owner-local':
            syn=model['definition']['synapses'][t['projection']]
            owner=prepared['population_owners'][syn['target_population']]
            assert t['rank_stats'][owner*2]==t['global_edges'] and t['rank_stats'][owner*2+1]==t['global_edges']
            assert all(value==0 for rank,value in enumerate(t['rank_stats'][::2]) if rank!=owner)
            assert all(value==0 for rank,value in enumerate(t['rank_stats'][1::2]) if rank!=owner)
    populations=[]
    for d,x in zip(model['definition']['populations'],result['populations'],strict=True):
        populations.append({'name':d['name'],'neurons':d['count'],'spikes':len(x['spike_ticks']),
            'final_state':{k:{'min':float(np.min(v)),'max':float(np.max(v)),'mean':float(np.mean(v))} for k,v in x['states'].items()},
            'sampled_trace_finite':all(bool(np.isfinite(v).all()) for v in x['trace'].values()),
            'spikes_per_1ms_bin':np.bincount(np.asarray(x['spike_ticks'])//10,minlength=100).tolist(),
            'sampled_v_every_1ms':np.asarray(x['trace']['v'])[::10,0].tolist()})
    assert all(r['sampled_trace_finite'] for r in populations)
    stages=np.asarray(runtime['rank_stage_seconds']).reshape(ranks,5)
    assert np.isfinite(stages).all() and np.all(stages>=0)
    files={}
    for f in (c/'result').iterdir():
        if f.is_file():
            with f.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
            files[f.name]={'bytes':f.stat().st_size,'sha256':h}
    report={'schema':'atlas-fraction-complete-output-audit-v1','passed':True,'case':a.case,'neurons':prepared['neurons'],
        'connections':prepared['connections'],'duration_ms':prepared['duration_ms'],'mean_indegree':prepared['mean_indegree'],
        'neuron_count_percent_of_86B':prepared['neuron_count_fraction_of_86B_percent'],
        'spikes':result['metadata']['spike_count'],'delivered_synaptic_events':result['synaptic_events'],
        'metadata':result['metadata'],'layers':prepared['layers'],'plan_sha256':prepared['plan_sha256'],
        'populations':populations,'runtime':runtime,'stage_columns':runtime['rank_stage_columns'],
        'stage_max_seconds':stages.max(axis=0).tolist(),'stage_min_seconds':stages.min(axis=0).tolist(),
        'spike_exchange_seconds_minmax':[min(runtime['spike_exchange_seconds']),max(runtime['spike_exchange_seconds'])],
        'output_files':files,'audit_seconds':time.monotonic()-started,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'scientific_scope':prepared['scientific_scope'],'scientific_human_brain_validation':False,'full_scale_86B_feasibility_validated':False}
    (c/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','neurons','connections','spikes','delivered_synaptic_events','stage_max_seconds','audit_seconds']}),flush=True)

if __name__=='__main__':main()
