"""Audit downloaded, unmodified node artifacts, full dumps and cgroup evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from brian2_rust.results import load_results


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def peak(path):
    text=path.read_text()
    assert 'Exit status: 0' in text
    return next(int(line.rsplit(':',1)[1])*1024 for line in text.splitlines() if 'Maximum resident set size' in line)


def audit(a,b,launches):
    completed=json.loads((launches/'completed.json').read_text());assert len(completed)==8
    report={'schema':'b2-mpi-procedural-audit-v1','complete':False,'runs':[],'guards':[],'artifact_hashes':{}}
    for edges in [262144,4194304]:
        folder=a/f'models/edges-{edges}';model=json.loads((folder/'model.json').read_text())
        assert sum(s['topology']['edge_count'] for s in model['instance']['synapses'])==edges
        reference=a/f'runs/edges-{edges}-reference'
        hashes={name:digest(reference/name) for name in ['results.bin','events.bin']}
        for ranks in [1,2,4]:
            label=f'edges-{edges}-rank-{ranks}';out=a/'runs'/label
            assert hashes=={name:digest(out/name) for name in hashes}
            summary=json.loads((out/'summary.json').read_text())
            runtime=json.loads((out/'mpi-runtime.json').read_text())
            assert runtime['ranks']==ranks and summary['neuron_count']==1000
            expected_hosts=['hk-prod-model-ae02-23'] if ranks==1 else [n for n in ['hk-prod-model-ae02-23','hk-prod-model-ae02-24'] for _ in range(ranks//2)]
            assert runtime['processor_names']==expected_hosts
            assert runtime['rank_cpu_ids']==([0] if ranks==1 else [i for _ in range(2) for i in range(ranks//2)])
            topology=runtime['procedural_topology'];assert len(topology)==4
            rank_edges=[0]*ranks;rank_build=[0.0]*ranks
            for q,row in enumerate(topology):
                assert row['projection']==q and row['global_edges']==edges//4
                stats=np.array(row['rank_stats']).reshape(ranks,2)
                assert stats[:,0].sum()==stats[:,1].sum()==edges//4
                assert stats[:,1].max()-stats[:,1].min()<=1
                rank_edges=[n+int(v) for n,v in zip(rank_edges,stats[:,0],strict=True)]
                rank_build=[n+v for n,v in zip(rank_build,row['build_seconds'],strict=True)]
            project=folder/f'rank-{ranks}';manifest=json.loads((project/'manifest.json').read_text())
            assert manifest['plan_sha256']==runtime['plan_sha256']
            for name,value in manifest['files'].items():
                assert digest(project/name)==value
                assert digest(b/'models'/folder.name/f'rank-{ranks}'/name)==value
            executable=digest(project/'b2-mpi')
            assert executable==json.loads((project/'build.json').read_text())['executable_sha256']
            assert executable==digest(b/'models'/folder.name/f'rank-{ranks}/b2-mpi')
            report['artifact_hashes'][f'{edges}/{ranks}']=executable
            states=load_results(model,out)['populations']
            assert all(np.isfinite(values).all() for pop in states for values in pop['states'].values())
            assert sum(len(pop['indices']) for pop in states)>0
            rss=[peak((a if ranks==1 or rank<ranks//2 else b)/'metrics'/f'{label}-rank{rank}.time') for rank in range(ranks)]
            report['runs'].append({'label':label,'global_edges':edges,'ranks':ranks,'rank_local_edges':rank_edges,
                                  'rank_build_seconds':rank_build,'rank_peak_rss_bytes':rss,'summary':summary,'hashes':hashes})
    for node in [a,b]:
        for path in sorted((node/'guards').glob('*.json')):
            guard=json.loads(path.read_text());assert guard['uid']==1000 and guard['admitted']
            before,after=guard['before'],guard['after']
            assert before['memory.swap.max']=='0' and after['memory.swap.max']=='0'
            assert before['pids.max']=='64'
            assert int(after['memory.peak'])<=int(before['memory.max'])
            if path.name=='memory-probe.json':
                assert guard['returncode']==-9 and 'oom_kill 1' in after['memory.events']
            else:
                assert guard['returncode']==0 and 'oom_kill 0' in after['memory.events']
                assert int(before['memory.max'])==2048*2**20 and before['cpu.max']=='200000 100000'
                if path.name!='build.json':
                    assert guard['host_memory_bytes']['MemAvailable']>=int(before['memory.max'])+guard['reserved_host_memory_bytes']
            report['guards'].append({'node':node.name,'name':path.name,**guard})
    assert len(report['guards'])==20
    report['complete']=True
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node23',type=Path,required=True);p.add_argument('--node24',type=Path,required=True)
    p.add_argument('--launches',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=audit(args.node23,args.node24,args.launches)
    args.output.write_text(json.dumps(r,indent=2)+'\n')
    for row in r['runs']:print(row['label'],'build seconds',row['rank_build_seconds'],'rank MiB',[x/2**20 for x in row['rank_peak_rss_bytes']])
    print('complete',r['complete'],'guards',len(r['guards']))
