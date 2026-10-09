"""Verify all worker file readbacks against the leader archive."""
from pathlib import Path
import argparse,hashlib,json,tarfile
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--ranks-per-node',type=int,default=8);a=p.parse_args();b=Path(__file__).resolve().parent
rpn=a.ranks_per_node;total=30*rpn
with tarfile.open(b/(a.case+'-mpi.tar.gz')) as t:
 expected={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers()}
rows=json.loads((b/(a.case+'-deployment.json')).read_text());assert len(rows)==29
seen=set()
for i,r in enumerate(rows,1):
 assert r['owned_ranks']==list(range(i*rpn,(i+1)*rpn))
 for name,h in r['files'].items():assert expected[name]==h
 actual={int(n.split('-')[1].split('.')[0]) for n in r['files'] if n.startswith('instance.rank-')}
 assert actual==set(r['owned_ranks']);seen|=actual
assert seen==set(range(rpn,total))
assert {int(n.split('-')[1].split('.')[0]) for n in expected if n.startswith('instance.rank-')}==set(range(total))
result={'passed':True,'case':a.case,'workers':29,'rank_shards':total,'worker_rank_shards':len(seen),'files_checked':sum(len(r['files']) for r in rows),'leader_archive_sha256':hashlib.sha256((b/(a.case+'-mpi.tar.gz')).read_bytes()).hexdigest()}
(b/(a.case+'-deployment-audit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
