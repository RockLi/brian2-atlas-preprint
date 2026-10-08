"""Bounded read-only snapshot of the same admitted replicate1750 job."""
import concurrent.futures,json,datetime,hashlib
from pathlib import Path
from mam_collect_native_primary_raw import remote
from mam_confirmation_identity import NODES,HOME,BRICK
from mam_launch_confirmation_run import CASE,SOURCE,IDENTITY_SHA,read,write


def host_code(index,v):
    return f'''from pathlib import Path
import os,json,subprocess,signal,datetime
signal.alarm(30);assert os.uname().nodename=={NODES[index]!r}
r=Path({v['receipts']!r});receipts=[]
if r.exists():
 for p in sorted(r.glob('rank*.json')):
  assert p.stat().st_size<65536;v=json.loads(p.read_text());assert v['identity_sha256']=={IDENTITY_SHA!r}
  receipts.append(dict(name=p.name,rank=v['rank'],event=v['event']))
units=[]
listing=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5)
for line in listing.splitlines():
 name=line.split()[0]
 raw=subprocess.check_output(['systemctl','show',name,'-p','ExecStart','-p','ControlGroup','-p','ActiveState','-p','MemoryMax','-p','TasksMax','-p','CPUQuotaPerSecUSec','-p','AllowedCPUs'],text=True,timeout=5)
 properties=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
 if {SOURCE!r} not in properties['ExecStart']:continue
 del properties['ExecStart'];cg=Path('/sys/fs/cgroup')/properties['ControlGroup'].lstrip('/')
 counters={{}}
 for n in ['memory.current','memory.peak','memory.events','memory.swap.current','cpu.stat','pids.current']:
  p=cg/n
  if p.exists():assert p.stat().st_size<2**20;counters[n]=p.read_text()
 units.append(dict(name=name,properties=properties,counters=counters))
volume=Path('/data/brick2') if {index==0!r} else Path('/');s=os.statvfs(volume)
memory={{k:int(x.split()[0])*1024 for k,x in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k=='MemAvailable'}}
output=None
if {index==0!r}:
 o=Path({v['output']!r});output=dict(exists=o.exists(),files={{}},spool_bytes=0,spool_files=0)
 if o.exists():
  for p in o.iterdir():
   if p.is_file():output['files'][p.name]=p.stat().st_size
  spool=o/'spike-spool'
  if spool.exists():
   for p in spool.iterdir():
    assert p.is_file() and not p.is_symlink();output['spool_bytes']+=p.stat().st_size;output['spool_files']+=1
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),units=units,receipts=receipts,
 free_bytes=s.f_bavail*s.f_frsize,memory=memory,output=output)))'''


def run(evidence):
    case=evidence/CASE;admission=read(case/'admission.json');v=read(case/'identity.json',IDENTITY_SHA)
    assert admission['admitted'] and admission['identity_sha256']==IDENTITY_SHA
    result=dict(schema='b2-mam-confirmation-observation-v1',admission_sha256=hashlib.sha256((case/'admission.json').read_bytes()).hexdigest(),hosts=[],errors=[],terminal_audit=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(remote,n,host_code(i,v)) for i,n in enumerate(NODES)]
        for node,future in zip(NODES,futures,strict=True):
            try:result['hosts'].append(future.result())
            except Exception as error:result['errors'].append(dict(host=node,error=str(error),stderr=str(getattr(error,'stderr',''))[-16384:]))
    folder=case/'snapshots';folder.mkdir(exist_ok=True)
    p=folder/('snapshot-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json');write(p,result)
    print(json.dumps(dict(path=str(p),errors=result['errors'],hosts=[dict(host=h['host'],units=len(h['units']),started=sum(r['event']=='rank_wrapper_started' for r in h['receipts']),failed=sum('failed' in r['event'] for r in h['receipts']),free_gib=h['free_bytes']/2**30,
        peak_gib=[int(u['counters'].get('memory.peak','0'))/2**30 for u in h['units']],output=h['output']) for h in result['hosts']])),flush=True)


if __name__=='__main__':run(Path(__file__).resolve().parents[1]/'mpi-evidence')
