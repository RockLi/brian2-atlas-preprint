"""Bounded read-only resource snapshot for the neuron-count capacity experiment."""
from pathlib import Path
import concurrent.futures, datetime, json, shlex, subprocess, time

HERE = Path(__file__).resolve().parent
nodes = [n['spec']['hostname'] for n in json.loads((HERE/'teleport-nodes.json').read_text())
         if n['spec']['hostname'].startswith('hk-prod-model-')]
CODE = r'''
import datetime,json,os,subprocess,time,resource,signal
from pathlib import Path
signal.alarm(25)
resource.setrlimit(resource.RLIMIT_AS,(128*2**20,128*2**20))
resource.setrlimit(resource.RLIMIT_CPU,(5,5))
def cmd(a):
 try:return subprocess.check_output(a,text=True,timeout=5).strip()
 except Exception as e:return type(e).__name__+': '+str(e)[:150]
mem={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemTotal','MemAvailable','SwapTotal','SwapFree']}
def ticks():return {int(v[0][3:]):list(map(int,v[1:9])) for l in Path('/proc/stat').read_text().splitlines() if (v:=l.split())[0].startswith('cpu') and v[0][3:].isdigit()}
a=ticks();time.sleep(1);b=ticks();busy={}
for c in a:
 d=[y-x for x,y in zip(a[c],b[c])];busy[c]=round(100*(1-(d[3]+d[4])/sum(d)),1) if sum(d) else None
volumes={}
for name in ['/home/rock','/data/brick2']:
 p=Path(name)
 if p.exists():
  v=os.statvfs(p);volumes[name]={'free_bytes':v.f_bavail*v.f_frsize,'root_device':p.stat().st_dev==Path('/').stat().st_dev,'mount':p.is_mount()}
prefix=Path('/atlas-home/0003/workspace/brian2-mpi-cluster-20260907')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':os.uname().nodename,'memory':mem,'cpus':os.cpu_count(),'load':os.getloadavg(),'cpu_busy_percent':busy,'volumes':volumes,'interfaces':cmd(['ip','-j','-4','addr']),'b2_services':cmd(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*']),'model_processes':cmd(['pgrep','-l','-f','b2-mpi|mpiexec.hydra|hydra_pmi_proxy|nest.*python|cargo build']),'runtime_files':list(str(p) for p in prefix.glob('*/bin/mpicc')),'base_exists':prefix.exists(),'rustc':cmd(['/atlas-home/0003/.cargo/bin/rustc','--version']),'python':cmd(['python3','--version'])}))
'''

def check(node):
    started=time.monotonic()
    try:
        p=subprocess.run(['tsh','ssh','rock@'+node,shlex.join(['python3','-c',CODE])],capture_output=True,text=True,timeout=45)
        if p.returncode:raise RuntimeError(p.stderr[-1000:])
        r=json.loads(p.stdout);assert r['host']==node
        return {'node':node,'ok':True,'facts':r,'seconds':time.monotonic()-started}
    except Exception as e:return {'node':node,'ok':False,'error':str(e),'seconds':time.monotonic()-started}

if __name__=='__main__':
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check,nodes))
    report={'schema':'atlas-brain-fraction-live-inventory-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'reservation':False,'rows':rows,'wall_seconds':time.monotonic()-started}
    (HERE/'live-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in rows:
        if not r['ok']:print(r['node'],'FAILED',r['error'][:120]);continue
        f=r['facts'];print(r['node'], 'RAM GiB',round(f['memory']['MemTotal']/2**30), 'available',round(f['memory']['MemAvailable']/2**30), 'CPUs',f['cpus'],'load',f['load'],'b2 jobs',bool(f['b2_services']))
