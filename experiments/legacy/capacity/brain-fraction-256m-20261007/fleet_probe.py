"""Read-only fleet capacity snapshot; launch no workloads or stop services."""
from pathlib import Path
import concurrent.futures,json
import control
HERE=Path(__file__).resolve().parent
inventory=json.loads(Path('/private/tmp/atlas-scale-fleet.json').read_text())
nodes=sorted(r['spec']['hostname'] for r in inventory if r['spec']['hostname'].startswith('hk-prod-model-'))
code='''from pathlib import Path
import json,os,subprocess,time
m={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemAvailable','MemTotal']}
v=os.statvfs('/');base=Path('/atlas-home/0003/atlas-capacity-86m-20261007')
data=Path('/data/brick2');dv=os.statvfs(data) if data.exists() else None
print(json.dumps({'host':os.uname().nodename,'utc_epoch':time.time(),'memory':m,'root_free_bytes':v.f_bavail*v.f_frsize,'data_free_bytes':dv.f_bavail*dv.f_frsize if dv else None,'data_is_mount':data.is_mount(),'experiment_base_exists':base.exists(),'active_mpi_services':subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True).strip()}))'''
def probe(node):
    try:return {'node':node,'data':control.c.remote(node,code)}
    except Exception as error:return {'node':node,'error':str(error)[-800:]}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(probe,nodes))
(HERE/'fleet-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
ok=[r['data'] for r in rows if 'data' in r]
print(json.dumps({'accessible_model_nodes':len(nodes),'probed':len(ok),'errors':len(rows)-len(ok),'available_memory_tib_sum':sum(r['memory']['MemAvailable'] for r in ok)/2**40,'memavailable_gib_minmax':[min(r['memory']['MemAvailable'] for r in ok)/2**30,max(r['memory']['MemAvailable'] for r in ok)/2**30],'scope':'Non-synchronized snapshots of available memory, not allocated or reserved capacity.'}))
