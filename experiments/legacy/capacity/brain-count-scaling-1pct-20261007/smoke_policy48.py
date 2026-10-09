"""Bounded real-kernel check of the staged admission guard on one worker."""
from pathlib import Path
import json
import control as c

HERE=Path(__file__).resolve().parent
c.configure(30);node=c.NODES[18]
child=f'''from pathlib import Path
import json,os,resource
assert os.getuid()==1000 and sorted(os.sched_getaffinity(0))==[1]
assert resource.getrlimit(resource.RLIMIT_FSIZE)==(64*2**30,64*2**30)
p=Path({c.BASE!r})/'policy48-smoke-child.json';assert not p.exists()
p.write_text(json.dumps({{'uid':os.getuid(),'cpu_ids':sorted(os.sched_getaffinity(0)),'file_limits':resource.getrlimit(resource.RLIMIT_FSIZE)}})+'\\n')'''
args=['systemd-run','--expand-environment=no','--wait','--collect',
      '--unit=b2mpi-policy48-smoke-20261008-v1','--uid=rock',
      '--property=MemoryMax=512M','--property=MemorySwapMax=0',
      '--property=CPUQuota=100%','--property=AllowedCPUs=1',
      '--property=TasksMax=16','--property=RuntimeMaxSec=30',
      '--property=TimeoutStopSec=5','--property=KillMode=control-group',
      'python3',c.BASE+'/guard-host-reserve48-v3.py',
      '--output',c.BASE+'/policy48-smoke-guard.json','--volume','/',
      '--allow-root-volume','--memory-mib','512','--cpu-percent','100',
      '--file-mib','65536','--min-free-gib','128','--timeout','20','--',
      'python3','-c',child]
code=f'''from pathlib import Path
import subprocess,json,hashlib
b=Path({c.BASE!r});assert not (b/'policy48-smoke-guard.json').exists()
p=subprocess.run({args!r},capture_output=True,text=True,timeout=40)
assert p.returncode==0,p.stderr
print(json.dumps({{'guard':json.loads((b/'policy48-smoke-guard.json').read_text()),'child':json.loads((b/'policy48-smoke-child.json').read_text()),'guard_sha256':hashlib.sha256((b/'guard-host-reserve48-v3.py').read_bytes()).hexdigest()}}))'''
result=c.remote(node,code,root=True,timeout=50)
g=result['guard'];assert g['admitted'] and g['returncode']==0
assert g['reserved_host_memory_bytes']==48*2**30
assert g['before']['memory.max']==str(512*2**20) and g['before']['memory.swap.max']=='0'
assert result['guard_sha256']==json.loads((HERE/'protocol-revision-v3.json').read_text())['guard_sha256']
c.record('policy48-kernel-smoke.json',dict(passed=True,host=node,**result))
print(json.dumps({'passed':True,'host':node,'guard_headroom_gib':48,'kernel_memory_mib':512,'physical_cpus':result['child']['cpu_ids']}))
