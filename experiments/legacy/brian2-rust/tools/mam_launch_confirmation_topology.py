"""One bounded new-seed topology audit, with no simulator execution."""
import hashlib,json,os,shlex,subprocess,time
from pathlib import Path
from mam_confirmation_identity import identity,BUILD,NODES
from mam_collect_native_primary_raw import remote
from mam_benchmark_terminal import guard

ROOT=Path(BUILD)/'confirmation-topology-v1-seed1750'
PREFIX='b2mpi-confirm1750-topology-v1'
LIMIT=300


def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def run(evidence,replicate=1750):
    global ROOT,PREFIX
    value=identity(evidence,replicate)
    ROOT=Path(BUILD)/f'confirmation-topology-v1-seed{replicate}'
    PREFIX=f'b2mpi-confirm{replicate}-topology-v1'
    out=evidence/f'confirmation-topology-v1-seed{replicate}';assert not out.exists()
    out.mkdir();write(out/'identity.json',value)
    tools=Path(__file__).parent
    bundle={n:(tools/n).read_text() for n in ['mam_confirmation_topology.py','mpi_resource_guard.py']}
    bundle['count.rs']=(evidence/'hotspot-sharing/hotspot-sharing--count.rs').read_text()
    hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in bundle.items()}
    write(out/'intent.json',dict(maximum_counter_compiles=1,maximum_full_recounts=1,neural_runs=0,
        maximum_edges=1100000000,wall_seconds=LIMIT,memory_gib=8,cpu_cores=4,cpu_ids=[8,9,10,11],
        file_limit_mib=64,minimum_free_gib=1280,source_sha256=hashes,automatic_retry=False,
        prior_counter_integer_reference_cases=2,prior_counter_source_reused_unchanged=True))
    started=time.monotonic()
    try:
        pre=remote(NODES[0],f'''import json,os,time,signal,subprocess,datetime
from pathlib import Path
signal.alarm(20);b=Path('/data/brick2');assert b.is_mount() and b.stat().st_dev!=Path('/').stat().st_dev
assert not Path({str(ROOT)!r}).exists()
assert not subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5).strip()
s=os.statvfs(b);free=s.f_bavail*s.f_frsize;assert free>=1296*2**30
m={{k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())}};assert m['MemAvailable']>=80*2**30
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for l in Path('/proc/stat').read_text().splitlines() if (x:=l.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for c in [8,9,10,11]:
 d=[y-x for x,y in zip(a[c],z[c])];assert sum(d)>0;busy[c]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/4<25
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,available_memory_bytes=m['MemAvailable'],cpu_busy_percent=busy,active_units=[])))''')
        write(out/'admission.json',pre)
        staged=remote(NODES[0],f'''import json,sys,hashlib
from pathlib import Path
r=json.loads(sys.stdin.read());b=Path({str(ROOT)!r});b.mkdir();(b/'tmp').mkdir()
for n,s in r['source'].items():(b/n).open('x').write(s)
(b/'identity.json').open('x').write(r['identity_raw'])
print(json.dumps({{n:hashlib.sha256((b/n).read_bytes()).hexdigest() for n in r['source']}}))''',json.dumps(dict(source=bundle,identity_raw=(out/'identity.json').read_text())).encode())
        assert staged==hashes;write(out/'stage.json',staged);assert time.monotonic()-started<60
        app=['env','PYTHONDONTWRITEBYTECODE=1','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
            'TMPDIR='+str(ROOT/'tmp'),'RUSTUP_HOME=/atlas-home/0003/workspace/brian2-lk-20260906/.rustup',
            'CARGO_HOME=/atlas-home/0003/workspace/brian2-lk-20260906/.cargo',
            '/usr/bin/python3',str(ROOT/'mam_confirmation_topology.py'),str(ROOT)]
        cmd=['systemd-run','--expand-environment=no','--quiet','--wait','--pipe','--collect',
            '--unit='+PREFIX+'-count','--uid=rock','--service-type=exec','--property=MemoryMax=8192M',
            '--property=MemorySwapMax=0','--property=CPUQuota=400%','--property=AllowedCPUs=8-11',
            '--property=TasksMax=64','--property=RuntimeMaxSec=305','--property=TimeoutStopSec=5',
            '--property=KillMode=control-group','--property=OOMPolicy=continue','/usr/bin/python3',
            str(ROOT/'mpi_resource_guard.py'),'--output',str(ROOT/'guard.json'),'--volume','/data/brick2',
            '--memory-mib','8192','--cpu-percent','400','--file-mib','64','--min-free-gib','1280',
            '--timeout',str(LIMIT),'--',*app]
        write(out/'command.json',dict(command=cmd))
        with (out/'count.log').open('x') as log:
            rc=subprocess.run(['tsh','ssh','root@'+NODES[0],shlex.join(cmd)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=LIMIT+25).returncode
        controls=remote(NODES[0],f'''import json,subprocess
from pathlib import Path
b=Path({str(ROOT)!r});names=['guard.json','pending.json','recipes.json','exact-counts.jsonl'];r={{}}
for n in names:
 p=b/n
 if p.exists():assert p.stat().st_size<2**20;r[n]=p.read_text()
r['active_units']=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',{PREFIX+'-*'!r}],text=True,timeout=5).strip()
print(json.dumps(r))''')
        for n,s in controls.items():
            if n!='active_units':(out/n).open('x').write(s)
        write(out/'terminal-state.json',dict(returncode=rc,active_units=controls['active_units']))
        assert rc==0 and not controls['active_units']
        g=json.loads(controls['guard.json']);assert g['command']==app
        spec=dict(host=NODES[0],role='count',memory_bytes=8*2**30,pids_max=64,cpu_ids=[8,9,10,11],cpu_quota_cores=4,
            volume='/data/brick2',allow_root_volume=False,file_limit_bytes=64*2**20,
            minimum_free_bytes=1280*2**30,reserved_host_memory_bytes=64*2**30)
        accounting=guard(g,spec,PREFIX,LIMIT);r=json.loads(controls['pending.json'])
        assert r['counter_source_sha256']==hashes['count.rs'] and r['instance_sha256']==value['instance_sha256']
        for n,key in [('recipes.json','recipe_sha256'),('exact-counts.jsonl','counts_sha256'),('identity.json','identity_sha256')]:
            assert hashlib.sha256((out/n).read_bytes()).hexdigest()==r[key]
        report=dict(complete=True,topology_audit_accepted=True,replicate=replicate,report=r,accounting=accounting,
            neural_runs=0,scientific_acceptance=False,elapsed_seconds=time.monotonic()-started,
            evidence_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
        write(out/'complete.json',report);print(json.dumps(report),flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(error=str(exc),elapsed_seconds=time.monotonic()-started,automatic_retry=False))
        raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parents[1]/'mpi-evidence')
    p.add_argument('--replicate',type=int,choices=[1750,1751],default=1750)
    a=p.parse_args();run(a.evidence,a.replicate)
