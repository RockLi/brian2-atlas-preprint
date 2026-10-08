"""Serial two-node procedural pilot with enforced systemd resource guards."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import uuid

from mpi_teleport_launch import launch


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    base=args.remote_base
    if not base.startswith('/data/brick2/'):
        raise ValueError('pilot artifacts must be on the independent data volume')
    guard=base+'/mpi_resource_guard.py'
    completed=[]
    for edges in [262144,4194304]:
        model=base+f'/models/edges-{edges}'
        label=f'edges-{edges}-reference'
        command=['systemd-run','--expand-environment=no','--quiet','--wait','--pipe','--collect',
                 '--unit=b2mpi-reference-'+uuid.uuid4().hex[:12],'--uid=rock','--service-type=exec',
                 '--property=MemoryMax=2048M','--property=MemorySwapMax=0','--property=CPUQuota=200%',
                 '--property=AllowedCPUs=0-7','--property=TasksMax=64','--property=RuntimeMaxSec=125',
                 '--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=OOMPolicy=continue',
                 '/usr/bin/python3',guard,'--output',base+'/guards/'+label+'.json','--timeout','120','--',
                 '/usr/bin/time','-v','-o',base+'/metrics/'+label+'.time',
                 base+'/repo/brian2-rust/target/release/b2-runner',model+'/model.json',base+'/runs/'+label]
        print('START',label,flush=True)
        result=subprocess.run(['tsh','ssh','root@'+args.nodes[0],shlex.join(command)],
                              stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=145)
        (args.output/(label+'.log')).write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(label+' failed; see log')
        completed.append(label);print('DONE',label,flush=True)
        for ranks in [1,2,4]:
            label=f'edges-{edges}-rank-{ranks}'
            project=model+f'/rank-{ranks}'
            nodes=args.nodes[:1] if ranks==1 else args.nodes
            ips=args.ips[:1] if ranks==1 else args.ips
            app=['sh','-c','exec /usr/bin/time -v -o '+shlex.quote(base+'/metrics/'+label+'-rank')
                 +'"${PMI_RANK}.time" '+shlex.join([project+'/b2-mpi',project+'/instance.bin',base+'/runs/'+label])]
            print('START',label,flush=True)
            launch(nodes=nodes,ips=ips,ranks_per_node=max(1,ranks//len(nodes)),remote_base=base,
                   application=app,output=args.output/label,mpi_prefix=args.mpi_prefix,timeout=120,
                   login='root',guard_script=guard,guard_memory_mib=2048,guard_cpu_percent=200)
            completed.append(label);print('DONE',label,flush=True)
        (args.output/'completed.json').write_text(json.dumps(completed,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nodes',nargs=2,required=True);p.add_argument('--ips',nargs=2,required=True)
    p.add_argument('--remote-base',required=True);p.add_argument('--mpi-prefix',required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
