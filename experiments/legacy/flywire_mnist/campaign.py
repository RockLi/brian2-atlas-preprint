"""Durable full-run coordinator; no services, cron jobs or shared checkout edits."""
import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from .protocol import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['data','graph','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12);parser.add_argument('--fit-threads',type=int,default=8)
    parser.add_argument('--shard-size',type=int,default=32)
    args=parser.parse_args();args.output=args.output.resolve()
    if args.workers<1 or args.fit_threads<1:parser.error('resource counts must be positive')
    state=args.output.with_name(args.output.name+'-campaign.json')
    log=args.output.with_name(args.output.name+'-campaign.log')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    with state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time()
        common=[sys.executable,'-m','flywire_mnist.full']
        suffix=['--data',str(args.data.resolve()),'--output',str(args.output)]
        steps=[]
        if not (args.output/'protocol.json').exists():
            if args.output.exists():raise RuntimeError('partial initialization; preserve it and choose a new output directory')
            steps.append(('init',['--graph',str(args.graph.resolve()),'--shard-size',str(args.shard_size)]))
        steps.append(('extract',['--phase','train','--workers',str(args.workers)]))
        if not (args.output/'test-lock.json').exists():steps.append(('fit',['--threads',str(args.fit_threads)]))
        steps.append(('extract',['--phase','test','--workers',str(args.workers)]))
        if not (args.output/'report.json').exists():steps.append(('evaluate',[]))
        info={'pid':os.getpid(),'hostname':socket.gethostname(),'platform':platform.platform(),
              'workers':args.workers,'fit_threads':args.fit_threads,'started_unix':started,
              'output':str(args.output),'log':str(log)}
        with log.open('a',buffering=1) as output:
            for stage,extra in steps:
                command=common+[stage]+suffix+extra
                atomic_json(state,{**info,'status':'running','stage':stage,'command':command})
                output.write(json.dumps({'command':command,'unix':time.time()})+'\n');output.flush()
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
                try:code=process.wait()
                except BaseException:
                    process.terminate();process.wait()
                    atomic_json(state,{**info,'status':'interrupted','stage':stage});raise
                if code:
                    atomic_json(state,{**info,'status':'failed','stage':stage,'exit_code':code})
                    raise SystemExit(code)
        atomic_json(state,{**info,'status':'complete','elapsed_seconds':time.time()-started,
                          'report':str(args.output/'report.json')})


if __name__=='__main__':main()
