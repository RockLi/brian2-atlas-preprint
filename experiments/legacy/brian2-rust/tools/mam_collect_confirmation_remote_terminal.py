"""Collect the same 32-rank terminal evidence using pinned remote recovery provenance.

Run only after authoritative observation shows all original services terminal.
This reader never writes launch.json and does not infer local transport success.
"""
import argparse,gzip,hashlib,json,shlex,subprocess,time
from pathlib import Path
from mam_collect_benchmark_terminal import pinned
from mam_collect_confirmation_terminal import host_code,CAP
from mam_confirmation_remote_terminal import audit,provenance_gate,PROVENANCE_NAME,PROVENANCE_SHA
from mam_confirmation_terminal import context_gate,ADMISSION_SHA
from mam_confirmation_terminal_sync import IDENTITY_SHA
from mam_confirmation_identity import NODES
from mam_launch_confirmation_run import read,write
from mam_benchmark_terminal import require


def run(args):
    started=time.monotonic()
    admission=pinned(args.case/'admission.json',ADMISSION_SHA,8*2**20)
    identity=read(args.case/'identity.json',IDENTITY_SHA)
    protocol=read(args.case/'protocol.json',admission['protocol_sha256'])
    provenance=pinned(args.case/PROVENANCE_NAME,PROVENANCE_SHA,2**20)
    context_gate(identity,protocol,admission)
    prefix=provenance_gate(provenance)
    require(not (args.case/'launch.json').exists(),'unexpected local launch receipt; inspect before choosing evidence route')
    require(0<args.wall_seconds<=900,'finite control collection budget required')
    args.output.mkdir(exist_ok=False)
    collected=[];receipts=[]
    try:
        for index,host in enumerate(NODES):
            require(args.wall_seconds-(time.monotonic()-started)>120,'collection budget exhausted before host read')
            code=host_code(admission,identity,index,prefix)
            output=args.output/f'host-{index}.json.gz';stderr=args.output/f'host-{index}.stderr'
            with output.open('xb') as out,stderr.open('xb') as err:
                process=subprocess.run(['tsh','ssh','rock@'+host,shlex.join(['taskset','-c','8,9','python3','-c',code])],
                    stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=120)
            require(process.returncode==0,'Rust terminal host read failed; retained, no retry')
            require(output.stat().st_size<=CAP,'compressed terminal controls too large')
            with gzip.open(output,'rb') as stream:raw=stream.read(CAP+1)
            require(len(raw)<=CAP,'expanded terminal controls too large')
            collected.append(json.loads(raw))
            receipts.append(dict(host=host,file=output.name,bytes=output.stat().st_size,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
        result=audit(identity,protocol,admission,provenance,collected)
        elapsed=time.monotonic()-started;require(elapsed<=args.wall_seconds,'terminal collection wall exceeded')
        result.update(collection_seconds=elapsed,collection_files=receipts,
            input_sha256=dict(admission=ADMISSION_SHA,recovery=PROVENANCE_SHA,identity=IDENTITY_SHA,protocol=admission['protocol_sha256']),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            verifier_sha256=hashlib.sha256(Path(__file__).with_name('mam_confirmation_remote_terminal.py').read_bytes()).hexdigest(),
            raw_binary_payloads_collected=False,automatic_retry=False)
        write(args.output/'report.json',result)
        print(json.dumps(dict(terminal_resource_audit_passed=True,metadata_sync_audit_passed=True,
                             raw_output_audit_passed=False,collection_seconds=elapsed)))
        return result
    except BaseException as error:
        write(args.output/'failure.json',dict(error_type=type(error).__name__,error=str(error),
            elapsed_seconds=time.monotonic()-started,completed_hosts=len(collected),automatic_retry=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['case','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--wall-seconds',type=int,required=True)
    run(p.parse_args())
