"""Run the pinned replicate1750 executable and durably finish its outputs.

All work remains inside the caller's job guard and external wall measurement.
This is not an independent binary/scientific audit or all-rank success receipt.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

IDENTITY_SHA = '9a5a3258ca7b33b3d5296bc74398ba19e3455f3e3e61ffcb872f7a6b374165de'
PINNED_IDENTITIES = {1750:IDENTITY_SHA,1751:'eaa1d4a12a666c32760ccb4660d0333e960c57f2b1c5a8417e274f7cf3178e19'}
HOSTS = ['hk-prod-model-ae02-23','hk-prod-model-ae08-81','hk-prod-model-ae08-83','hk-prod-model-ae07-71']
FINAL_CAP = 128*2**30
META_CAP = 32*2**20


def require(ok, message):
    if not ok: raise ValueError(message)


def parse(raw):
    def pairs(items):
        result = {}
        for key,value in items:
            require(key not in result, 'duplicate metadata key')
            result[key] = value
        return result
    def constant(value): raise ValueError('nonfinite metadata: '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def stamp(value):
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


def sync_directory(path):
    descriptor = os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def receipt(path, value):
    raw = (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    require(len(raw) <= 65536,'bounded terminal receipt required')
    with path.open('xb') as stream:
        require(stream.write(raw)==len(raw),'short receipt write')
        stream.flush(); os.fsync(stream.fileno())
    sync_directory(path.parent)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def sync_outputs(directory, plan_sha256):
    """Validate and sync four closed output files, using bounded metadata reads."""
    started = time.monotonic(); directory = Path(directory)
    require(directory.resolve()==directory and directory.is_dir(),'resolved output directory required')
    require(not (directory/'spike-spool').exists(),'spool lifecycle is not terminal')
    names = ['results.bin','events.bin','mpi-runtime.json','summary.json']
    with ExitStack() as stack:
        streams={}; stats={}; metadata={}; digests={}
        for name in names:
            descriptor = os.open(directory/name,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
            stream = stack.enter_context(os.fdopen(descriptor,'rb',buffering=0))
            value = os.fstat(stream.fileno())
            require(stat.S_ISREG(value.st_mode) and 0 < value.st_size <= (META_CAP if name.endswith('.json') else FINAL_CAP),
                    'bounded regular output required')
            streams[name]=stream; stats[name]=value
            if name.endswith('.json'):
                raw = stream.read(META_CAP+1)
                require(len(raw)==value.st_size,'metadata size changed')
                metadata[name]=parse(raw); digests[name]=hashlib.sha256(raw).hexdigest()
        require(sum(s.st_size for s in stats.values()) <= FINAL_CAP,'combined final output ceiling')
        summary,runtime = metadata['summary.json'],metadata['mpi-runtime.json']
        require(summary['schema']=='b2-result-dump-v4' and summary['population_count']==254
                and summary['neuron_count']==4129924 and summary['final_time_seconds']==100.5,
                'full target summary identity')
        require(runtime['schema']=='b2-mpi-runtime-v0' and runtime['ranks']==32
                and runtime['plan_sha256']==plan_sha256 and summary['mpi']==runtime,
                'terminal runtime identity')
        require(summary['dump_bytes']==stats['results.bin'].st_size
                and summary['event_dump_bytes']==stats['events.bin'].st_size,'declared binary size differs')
        for name,stream in streams.items():
            os.fsync(stream.fileno())
            if name in digests:
                stream.seek(0); digest=hashlib.sha256()
                while raw := stream.read(65536): digest.update(raw)
                require(digest.hexdigest()==digests[name],'metadata readback differs')
            require(stamp(os.fstat(stream.fileno()))==stamp(stats[name]),'output changed during synchronization')
            require(stamp(os.stat(directory/name,follow_symlinks=False))==stamp(stats[name]),'output path replaced')
        sync_directory(directory); sync_directory(directory.parent)
    return dict(files={n:dict(bytes=stats[n].st_size,**({'sha256':digests[n]} if n in digests else {})) for n in names},
                file_sync_calls=4,output_and_parent_directories_synced=True,
                metadata_readback_verified=True,binary_contents_independently_audited=False,
                elapsed_seconds=time.monotonic()-started)


def finish_child(returncode, rank, output, receipts, common, started):
    """Failures cannot accept the job; a partially persisted receipt is retained.

    Collectors must reject failure files and nonzero exits even when a done
    filename exists after a receipt/directory synchronization error.
    """
    require(type(returncode) is int,'explicit child return code required')
    if returncode:
        receipt(receipts/f'rank{rank}.failed.json',dict(**common,event='child_failed',returncode=returncode))
        raise RuntimeError('Frozen Rust executable failed; no synchronization success')
    try:
        synchronization = sync_outputs(output, common['plan_sha256']) if rank==0 else None
        value = dict(**common,event='rank_wrapper_complete',child_returncode=0,
                     leader_output_synchronization=synchronization,
                     wall_through_child_and_output_sync_seconds=time.monotonic()-started,
                     whole_job_accepted=False,scientific_acceptance=False)
        receipt(receipts/f'rank{rank}.done.json',value)
        print(json.dumps(dict(event='rust_rank_wrapper_complete',rank=rank,leader_output_synced=rank==0)),flush=True)
        return value
    except BaseException as error:
        failure=receipts/f'rank{rank}.failed.json'
        if not failure.exists():
            try: receipt(failure,dict(**common,event='terminal_sync_failed',error_type=type(error).__name__,error=str(error)))
            except Exception: pass
        raise


def pmi_pass_fds(environment=None):
    """Hydra passes its PMI socket by descriptor; preserve only that channel."""
    environment=os.environ if environment is None else environment
    value=environment.get('PMI_FD','')
    require(isinstance(value,str) and value.isascii() and value.isdecimal(),
            'frozen Hydra launch requires a numeric PMI_FD')
    descriptor=int(value)
    require(descriptor>=3,'PMI_FD must not alias standard streams')
    os.fstat(descriptor)
    return (descriptor,)


def run_child(command):
    return subprocess.run(command,check=False,pass_fds=pmi_pass_fds())


def load_identity(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 2**20,
            'bounded regular identity required')
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() in PINNED_IDENTITIES.values(),'replicate identity digest differs')
    value=parse(raw)
    replica=value.get('replicate')
    require(type(replica) is int and replica in PINNED_IDENTITIES,'replicate identity differs')
    require(hashlib.sha256(raw).hexdigest()==PINNED_IDENTITIES[replica],'replicate identity digest differs')
    require(value['schema']=='b2-mam-confirmation-identity-v1','replicate identity schema differs')
    return value


def verify_input(path, item, root):
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
            and path.stat().st_size==item['bytes'],'input location or size differs')
    with path.open('rb') as stream:
        require(hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256'],'input digest differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity',type=Path,required=True)
    args=parser.parse_args();value=load_identity(args.identity)
    require(sys.platform=='linux' and sys.maxsize>2**32,'Linux64 terminal synchronization required')
    rank,size=int(os.environ['PMI_RANK']),int(os.environ['PMI_SIZE'])
    host=os.uname().nodename
    require(size==32 and 0<=rank<32 and HOSTS[rank//8]==host,'frozen rank/host placement required')
    root=Path('/data/brick2') if rank<8 else Path('/home/rock')
    project=Path(value['project']);executable=project/'mpi/b2-mpi';instance=project/'mpi/instance.bin'
    for name in ['mpi/b2-mpi','mpi/instance.bin',f'mpi/instance.rank-{rank}.bin']:
        verify_input(project/name,value['files'][name],root)
    require(os.access(executable,os.X_OK),'executable permission missing')
    output=Path(value['output']);receipts=Path(value['receipts'])
    require(output.is_absolute() and output.is_relative_to(Path('/data/brick2')),'leader output must be on brick2')
    if rank==0:
        require(output.parent.is_dir() and output.parent.resolve()==output.parent
                and not output.exists(),'new resolved output required')
    require(receipts.resolve().is_relative_to(root),'receipts outside admitted host storage')
    require(receipts.parent.is_dir(),'pre-existing receipt parent required')
    receipts.mkdir(exist_ok=True);sync_directory(receipts.parent)
    common=dict(rank=rank,ranks=32,host=host,replicate=value['replicate'],
                identity_sha256=PINNED_IDENTITIES[value['replicate']],model_sha256=value['model_sha256'],
                instance_sha256=value['instance_sha256'],executable_sha256=value['executable_sha256'],
                plan_sha256=value['plan_sha256'],
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    receipt(receipts/f'rank{rank}.started.json',dict(**common,event='rank_wrapper_started'))
    started=time.monotonic()
    result=run_child([str(executable),str(instance),str(output)])
    finish_child(result.returncode,rank,output,receipts,common,started)


if __name__=='__main__':main()
