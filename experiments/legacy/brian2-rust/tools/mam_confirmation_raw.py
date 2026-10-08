"""Scan complete confirmation outputs after identity-bound terminal success.

Run once under a separately admitted Linux guard on node 23. The pending
report is not acceptance until that guard and the external controller finish.
Existing full binary readers and ownership accounting are reused unchanged.
"""
from mam_confirmation_profile import REPLICATE,selected,worker_environment,IDENTITY_SHA_1751
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

from mam_confirmation_terminal import audit as audit_terminal,require,context_gate,ADMISSION_SHA,COUNTS_SHA
from mam_confirmation_remote_terminal import audit as audit_remote_terminal,PROVENANCE_NAME,PROVENANCE_SHA
from mam_confirmation_identity import BRICK,NODES
from mam_launch_confirmation_run import CASE,LABEL
from mam_confirmation_terminal_sync import IDENTITY_SHA,parse,stamp
IDENTITY_SHA=selected(IDENTITY_SHA,IDENTITY_SHA_1751)

BASE=Path('/data/brick2/brian2-mpi-region-20260907')
PARAMETERS_SHA='ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e'
CAP=64*2**20


def read_pinned(path,digest,cap):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'bounded regular control required')
    raw=path.read_bytes()
    require(len(raw)<=cap and hashlib.sha256(raw).hexdigest()==digest,'control hash differs: '+str(path))
    return parse(raw)


def terminal_names(report):
    """Select an explicit evidence schema; never infer success from missing launch data."""
    require(report['schema'] in {'b2-mam-confirmation-terminal-v1','b2-mam-confirmation-remote-terminal-v1'},
            'unknown terminal evidence schema')
    control=PROVENANCE_NAME if report['schema']=='b2-mam-confirmation-remote-terminal-v1' else 'launch.json'
    return ['admission.json',control,'identity.json','protocol.json','terminal/report.json']+[
        'terminal/host-'+str(i)+'.json.gz' for i in range(4)]


def terminal_gate(case,binding_path,binding_sha):
    """No simulator output is opened and no output directory is created here."""
    binding=read_pinned(binding_path,binding_sha,2**20)
    require(binding['schema'] in {'b2-mam-confirmation-raw-binding-v1','b2-mam-confirmation-remote-raw-binding-v1'} and binding['case_id']==CASE
            and binding['identity_sha256']==IDENTITY_SHA,'raw input binding identity')
    files=binding['files']
    recovered=binding['schema']=='b2-mam-confirmation-remote-raw-binding-v1'
    control=PROVENANCE_NAME if recovered else 'launch.json'
    expected={'admission.json',control,'identity.json','protocol.json','terminal/report.json'}|{
        'terminal/host-'+str(i)+'.json.gz' for i in range(4)}
    require(set(files)==expected,'raw control binding coverage')
    values={}
    for name in sorted(files):
        path=case/name
        require(path.resolve().is_relative_to(case.resolve()),'control path outside case')
        if name.endswith('.gz'):
            require(path.is_file() and not path.is_symlink() and path.stat().st_size<=CAP,'bounded compressed controls')
            require(hashlib.sha256(path.read_bytes()).hexdigest()==files[name],'compressed control hash differs')
            with gzip.open(path,'rb') as stream:raw=stream.read(CAP+1)
            require(len(raw)<=CAP,'expanded controls exceed cap')
            values[name]=parse(raw)
        else:values[name]=read_pinned(path,files[name],16*2**20)
    require(files['identity.json']==IDENTITY_SHA and files['admission.json']==ADMISSION_SHA,'pinned launch identity')
    contract=context_gate(values['identity.json'],values['protocol.json'],values['admission.json'])
    require(binding['case_id']==contract['case_id'] and binding['protocol_sha256']==contract['protocol_sha256'],
            'raw binding and admitted target differ')
    report=values['terminal/report.json']
    require(set(terminal_names(report))==set(files),'terminal schema and binding route differ')
    if recovered:require(files[control]==PROVENANCE_SHA and not (case/'launch.json').exists(),'remote provenance pin or mixed local receipt')
    inputs=dict(admission=files['admission.json'],identity=IDENTITY_SHA,protocol=files['protocol.json'])
    inputs['recovery' if recovered else 'launch']=files[control]
    require(report['input_sha256']==inputs,
            'terminal input hashes differ')
    require(report['verifier_sha256']==hashlib.sha256(Path(__file__).with_name('mam_confirmation_remote_terminal.py' if recovered else 'mam_confirmation_terminal.py').read_bytes()).hexdigest(),
            'terminal verifier changed')
    rows=report['collection_files']
    require(len(rows)==4 and [r['host'] for r in rows]==NODES
            and [r['file'] for r in rows]==['host-'+str(i)+'.json.gz' for i in range(4)],'collected host file coverage')
    for item in rows:
        name='terminal/'+item['file']
        require(files[name]==item['sha256'] and (case/name).stat().st_size==item['bytes'],'terminal source receipt differs')
    hosts=[values['terminal/'+r['file']] for r in rows]
    verifier=audit_remote_terminal if recovered else audit_terminal
    reproduced=verifier(values['identity.json'],values['protocol.json'],values['admission.json'],values[control],hosts)
    require(all(report[k]==v for k,v in reproduced.items()),'terminal audit not reproducible')
    require(report['raw_binary_payloads_collected'] is False and report['automatic_retry'] is False,
            'unexpected terminal collection scope')
    return binding,values['identity.json'],hosts,reproduced


def run(case,binding_path,binding_sha,output):
    started=time.monotonic()
    binding,identity,hosts,terminal=terminal_gate(case,binding_path,binding_sha)
    run_directory=Path(identity['output'])
    MODEL,PLAN,EXE_SHA=[identity[k] for k in ['model_sha256','plan_sha256','executable_sha256']]
    require(output==Path(identity['raw_audit']),'confirmation audit output path differs')
    require(os.uname().sysname=='Linux' and os.uname().nodename==NODES[0],'node 23 Linux required')
    require(Path('/data/brick2').is_mount() and output.parent.is_dir()
            and output.parent.resolve().is_relative_to(BASE.resolve()) and not output.exists(),'new output on brick2 required')
    units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5)
    own=f'b2mpi-confirm{REPLICATE}-raw-v1-audit.service'
    require(all(line.split()[0]==own for line in units.splitlines() if line.strip()),'another own job remains live')
    from mam_correlation_stream import file_sha,mapped_cache
    from mam_output_work_audit import audit_work,select_shared_counts
    from mam_extend_duration import output_bytes
    from brian2_rust.results import load_results
    import numpy as np
    output.mkdir()
    attempt=dict(schema='b2-mam-confirmation-raw-attempt-v1',replicate=REPLICATE,binding_sha256=binding_sha,
                 wall_limit_seconds=7200,automatic_retry=False)
    (output/'attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
    try:
        source_hashes={}
        def source(path,digest,cap=512*2**20):
            require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'bounded source file')
            require(file_sha(path)==digest,'frozen source changed: '+str(path))
            source_hashes[str(path)]=digest
            return parse(path.read_bytes())
        primary=Path(identity['source_artifact'])
        model=source(primary/'model.json',MODEL)
        source(primary/'parameters.json',PARAMETERS_SHA)
        plan=source(primary/'mpi/execution-plan.json',identity['files']['mpi/execution-plan.json']['sha256'])
        placement=source(primary/'placement.json',identity['files']['placement.json']['sha256'])
        require(model['protocol']['layers']['instance']==identity['instance_sha256']
                and plan['instance_sha256']==identity['instance_sha256'],'new instance layer identity')
        owners=plan['population_owners']
        require(owners==placement['population_owners'] and len(owners)==254
                and owners.count(None)==1 and owners[88] is None,'frozen population placement')
        require(len(model['definition']['populations'])==254 and len(model['definition']['synapses'])==8344
                and sum(p['count'] for p in model['definition']['populations'])==4129924,'full model geometry')
        hist_path=BASE/f'confirmation-topology-v1-seed{REPLICATE}/exact-counts.jsonl'
        require(0<hist_path.stat().st_size<2**20 and file_sha(hist_path)==COUNTS_SHA,'shared edge histogram pin')
        source_hashes[str(hist_path)]=COUNTS_SHA
        histogram=select_shared_counts([parse(line) for line in hist_path.read_bytes().splitlines()],model,owners)
        expected=hosts[0]['leader_outputs']
        snapshots={}
        for name in ['results.bin','events.bin','summary.json','mpi-runtime.json']:
            path=run_directory/name;s=path.stat(follow_symlinks=False)
            require(stat.S_ISREG(s.st_mode) and path.resolve().is_relative_to(Path(BRICK).resolve()),'regular admitted output')
            snapshots[name]=stamp(s)
        require(not (run_directory/'spike-spool').exists(),'spool lifecycle not terminal')
        for name,key in [('summary.json','summary_json'),('mpi-runtime.json','runtime_json')]:
            require((run_directory/name).stat().st_size<=32*2**20 and (run_directory/name).read_bytes()==expected[key].encode(),
                    'metadata changed after terminal collection')
        runtime=parse(expected['runtime_json'])
        hashes={}
        for name in ['results.bin','events.bin']:
            require((run_directory/name).stat().st_size==expected['binary_file_bytes'][name]<=65504*2**20,'binary size changed')
            hashes[name]=file_sha(run_directory/name)
        with ExitStack() as stack:
            loaded=load_results(model,run_directory,include_times=False,release_file_cache=True)
            releases=[]
            for key,name in [('_dump','results.bin'),('_event_dump','events.bin')]:
                require(isinstance(loaded[key],np.memmap),'independent full binary mapping required')
                stack.callback(loaded[key]._mmap.close)
                releases.append(stack.enter_context(mapped_cache(loaded[key],run_directory/name)))
            summary=loaded['metadata']
            require(summary==parse(expected['summary_json']) and summary['mpi']==runtime,'independent reader metadata differs')
            work=audit_work(model,loaded,runtime,owners,histogram,release=releases[0],event_release=releases[1])
            # Per-rank edge/CSR counts were checked against the new-seed histogram by audit_work.
            # Retained placement estimates belong to 1729 and are not expected 1750 realizations.
            require(sum(work['exact_local_edges'])==24126516728
                    and max(work['exact_local_edges'])<=1024000000,'recurrent edge ownership/cap differs')
            sizes=output_bytes(model,summary['spike_count'],sum(len(p['last_spikes']) for p in loaded['populations']),compact_spikes=True)
            require(sizes['results_bytes']==expected['binary_file_bytes']['results.bin']
                    and sizes['events_bytes']==expected['binary_file_bytes']['events.bin'],'exact output byte formula')
            require(max(p['spikes']*8 for p in work['population_profile'])<=16*2**30,'per-population spool ceiling')
        require(all(stamp((run_directory/name).stat(follow_symlinks=False))==s for name,s in snapshots.items()),'output changed during full audit')
        elapsed=time.monotonic()-started;require(elapsed<=7200,'raw audit deadline exceeded')
        report=dict(schema='b2-mam-confirmation-raw-pending-v1',replicate=REPLICATE,identity_sha256=IDENTITY_SHA,case_id=terminal['case_id'],label=terminal['label'],
            protocol_sha256=binding['protocol_sha256'],binding_sha256=binding_sha,model_sha256=MODEL,plan_sha256=PLAN,
            executable_sha256=EXE_SHA,complete_binary_scan_passed=True,terminal_resource_audit_passed=True,
            raw_output_audit_passed=False,audit_guard_passed=False,pending_guard_acceptance=True,
            scientific_acceptance=False,nest_statistical_acceptance=False,performance_cost_acceptance=False,
            sources=source_hashes,dump_sha256=hashes,dump_bytes=expected['binary_file_bytes'],work=work,
            spikes=summary['spike_count'],delivered_edges=summary['synaptic_events'],elapsed_seconds=elapsed,
            reused_scientific_summaries=False,prior_raw_identity_assumed=False,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=['Engineering audit only; shared delivery rank histories cannot be reconstructed from these dumps.',
                'Full file hashes do not themselves establish scientific agreement or cross-run byte identity.',
                'Publish only after the separate whole-job resource guard and controller return successfully.'])
        with (output/'pending.json').open('x') as stream:
            stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        return report
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),
            elapsed_seconds=time.monotonic()-started,automatic_retry=False),indent=2)+'\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['case','binding','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--binding-sha256',required=True)
    a=p.parse_args();result=run(a.case,a.binding,a.binding_sha256,a.output)
    print(json.dumps({k:result[k] for k in ['complete_binary_scan_passed','raw_output_audit_passed','elapsed_seconds']}))
