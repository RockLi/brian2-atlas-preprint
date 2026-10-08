"""Scan complete new Rust target outputs after hash-bound terminal success.

Run once under a separately admitted Linux guard on node 23. The pending
report is not acceptance until that guard and the external controller finish.
Existing full binary readers and ownership accounting are reused unchanged.
"""
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

from mam_rust_benchmark_terminal import audit as audit_terminal,require,MODEL,PLAN,EXE_SHA,target_contract
from mam_launch_rust_performance import BRICK,OLD,LABEL,CASE,PACKAGE_SHA,NODES
from mam_launch_performance_tuning import PROTOCOL_SHA
from mam_rust_terminal_sync import parse,stamp

BASE=Path('/data/brick2/brian2-mpi-region-20260907')
RUN=Path(BRICK)/'runs'/LABEL
PARAMETERS_SHA='ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e'
HIST_SHA='2911d901cd8d82f51beaaec5122b7888ac0dfc618aad245ef58af70ccf36cbab'
CAP=64*2**20


def read_pinned(path,digest,cap):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'bounded regular control required')
    raw=path.read_bytes()
    require(len(raw)<=cap and hashlib.sha256(raw).hexdigest()==digest,'control hash differs: '+str(path))
    return parse(raw)


def terminal_gate(case,binding_path,binding_sha):
    """No simulator output is opened and no output directory is created here."""
    binding=read_pinned(binding_path,binding_sha,2**20)
    from mam_launch_rust_recovery import CASE as NEW_CASE,PROTOCOL_SHA as NEW_PROTOCOL
    protocols={CASE:PROTOCOL_SHA,NEW_CASE:NEW_PROTOCOL}
    require(binding['schema']=='b2-mam-rust-raw-input-binding-v1' and binding['case_id'] in protocols
            and binding['protocol_sha256']==protocols[binding['case_id']],'raw input binding identity')
    files=binding['files']
    expected={'admission.json','launch.json','package.json','terminal/report.json'}|{
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
    require(files['package.json']==PACKAGE_SHA,'original package pin')
    contract=target_contract(values['admission.json'])
    require(binding['case_id']==contract['case_id'] and binding['protocol_sha256']==contract['protocol_sha256'],
            'raw binding and admitted target differ')
    report=values['terminal/report.json']
    require(report['input_sha256']==dict(admission=files['admission.json'],launch=files['launch.json'],package=PACKAGE_SHA),
            'terminal input hashes differ')
    require(report['verifier_sha256']==hashlib.sha256(Path(__file__).with_name('mam_rust_benchmark_terminal.py').read_bytes()).hexdigest(),
            'terminal verifier changed')
    rows=report['collection_files']
    require(len(rows)==4 and [r['host'] for r in rows]==NODES
            and [r['file'] for r in rows]==['host-'+str(i)+'.json.gz' for i in range(4)],'collected host file coverage')
    for item in rows:
        name='terminal/'+item['file']
        require(files[name]==item['sha256'] and (case/name).stat().st_size==item['bytes'],'terminal source receipt differs')
    hosts=[values['terminal/'+r['file']] for r in rows]
    reproduced=audit_terminal(values['admission.json'],values['launch.json'],hosts)
    require(all(report[k]==v for k,v in reproduced.items()),'terminal audit not reproducible')
    require(report['raw_binary_payloads_collected'] is False and report['automatic_retry'] is False,
            'unexpected terminal collection scope')
    return binding,values['package.json'],hosts,reproduced


def run(case,binding_path,binding_sha,output):
    started=time.monotonic()
    binding,package,hosts,terminal=terminal_gate(case,binding_path,binding_sha)
    run_directory=Path(BRICK)/'runs'/terminal['label']
    require(os.uname().sysname=='Linux' and os.uname().nodename==NODES[0],'node 23 Linux required')
    require(Path('/data/brick2').is_mount() and output.parent.is_dir()
            and output.parent.resolve().is_relative_to(BASE.resolve()) and not output.exists(),'new output on brick2 required')
    units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5)
    own='b2mpi-rust-target-raw-v1-audit.service'
    require(all(line.split()[0]==own for line in units.splitlines() if line.strip()),'another own job remains live')
    from mam_correlation_stream import file_sha,mapped_cache
    from mam_output_work_audit import audit_work,select_shared_counts
    from mam_extend_duration import output_bytes
    from brian2_rust.results import load_results
    import numpy as np
    output.mkdir()
    attempt=dict(schema='b2-mam-rust-target-raw-attempt-v1',binding_sha256=binding_sha,
                 wall_limit_seconds=7200,automatic_retry=False)
    (output/'attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
    try:
        source_hashes={}
        def source(path,digest,cap=512*2**20):
            require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'bounded source file')
            require(file_sha(path)==digest,'frozen source changed: '+str(path))
            source_hashes[str(path)]=digest
            return parse(path.read_bytes())
        primary=BASE/OLD
        model=source(primary/'model.json',MODEL)
        source(primary/'parameters.json',PARAMETERS_SHA)
        plan=source(primary/'mpi/execution-plan.json',package['catalog']['mpi/execution-plan.json']['sha256'])
        placement=source(primary/'placement.json',package['catalog']['placement.json']['sha256'])
        owners=plan['population_owners']
        require(owners==placement['population_owners'] and len(owners)==254
                and owners.count(None)==1 and owners[88] is None,'frozen population placement')
        require(len(model['definition']['populations'])==254 and len(model['definition']['synapses'])==8344
                and sum(p['count'] for p in model['definition']['populations'])==4129924,'full model geometry')
        hist_path=BASE/'hotspot-sharing/exact-counts.jsonl'
        require(hist_path.stat().st_size==17664 and file_sha(hist_path)==HIST_SHA,'shared edge histogram pin')
        source_hashes[str(hist_path)]=HIST_SHA
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
            require(work['exact_local_edges']==[r['exact_total_edges'] for r in placement['ranks']]
                    and sum(work['exact_local_edges'])==24126516728
                    and max(work['exact_local_edges'])<=1024000000,'recurrent edge ownership/cap differs')
            sizes=output_bytes(model,summary['spike_count'],sum(len(p['last_spikes']) for p in loaded['populations']),compact_spikes=True)
            require(sizes['results_bytes']==expected['binary_file_bytes']['results.bin']
                    and sizes['events_bytes']==expected['binary_file_bytes']['events.bin'],'exact output byte formula')
            require(max(p['spikes']*8 for p in work['population_profile'])<=16*2**30,'per-population spool ceiling')
        require(all(stamp((run_directory/name).stat(follow_symlinks=False))==s for name,s in snapshots.items()),'output changed during full audit')
        elapsed=time.monotonic()-started;require(elapsed<=7200,'raw audit deadline exceeded')
        report=dict(schema='b2-mam-rust-target-raw-pending-v1',case_id=terminal['case_id'],label=terminal['label'],
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
