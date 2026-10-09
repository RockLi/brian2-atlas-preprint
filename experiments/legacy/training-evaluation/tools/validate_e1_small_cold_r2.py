#!/usr/bin/env python3
"""Read-only stdlib audit of the corrected E1-small cold-r2 evidence.

Never imports/runs an engine, oracle, fixture generator or old cold benchmark.
Evidence consistency and a successful scientific measurement are separate.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import statistics
import sys

MEASURE = 'tools/measure_e1_small_cold_r2.py'
MEASURE_SHA = '7d1e9df13d9252a3bc5d636621975185ceeb8d7f29306c571270473397114424'
SEEDS = [11,23,37,51,71]
VIEWS = [('atlas','cpu','atlas',False),('sj-layerwise','cpu','spikingjelly_frontier',False),
         ('snn-layerwise','cpu','snntorch_fp64',False),('spyx','jax','spyx',False),
         ('brainstate','jax','brainx_state',False),('sj-compile','cpu','spikingjelly_frontier',True),
         ('snn-compile','cpu','snntorch_fp64',True)]
JAX = {'spyx','brainstate'}
REQUIRED_IDENTITIES = [MEASURE,'tools/benchmark_e1.py','tools/benchmark_competitor.py',
    'tools/run_recurrent_queue.py','tools/generate_dense_large.py','tools/generate_recurrent_cases.py',
    'tools/run_a1_queue_v4.py','tools/run_qualification_followup_queue.py',
    'adapters/atlas_adapter.py','adapters/torch_adapter.py','adapters/jax_adapter.py','adapters/oracle.py',
    'runtime/b2-train','environment/cpu-lock.txt','environment/jax-lock.txt',
    'fixtures/e1-small/manifest.json','evidence/e1-small-validation-full-r1.json',
    'environment/hardware.json','sources/snapshot-manifest.json']


def order():
    return [dict(seed=s,view=v,environment=e,engine=n,compiled=c)
            for i,s in enumerate(SEEDS) for v,e,n,c in VIEWS[i:]+VIEWS[:i]]


def finite(value,positive=False):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and (value>0 if positive else value>=0)


def near(a,b):
    return finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)


def hash_string(value):
    return isinstance(value,str) and bool(re.fullmatch('[0-9a-f]{64}',value))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('Duplicate JSON key: '+k)
            out[k]=v
        return out
    def invalid(token):raise ValueError('Nonfinite JSON token: '+token)
    with Path(path).open() as f:return json.load(f,object_pairs_hook=pairs,parse_constant=invalid)


def within(root,value):
    p=(root/value).resolve();p.relative_to(root.resolve());return p


class Audit:
    def __init__(self):self.errors=[];self.limitations=[]
    def check(self,condition,message):
        if not condition:self.errors.append(message)
        return bool(condition)
    def note(self,message):
        if message not in self.limitations:self.limitations.append(message)
    def load(self,path,required=False,interrupted=False):
        if not path.is_file():
            if required:self.errors.append('Missing '+str(path))
            return None
        try:return read(path)
        except (OSError,ValueError) as e:
            (self.limitations if interrupted else self.errors).append(f'{path}: {type(e).__name__}: {e}')
            return None
    def identity(self,path,expected):
        try:return self.check(hash_string(expected) and sha(path)==expected,'SHA256 mismatch: '+str(path))
        except OSError as e:return self.check(False,f'Cannot hash {path}: {e}')


def command_args(command):
    if not isinstance(command,list) or len(command)<2 or not all(isinstance(v,str) for v in command):raise ValueError('Invalid command vector')
    values={};flags=set();i=2
    bare={'--worker','--compiled','--resource-qualified'}
    valued={'--output','--engine','--seed','--expected-loss','--array-sha','--phase-token','--started-ns'}
    while i<len(command):
        k=command[i]
        if k in flags or k in values:raise ValueError('Duplicate command argument '+k)
        if k in bare:flags.add(k);i+=1
        elif k in valued and i+1<len(command):values[k]=command[i+1];i+=2
        else:raise ValueError('Unexpected/incomplete command argument '+k)
    if set(values)!=valued or '--worker' not in flags:raise ValueError('Missing formal worker command arguments')
    return values,flags


def status_for(row,supervisor,child,outer_reason=None,cap=None,spent_after=None):
    reason=supervisor.get('termination_reason')
    # Outer observed termination always beats a child completed record. A
    # polling/cleanup overrun remains timeout; it is not grace for ranking.
    if reason in ('timeout','resource_limit'):return reason
    if supervisor and ((finite(supervisor.get('elapsed_s')) and finite(cap) and supervisor['elapsed_s']>cap)
        or (finite(spent_after) and spent_after>1800)):return 'timeout'
    if child and finite(child.get('process_to_first_update_s')) and finite(cap) and child['process_to_first_update_s']>cap:return 'timeout'
    if outer_reason in ('timeout','resource_limit') and row.get('status') in ('launching','not_launched','coordinator_failed_during_slot'):
        return outer_reason if supervisor or row.get('status') in ('launching','coordinator_failed_during_slot') else 'not_launched_phase_'+outer_reason
    return row.get('status') or child.get('status') or 'unresolved'


def cache_check(cache,folder,a):
    expected={'TORCHINDUCTOR_CACHE_DIR':'torch','TRITON_CACHE_DIR':'triton','JAX_COMPILATION_CACHE_DIR':'jax'}
    a.check(isinstance(cache,dict),'Missing initial cache inventory')
    if not isinstance(cache,dict):return
    a.check(set(cache.get('directories',{}))==set(expected),'Cache inventory does not contain all three declared caches')
    for key,leaf in expected.items():
        value=cache.get('directories',{}).get(key,{})
        a.check(value.get('path')==str(folder/'cache'/leaf) and value.get('exists') is False,'Cache was not declared fresh: '+key)
    removed=cache.get('cleared_environment_keys')
    a.check(isinstance(removed,list) and all(isinstance(v,str) for v in removed) and len(removed)==len(set(removed)),'Malformed cache environment-removal receipt')
    a.check('not claimed' in cache.get('scope','') or 'No claim' in cache.get('scope',''),'Cache receipt lacks bounded cold-scope disclosure')
    a.check('disabled' in cache.get('remote_cache_policy',''),'Cache receipt lacks remote-cache policy')


def worker_check(child,args,slot,reference,preflight,a):
    if child.get('launcher_start_ns') is None:
        # Atlas software refusal reports status+failure before clock fields.
        a.check(slot['engine']=='atlas' and child.get('status')!='completed' and isinstance(child.get('failure'),dict),'Worker missing identity/clock without a recorded Atlas software refusal')
        return False
    for k,want in [('engine',slot['engine']),('seed',slot['seed']),('compiled',slot['compiled']),('array_sha256',reference.get('array_sha256'))]:
        a.check(child.get(k)==want,'Worker identity differs: '+k)
    start,end=child.get('launcher_start_ns'),child.get('first_update_complete_ns')
    clock_ok=(isinstance(start,int) and not isinstance(start,bool) and isinstance(end,int) and not isinstance(end,bool) and end>start>0)
    a.check(clock_ok,'Invalid integer monotonic clock endpoints')
    if clock_ok:
        a.check(str(start)==args.get('--started-ns'),'Worker start clock differs from launcher command')
        a.check(near(child.get('process_to_first_update_s'),(end-start)/1e9),'First-update clock arithmetic differs')
    a.check(child.get('clock')=='same-host time.monotonic_ns across coordinator and fresh worker processes','Clock-domain declaration differs')
    expected_loss=reference.get('raw_loss_trajectory',[None])[0]
    a.check(child.get('reference_loss')==expected_loss,'Worker first-loss reference differs from inherited full qualification')
    loss=child.get('loss');finite_loss=isinstance(loss,(int,float)) and not isinstance(loss,bool) and math.isfinite(loss)
    a.check(child.get('loss_finite') is finite_loss,'Finite-loss flag differs from actual recorded loss')
    match=finite_loss and isinstance(expected_loss,(int,float)) and abs(loss-expected_loss)<=1e-10+1e-8*abs(expected_loss)
    inherited=preflight.get('environments',{}).get(slot['environment'],{})
    versions=inherited.get('historical_version_binding',{})
    a.check(child.get('historical_runtime_versions')==versions,'Worker runtime-version binding differs from metadata preflight')
    runtime_match=child.get('current_python')==inherited.get('python') and all(child.get('implementation',{}).get(k)==v for k,v in versions.items() if slot['engine']!='atlas')
    a.check(child.get('runtime_version_match') is runtime_match,'Worker runtime-version comparison differs')
    expected_status='runtime_version_mismatch' if not runtime_match else 'numerical_divergence' if not finite_loss else 'completed' if match else 'first_loss_mismatch'
    a.check(child.get('status')==expected_status,'Worker scientific status differs from first-loss/runtime checks')
    resource=bool(reference.get('resource_qualification')) and slot['environment']!='jax'
    for key in ('resource_qualification','strict_ranking_eligible'):a.check(child.get(key) is resource,'Worker resource flag differs: '+key)
    if slot['engine']=='atlas' and child.get('status')=='completed':a.check(child.get('implementation',{}).get('returned_step')==1,'Atlas first update did not return Adam step1')
    return bool(clock_ok and match and runtime_match)


def normalize(name):return re.sub(r'[-_.]+','-',name).lower()


def lock_versions(path):
    out={}
    for raw in path.read_text().splitlines():
        line=raw.strip()
        if not line or line.startswith('#'):continue
        if line.count('==')!=1:raise ValueError('Unsupported lock line')
        k,v=line.split('==');k=normalize(k)
        if k in out:raise ValueError('Duplicate lock package')
        out[k]=v
    return out


def provenance(root,run,freeze,a,full):
    a.check(freeze.get('schema')=='E1-small-cold-r2' and freeze.get('status')=='execution_frozen','Wrong frozen cold revision/status')
    a.check(freeze.get('order')==order(),'Cold freeze does not preserve the exact35-slot rotated order')
    a.check(freeze.get('per_slot_cap_s')==360 and freeze.get('per_view_cap_s')==1800,'Cold caps differ from360/1800')
    a.check(freeze.get('preregistered_metric')=='fresh process launch through complete first update','Cold metric boundary differs')
    ids=freeze.get('identities',{})
    a.check(all(k in ids for k in REQUIRED_IDENTITIES),'Freeze missing required source/runtime/qualification identities')
    a.check(ids.get(MEASURE)==MEASURE_SHA,'Freeze names another cold implementation revision')
    prior=a.load(root/'evidence/e1-small-validation-full-r1.json',True) or {}
    a.check(prior.get('status')=='passed_full_evidence_checks' and prior.get('raw_verified') is True and prior.get('errors')==[],'Inherited full qualification audit has not passed')
    a.check(freeze.get('qualified_reference_sha256')==ids.get('evidence/e1-small-validation-full-r1.json'),'Inherited audit digest differs inside freeze')
    refs={}
    for row in prior.get('rows',[]):
        key=(row.get('view'),row.get('seed'))
        a.check(key not in refs,'Duplicate inherited qualified slot');refs[key]=row
    a.check(len(prior.get('rows',[]))==35 and set(refs)=={(r['view'],r['seed']) for r in order()},'Inherited qualification does not contain all35 slots')
    for key,row in refs.items():
        a.check(row.get('numerical_qualification') is True and row.get('effective_status')=='completed' and row.get('errors')==[],'Inherited slot not numerically qualified: '+str(key))
        trajectory=row.get('raw_loss_trajectory',[])
        a.check(len(trajectory)==60 and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in trajectory),'Inherited complete60-step trajectory missing: '+str(key))
    old=freeze.get('original_benchmark_freeze',{})
    current_old=a.load(root/'evidence/remote-benchmark-v1/freeze.json',True) or {}
    a.check(old==current_old,'Embedded historical benchmark freeze differs')
    inherited={**old.get('scripts',{}),**{'environment/'+k:v for k,v in old.get('locks',{}).items()},
       'runtime/b2-train':old.get('runtime_sha256'),'environment/hardware.json':old.get('hardware'),
       'sources/snapshot-manifest.json':old.get('source_manifest')}
    for k,v in inherited.items():a.check(ids.get(k)==v,'Inherited source/runtime differs inside cold freeze: '+k)
    preflight=a.load(run/'installed-preflight.json',True) or {}
    a.check(preflight.get('status')=='matched_historical_versions','Installed preflight did not match qualified versions')
    a.check(set(preflight.get('environments',{}))=={'cpu','jax'},'Installed preflight must cover cpu and jax')
    expected_raw={f"evidence/remote-benchmark-v1/{s['view']}-seed-{s['seed']}/result.json" for s in order()}
    a.check(set(preflight.get('qualified_raw_sha256',{}))==expected_raw,'Preflight must bind all35 inherited raw worker reports')
    for rel,h in preflight.get('qualified_raw_sha256',{}).items():a.check(ids.get(rel)==h,'Inherited worker digest differs inside freeze: '+rel)
    if full:
        for rel,h in ids.items():a.identity(within(root,rel),h)
        a.identity(root/'evidence/remote-benchmark-v1/freeze.json',prior.get('freeze_sha256'))
        a.identity(run/'installed-preflight.json',freeze.get('installed_preflight_sha256'))
        manifest=a.load(root/'fixtures/e1-small/manifest.json',True) or {}
        a.check(manifest.get('schema')=='e1-common-arrays-v1','Common-array manifest revision differs')
        for key,value in [('B',16),('T',128),('sizes',[128,128,10]),('beta',.95),('theta',1.),('seeds',SEEDS),('input_values',[0,1,2]),('input_probabilities',[.9,.095,.005])]:
            a.check(manifest.get('rule',{}).get(key)==value,'Common-array workload differs: '+key)
        files=manifest.get('files',{})
        a.check(set(files)=={f'seed-{s}.json' for s in SEEDS},'Common-array manifest does not bind all five seeds')
        for seed in SEEDS:
            meta=files.get(f'seed-{seed}.json',{});path=root/'fixtures/e1-small'/f'seed-{seed}.json'
            a.identity(path,meta.get('sha256'))
            if path.is_file() and 'bytes' in meta:a.check(path.stat().st_size==meta['bytes'],'Fixture byte size differs')
            for view,_,_,_ in VIEWS:a.check(refs.get((view,seed),{}).get('array_sha256')==meta.get('sha256'),'Inherited slot uses different common arrays')
        snap=a.load(root/'sources/snapshot-manifest.json',True) or {}
        for rel,h in snap.get('source_hashes',{}).items():a.check(ids.get('snapshot/'+rel)==h,'Frozen source snapshot member missing/different: '+rel)
        hardware=a.load(root/'environment/hardware.json',True) or {}
        historical={'cpu':{},'jax':{}}
        for slot in order():
            rel=f"evidence/remote-benchmark-v1/{slot['view']}-seed-{slot['seed']}/result.json"
            raw=a.load(root/rel,True) or {}
            a.check(raw.get('status')=='completed','Inherited raw worker did not complete: '+rel)
            keys=() if slot['engine']=='atlas' else ('jax','optax') if slot['environment']=='jax' else ('torch',)
            for key in keys:
                version=raw.get('implementation',{}).get(key)
                a.check(isinstance(version,str),'Missing historical runtime version: '+key)
                oldversion=historical[slot['environment']].setdefault(key,version)
                a.check(oldversion==version,'Historical runtime versions disagree')
        for env,data in preflight.get('environments',{}).items():
            if env not in ('cpu','jax'):continue
            a.check(data.get('duplicates')==[] and data.get('implementation')=='CPython' and data.get('python')==hardware.get('python'),'Installed Python metadata differs')
            a.check(data.get('distributions')==lock_versions(root/f'environment/{env}-lock.txt'),'Installed distributions differ from qualified lock')
            a.check(data.get('historical_version_binding')==historical[env],'Historical runtime binding differs from raw workers')
            for name,version in historical[env].items():a.check(data.get('distributions',{}).get(normalize(name))==version,'Installed runtime version differs from inherited worker')
            stdout=a.load(run/f'{env}-metadata.stdout.txt',True)
            a.check(stdout=={k:v for k,v in data.items() if k not in ('historical_version_binding','historical_python_binding')},'Metadata stdout differs from preflight structured receipt')
            identity_files=data.get('identity_files',{})
            a.check(bool(identity_files) and data.get('resolved_executable') in identity_files,'Installed identity receipts lack executable')
            for path,h in identity_files.items():a.identity(Path(path),h)
    else:a.note('Report-only: source/runtime/array/installed-file bytes remain unverified; no performance ratio is eligible.')
    return refs,preflight


def summarize(rows,allow):
    by={(r['view'],r['seed']):r for r in rows};summaries={}
    for view,_,_,_ in VIEWS:
        selected=[by[(view,s)] for s in SEEDS]
        allfive=all(r.get('measurement_eligible') is True for r in selected)
        atlasfive=all(by[('atlas',s)].get('measurement_eligible') is True for s in SEEDS)
        resource=all(r.get('resource_qualification') is True for r in selected) and view not in JAX
        atlasresource=all(by[('atlas',s)].get('resource_qualification') is True for s in SEEDS)
        paired=bool(allow and allfive and atlasfive and resource and atlasresource and view!='atlas')
        item=dict(scheduled_n=5,completed_n=sum(r['effective_status']=='completed' for r in selected),
            strict_eligible_n=sum(r.get('measurement_eligible') is True and r.get('resource_qualification') is True for r in selected),
            complete_five_seed_summary=bool(allow and allfive),resource_qualification=resource,
            requested_thread_group=view not in JAX,paired_five_seed_ratio_eligible=paired,
            statuses={str(r['seed']):r['effective_status'] for r in selected},paired_seed_ratios=[])
        if allow and allfive:
            times=[r['process_to_first_update_s'] for r in selected]
            item.update(median_first_update_s=statistics.median(times),min_first_update_s=min(times),max_first_update_s=max(times))
        for seed in SEEDS:
            ratio=by[('atlas',seed)]['process_to_first_update_s']/by[(view,seed)]['process_to_first_update_s'] if paired else None
            item['paired_seed_ratios'].append(dict(seed=seed,eligible=paired,ratio_atlas_over_view=ratio))
        if paired:item['median_paired_ratio_atlas_over_view']=statistics.median(r['ratio_atlas_over_view'] for r in item['paired_seed_ratios'])
        summaries[view]=item
    return summaries


def validate(root,run,mode,phase_path=None):
    root=root.resolve();run=within(root,run)
    if phase_path is not None:phase_path=within(root,phase_path)
    a=Audit();full=mode=='full'
    if full and socket.gethostname().split('.')[0]!='rock-mac-studio-1':raise ValueError('Full validation must run on100.90.28.27')
    phase=a.load(phase_path,True) if phase_path else None
    phase=phase or {};outer_reason=phase.get('termination_reason')
    if phase and outer_reason=='exited' and finite(phase.get('elapsed_s')) and finite(phase.get('timeout_s'),True) and phase['elapsed_s']>phase['timeout_s']:
        outer_reason='timeout'
        a.note('Outer phase normal exit exceeded its recorded cap; treated as phase timeout for comparative eligibility.')
    interrupted=outer_reason in ('timeout','resource_limit')
    terminal=a.load(run/'terminal.json',interrupted=interrupted)
    progress=a.load(run/'progress.json',required=terminal is None and not interrupted,interrupted=interrupted)
    records=terminal.get('records',[]) if isinstance(terminal,dict) else progress if isinstance(progress,list) else []
    a.check(isinstance(records,list),'Cold records must be a list')
    if not isinstance(records,list):records=[]
    ledger={}
    for row in records:
        if not isinstance(row,dict):a.check(False,'Malformed slot record');continue
        key=(row.get('view'),row.get('seed'))
        a.check(key not in ledger,'Duplicate cold slot '+str(key));ledger[key]=row
    formal={(s['view'],s['seed']) for s in order()}
    a.check(not(set(ledger)-formal),'Unexpected cold slot outside declared35')
    if records:
        a.check([{k:r.get(k) for k in ('seed','view','environment','engine','compiled')} for r in records if isinstance(r,dict)]==order(),'Recorded cold ledger does not preserve exact35-slot rotated order')
    if terminal:
        a.check(terminal.get('finite_slots')==35 and len(records)==35 and set(ledger)==formal,'Terminal must preserve all35 outcomes')
        a.check(progress==records,'Final progress and terminal ledger differ')
        a.check(terminal.get('completed_slots')==sum(r.get('status')=='completed' for r in records),'Terminal completed-slot count differs')
        a.check(terminal.get('status') in ('completed','incomplete','coordinator_failed'),'Unknown coordinator terminal status')
        if terminal.get('supervisor_completed') is True:
            a.check(terminal.get('status')==('completed' if all(r.get('status')=='completed' for r in records) else 'incomplete'),'Coordinator aggregate status differs')
        cleanup=terminal.get('exception_cleanup')
        if cleanup and not cleanup.get('observed_owned_tree_clean'):a.note('Exception cleanup was not verified; all comparative eligibility withheld.')
    elif not interrupted:a.note('Coordinator terminal is absent; all35 slots retained but the phase is unresolved and no ratios are eligible.')
    if phase:
        if phase.get('remaining_owned_processes'):a.note('External phase reports surviving owned processes; no ratio eligible.')
        if outer_reason not in ('exited','timeout','resource_limit'):a.check(False,'Unknown external phase termination')
        # The phase binding is a path/command receipt, not a substitute for the
        # cold freeze. A phase stdout/log is never treated as worker evidence.
        command=phase.get('command',[])
        a.check(len(command)>1 and within(root,command[1])==root/MEASURE,'External phase command does not identify cold-r2 measurement')
        if '--output' in command:a.check(within(root,command[command.index('--output')+1])==run,'External phase output differs')
        else:a.check(False,'External phase does not identify output directory')
    freeze=a.load(run/'freeze.json');refs={};preflight={}
    if freeze:refs,preflight=provenance(root,run,freeze,a,full)
    else:
        a.note('No execution freeze exists: consistent preflight/coordinator non-execution is not an engine result or evidence corruption.')
        if full:a.identity(root/MEASURE,MEASURE_SHA)
    out=dict(schema='e1-small-cold-evidence-validation-r2',root=str(root),run=str(run),mode=mode,
        audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_revision=MEASURE_SHA,
        errors=a.errors,limitations=a.limitations,rows=[],views={},frozen=bool(freeze),
        cold_scope='fresh process launch through complete first actual Adam update; fixture/import/constructor/compile included; supervisor post-update receipt/cleanup separately recorded',
        inherited_qualification='Prior frozen full-shape60-step evidence is reused; this cold run adds only first-loss/runtime checks, not fresh full numerical qualification',
        previous_cold_comparison_performed=False,model_or_oracle_executed=False,
        denominator=dict(formal_slots=35,formal_seeds=SEEDS,views=[v[0] for v in VIEWS],recorded_slots=len(ledger),retained_rows=35,
            terminal_present=bool(terminal),phase_interrupted=interrupted))
    spent={v[0]:0. for v in VIEWS};unaccounted={v[0]:0. for v in VIEWS};tokens=set()
    for slot in order():
        key=(slot['view'],slot['seed']);record=ledger.get(key,{})
        name=f"{slot['view']}-seed-{slot['seed']}";folder=run/name;ra=Audit()
        row=dict(**slot,recorded_status=record.get('status'),errors=ra.errors,limitations=ra.limitations,
            resource_qualification=False,strict_ranking_eligible=False,measurement_eligible=False,
            inherited_numerical_qualification=False,first_loss_matches=False)
        out['rows'].append(row)
        if record:
            for field,want in slot.items():ra.check(record.get(field)==want,'Slot identity differs: '+field)
        for field in ('resource_qualification','strict_ranking_eligible'):
            if slot['environment']=='jax':ra.check(record.get(field,False) is False,'JAX resource/ranking flag must remain false')
        supervisor=record.get('supervisor') or record.get('partial_supervisor') or {}
        perjob_path=run/(name+'-terminal.json');perjob=ra.load(perjob_path,interrupted=interrupted)
        if perjob:
            if supervisor:ra.check(supervisor==perjob,'Coordinator/per-job supervisor receipts differ')
            else:supervisor=perjob;ra.note('Recovered per-job receipt absent from final coordinator accounting')
        if record.get('supervisor') and not perjob:ra.check(False,'Recorded launch lacks its per-job terminal')
        cap=min(360.,1800.-spent[slot['view']])
        args={};flags=set();command=supervisor.get('command')
        if command:
            ra.check(bool(freeze),'A worker was launched without an execution freeze')
            ra.check(supervisor.get('name')==name,'Supervisor job name differs')
            ra.check(near(supervisor.get('timeout_s'),cap) and cap>0,'Launched cap differs from min(360, remaining1800)')
            ra.check(supervisor.get('rss_guard_bytes')==64*1024**3,'Memory guard differs from64GiB')
            ra.check(finite(supervisor.get('elapsed_s'),True),'Invalid supervisor wall duration')
            ra.check(supervisor.get('termination_reason') in ('exited','timeout','resource_limit'),'Unknown per-slot termination reason')
            ra.check(isinstance(supervisor.get('exit_code'),int),'Missing supervisor exit code')
            if supervisor.get('termination_reason')=='timeout' and finite(supervisor.get('elapsed_s')):ra.check(supervisor['elapsed_s']>=cap,'Supervisor claims timeout before cap')
            if supervisor.get('remaining_owned_processes'):ra.note('Owned cleanup not verified; no measurement eligibility')
            identities=supervisor.get('process_identities',[])
            ra.check(isinstance(identities,list) and bool(identities),'Missing observed process identities')
            for identity in identities:ra.check(isinstance(identity.get('pid'),int) and finite(identity.get('create_time'),True),'Malformed process identity')
            try:
                args,flags=command_args(command)
                ra.check(command[:2]==[str(root/f"environment/{slot['environment']}/bin/python"),str(root/MEASURE)],'Worker executable/source path differs')
                for flag,want in [('--output',str(folder)),('--engine',slot['engine']),('--seed',str(slot['seed']))]:ra.check(args.get(flag)==want,'Worker launch argument differs: '+flag)
                ra.check(('--compiled' in flags) is slot['compiled'],'Worker compile flag differs')
                ra.check(bool(re.fullmatch('[0-9a-f]{32}',args['--phase-token'])),'Malformed coordinator lease token')
                tokens.add(args['--phase-token'])
                ra.check(int(args['--started-ns'])>0,'Invalid launcher monotonic start')
            except (ValueError,KeyError) as e:ra.check(False,'Cannot parse formal worker command: '+str(e))
            cache_check(ra.load(folder/'initial-cache-inventory.json',True),folder,ra)
            if finite(supervisor.get('elapsed_s')):
                (spent if record.get('supervisor') else unaccounted)[slot['view']]+=supervisor['elapsed_s']
        elif record.get('status')=='aggregate_timeout_not_launched':ra.check(cap<=0,'Aggregate timeout omitted a slot while per-view budget remained')
        elif not freeze:ra.check(record.get('status') in (None,'not_launched','not_launched_coordinator_failure','coordinator_failed_during_slot'),'Missing freeze with a claimed engine outcome')
        childpath=folder/'cold-result.json'
        child=ra.load(childpath,interrupted=interrupted or supervisor.get('termination_reason') in ('timeout','resource_limit')) or {}
        if child:
            if not command and interrupted:
                ra.note('Worker result recovered after outer interruption before a per-job launch receipt was saved; identity cannot qualify for comparison.')
            else:ra.check(bool(command),'Worker result exists without a bound launch receipt')
            if record.get('worker') is not None:
                ra.check(record['worker']==child,'Coordinator/worker results differ')
                if full:ra.identity(childpath,record.get('worker_sha256'))
            elif not interrupted and record.get('status')!='coordinator_failed_during_slot':ra.check(False,'Completed coordinator did not retain worker report/hash')
            else:ra.note('Recovered worker report after coordinator interruption; it cannot qualify for comparison')
        elif record.get('worker') is not None:ra.check(False,'Coordinator embeds a worker report but its original file is missing/unreadable')
        reference=refs.get(key,{})
        resource=bool(reference.get('resource_qualification')) and slot['environment']!='jax'
        if freeze:
            for field,want in [('resource_qualification',resource),('strict_ranking_eligible',resource),('inherited_numerical_qualification',True),('array_sha256',reference.get('array_sha256'))]:
                if record:ra.check(record.get(field)==want,'Coordinator inherited qualification differs: '+field)
        if command and reference:
            ra.check(args.get('--array-sha')==reference.get('array_sha256'),'Command arrays differ from inherited qualification')
            ra.check(('--resource-qualified' in flags) is resource,'Command resource gate differs')
            trajectory=reference.get('raw_loss_trajectory',[])
            if trajectory:
                try:ra.check(float(args.get('--expected-loss','nan'))==trajectory[0],'Command first-loss reference differs')
                except ValueError:ra.check(False,'Invalid command expected loss')
        matched=worker_check(child,args,slot,reference,preflight,ra) if child and reference and args else False
        recorded_expected=None
        if supervisor:
            if supervisor.get('termination_reason')!='exited':recorded_expected=supervisor.get('termination_reason')
            elif supervisor.get('exit_code')!=0:recorded_expected=child.get('status','worker_failed') if child else 'worker_failed_without_result'
            elif not child:recorded_expected='worker_missing_result'
            else:recorded_expected=child.get('status')
            if record.get('status') not in ('coordinator_failed_during_slot',None):ra.check(record.get('status')==recorded_expected,'Recorded worker/supervisor outcome differs')
        effective=status_for(record,supervisor,child,outer_reason,cap,spent[slot['view']]+unaccounted[slot['view']])
        if not record:effective='not_launched_phase_'+outer_reason if interrupted else 'unresolved'
        if child.get('status')=='completed' and supervisor.get('termination_reason')=='exited' and supervisor.get('exit_code')!=0:effective='worker_failed_after_update'
        row.update(effective_status=effective,worker_status=child.get('status'),termination_reason=supervisor.get('termination_reason'),
            exit_code=supervisor.get('exit_code'),budget_s=supervisor.get('timeout_s'),supervisor_wall_s=supervisor.get('elapsed_s'),
            process_to_first_update_s=child.get('process_to_first_update_s'),array_sha256=reference.get('array_sha256'),
            inherited_numerical_qualification=bool(reference.get('numerical_qualification')),first_loss_matches=matched,
            resource_qualification=resource,source_and_fixture_bytes_verified=bool(full and freeze),
            worker_sha256=sha(childpath) if childpath.is_file() else None,
            strict_slot_wall_pass=bool(finite(supervisor.get('elapsed_s'),True) and 0<cap and supervisor['elapsed_s']<=cap
                and finite(child.get('process_to_first_update_s'),True) and child['process_to_first_update_s']<=cap),
            per_view_spent_after_s=spent[slot['view']]+unaccounted[slot['view']])
        if effective=='timeout' and supervisor.get('termination_reason')=='exited':ra.note('Observed wall/clock or cumulative wall exceeded the strict cap despite normal exit; retained as timeout and excluded from ratios.')
        row['measurement_eligible']=bool(effective=='completed' and matched and row['strict_slot_wall_pass']
            and row['per_view_spent_after_s']<=1800 and not ra.errors and not supervisor.get('remaining_owned_processes')
            and record.get('supervisor') and record.get('worker_sha256'))
        for message in ra.errors:a.errors.append(name+': '+message)
    a.check(len(tokens)<=1,'Cold workers belong to different coordinator lease tokens')
    if terminal:
        declared=terminal.get('spent_seconds',{})
        for view,total in spent.items():
            if freeze:a.check(near(declared.get(view),total),'Per-view charged wall differs: '+view)
            elif view in declared:a.check(near(declared[view],total),'Preflight-only spent total differs: '+view)
    fully_traversed=bool(terminal and terminal.get('supervisor_completed') is True)
    allow=bool(full and freeze and fully_traversed and not a.errors and not interrupted and not phase.get('remaining_owned_processes')
        and not(terminal.get('exception_cleanup') and not terminal['exception_cleanup'].get('observed_owned_tree_clean')))
    for row in out['rows']:
        row['measurement_eligible']=bool(allow and row['measurement_eligible'])
        row['strict_ranking_eligible']=bool(row['measurement_eligible'] and row['resource_qualification'])
    out.update(recomputed_spent_seconds=spent,unaccounted_observed_wall_seconds=unaccounted,
        views=summarize(out['rows'],allow),all35_recorded=len(ledger)==35,supervisor_completed=fully_traversed,
        all35_scientifically_completed=all(r['effective_status']=='completed' for r in out['rows']),
        freeze_sha256=sha(run/'freeze.json') if freeze else None,
        phase_terminal_sha256=sha(phase_path) if phase_path and phase_path.is_file() else None)
    a.limitations.extend([
        'No prior cold values are compared: only this corrected fresh-process boundary can enter the new paired summaries.',
        'A first-loss match is not a new full numerical qualification; eligibility inherits the exact frozen full E1-small audit and current runtime/version/source identity.',
        'Requested-thread group has no CPU-core affinity proof. JAX resource_qualification and strict_ranking_eligible stay false; no Atlas/JAX ratio is emitted.',
        'Five complete strict-cap seeds for both sides are required before any paired ratio; missing, refused, failed, timed-out and unlaunched seeds remain in the denominator.',
        'Even small supervisor polling/cleanup overshoot exceeds the strict cap and yields timeout, not a usable speed ratio. Post-update receipt/cleanup remains separately charged to the supervisor and1800s per-view budget.',
        'Cold means fresh process and the three declared persistent cache paths plus TMPDIR. OS/page caches, compiler executables and unenumerated library caches are not proved cold; metadata preflight precedes the clock.',
        'Historical installed interpreter/package file bytes were not archived. Recorded versions bind current metadata to old qualification; package code payloads beyond METADATA/RECORD are not individually rehashed.',
        'This audit checks records, hashes and arithmetic, not transient host contention or prediction/gradient equivalence; no model or numerical oracle is run.'])
    out['status']='evidence_invalid' if a.errors else 'consistent_non_execution_or_interrupted' if not freeze or not fully_traversed else 'passed_full_cold_evidence_checks' if full else 'passed_report_checks_bytes_unverified'
    return out


def self_test():
    tests={};sup={'termination_reason':'exited','elapsed_s':10.};child={'status':'completed','process_to_first_update_s':9.};row={'status':'completed'}
    tests['outer_timeout_beats_worker']=status_for(row,{**sup,'termination_reason':'timeout'},child,cap=360,spent_after=10)=='timeout'
    tests['memory_guard_is_not_oom']=status_for(row,{**sup,'termination_reason':'resource_limit'},child,cap=360,spent_after=10)=='resource_limit'
    tests['poll_overshoot_timeout']=status_for(row,{**sup,'elapsed_s':360.00001},child,cap=360,spent_after=360.00001)=='timeout'
    tests['view_overshoot_timeout']=status_for(row,sup,child,cap=360,spent_after=1800.00001)=='timeout'
    tests['clock_overshoot_timeout']=status_for(row,sup,{**child,'process_to_first_update_s':360.00001},cap=360,spent_after=10)=='timeout'
    tests['exact_cap_permitted']=status_for(row,{**sup,'elapsed_s':360.},{**child,'process_to_first_update_s':360.},cap=360,spent_after=1800)=='completed'
    tests['outer_phase_active_timeout']=status_for({'status':'launching'},{},{},'timeout',360,0)=='timeout'
    tests['outer_phase_unlaunched_retained']=status_for({'status':'not_launched'},{},{},'timeout',360,0)=='not_launched_phase_timeout'
    tests['preflight_failure_retained']=status_for({'status':'not_launched_coordinator_failure'},{},{},cap=360,spent_after=0)=='not_launched_coordinator_failure'
    command=['python','measure.py','--worker','--output','/tmp/x','--engine','atlas','--seed','11','--expected-loss','0.1','--array-sha','a'*64,'--phase-token','b'*32,'--started-ns','1']
    values,flags=command_args(command);tests['valid_command_parsed']=values['--seed']=='11' and flags=={'--worker'}
    try:command_args(command+['--seed','23']);tests['duplicate_command_rejected']=False
    except ValueError:tests['duplicate_command_rejected']=True
    rows=[dict(**s,effective_status='completed',measurement_eligible=True,resource_qualification=s['environment']!='jax',process_to_first_update_s=2. if s['view']=='atlas' else 4.) for s in order()]
    complete=summarize(rows,True)
    tests['paired_five_seed_ratio']=all(r['ratio_atlas_over_view']==.5 for r in complete['snn-layerwise']['paired_seed_ratios'])
    tests['jax_never_ratio']=not complete['spyx']['paired_five_seed_ratio_eligible'] and all(r['ratio_atlas_over_view'] is None for r in complete['brainstate']['paired_seed_ratios'])
    censored=[{**r,'measurement_eligible':False,'effective_status':'timeout'} if r['view']=='snn-layerwise' and r['seed']==71 else dict(r) for r in rows]
    result=summarize(censored,True)['snn-layerwise']
    tests['censored_seed_blocks_all_five_ratios']=result['scheduled_n']==5 and result['completed_n']==4 and all(r['ratio_atlas_over_view'] is None for r in result['paired_seed_ratios'])
    atlasbad=[{**r,'measurement_eligible':False} if r['view']=='atlas' and r['seed']==11 else dict(r) for r in rows]
    tests['missing_atlas_seed_blocks_pairs']=not summarize(atlasbad,True)['snn-layerwise']['paired_five_seed_ratio_eligible']
    tests['report_only_no_ratios']=all(not v['paired_five_seed_ratio_eligible'] for v in summarize(rows,False).values())
    badresource=[{**r,'resource_qualification':False} if r['view']=='atlas' and r['seed']==11 else dict(r) for r in rows]
    tests['inherited_resource_gate_required']=not summarize(badresource,True)['snn-layerwise']['paired_five_seed_ratio_eligible']
    a=Audit();cache_check({'directories':{k:{'path':'/tmp/x/cache/'+v,'exists':False} for k,v in [('TORCHINDUCTOR_CACHE_DIR','torch'),('TRITON_CACHE_DIR','triton'),('JAX_COMPILATION_CACHE_DIR','jax')]},'cleared_environment_keys':[],'scope':'No claim about OS caches','remote_cache_policy':'disabled'},Path('/tmp/x'),a)
    tests['fresh_cache_receipt_passes']=not a.errors
    b=Audit();cache_check({'directories':{},'cleared_environment_keys':[],'scope':'No claim','remote_cache_policy':'disabled'},Path('/tmp/x'),b)
    tests['missing_cache_rejected']=bool(b.errors)
    tests['bool_clock_not_number']=not finite(True)
    tests['all35_rotation_denominator']=len(order())==35 and len({(r['view'],r['seed']) for r in order()})==35
    return dict(schema='e1-small-cold-validator-synthetic-v2',status='passed' if all(tests.values()) else 'failed',tests=tests,
        real_evidence_read=False,model_imports=False,old_cold_comparison=False)


def markdown(value):
    lines=['E1-small corrected cold-r2 evidence audit', '', 'Status: '+value['status'], '',
           'All 35 formal slots and five seeds per view remain in the denominator. Paired ratios require five completed strict-cap seeds on both sides and inherited requested-thread qualification.', '',
           '| View | Completed/5 | Strict eligible/5 | Five-seed median first update (s) | Median Atlas/view ratio |',
           '| --- | --- | --- | --- | --- |']
    for view,_,_,_ in VIEWS:
        v=value.get('views',{}).get(view,{})
        lines.append('| '+view+' | '+str(v.get('completed_n',0))+'/5 | '+str(v.get('strict_eligible_n',0))+'/5 | '+str(v.get('median_first_update_s','—'))+' | '+str(v.get('median_paired_ratio_atlas_over_view','—'))+' |')
    lines += ['', '| View | Seed | Recorded outcome | Effective outcome | First update (s) | Supervisor wall (s) |', '| --- | --- | --- | --- | --- | --- |']
    for r in value.get('rows',[]):
        lines.append('| '+' | '.join(str(r.get(k,'—')) for k in ('view','seed','recorded_status','effective_status','process_to_first_update_s','supervisor_wall_s'))+' |')
    if value.get('errors'):lines += ['', 'Evidence errors:', '']+['- '+e for e in value['errors']]
    lines += ['', 'Interpretation limits:', '']+['- '+e for e in value.get('limitations',[])]
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path);p.add_argument('--run',type=Path)
    p.add_argument('--mode',choices=['full','report-only'],default='report-only')
    p.add_argument('--phase-terminal',type=Path);p.add_argument('--output',type=Path);p.add_argument('--report',type=Path)
    p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:value=self_test()
    else:
        if args.root is None:p.error('--root is required; default paths refer only to the new r2 phase')
        args.run=args.run or Path('evidence/remote-e1-small-cold-r2')
        args.phase_terminal=args.phase_terminal or Path('evidence/extended-followup-r2/e1-small-cold-terminal.json')
        args.output=args.output or args.root/'evidence/e1-small-cold-validation-full-r2.json'
        args.report=args.report or args.root/'evidence/e1-small-cold-validation-full-r2.md'
        try:value=validate(args.root.resolve(),within(args.root,args.run),args.mode,within(args.root,args.phase_terminal) if args.phase_terminal else None)
        except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,OverflowError) as e:
            value=dict(schema='e1-small-cold-evidence-validation-r2',status='evidence_invalid',mode=args.mode,
                rows=[dict(**s,effective_status='unresolved_parse_failure',strict_ranking_eligible=False) for s in order()],views={},
                errors=[f'Cannot complete evidence parsing: {type(e).__name__}: {e}'],limitations=['All35 slots retained; no ratio eligible.'])
    value['validator_sha256']=sha(__file__)
    if args.output:
        with args.output.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    if args.report:
        with args.report.open('x') as f:f.write(markdown(value))
    print(json.dumps(dict(status=value['status'],errors=len(value.get('errors',[])),rows=len(value.get('rows',[])),
        output=str(args.output) if args.output else None,tests=value.get('tests')),allow_nan=False))
    return 2 if value.get('errors') or value['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
