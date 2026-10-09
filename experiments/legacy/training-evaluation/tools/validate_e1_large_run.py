#!/usr/bin/env python3
"""Read-only stdlib E1-large evidence validator and Chinese report preparation.

Run full mode on the remote host after the finite queue terminates. This script
streams hashes/raw/resource records, never imports an engine or runs an oracle.
Report-only mode reads small metadata and cannot produce verified rankings.
All output files are exclusive-create. Frozen workers and raw evidence remain
unchanged. Scientific rejection/timeout is an outcome, not an evidence error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

SEEDS = (11, 23, 37, 51, 71)
VIEWS = {'atlas': ('atlas', False, False), 'sj-layerwise': ('spikingjelly_frontier', False, True),
         'snn-layerwise': ('snntorch_fp64', False, True), 'spyx': ('spyx', False, False),
         'brainstate': ('brainx_state', False, False), 'sj-compile': ('spikingjelly_frontier', True, True),
         'snn-compile': ('snntorch_fp64', True, True)}
JAX = ('spyx', 'brainstate')
SIZES = [512, 1024, 1024, 20]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def near(a, b):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def passed(checks):
    return bool(checks) and all(isinstance(row, dict) and row.get('passed') is True for row in checks.values())


def required(checks, names, issues, label):
    missing = set(names) - set(checks)
    if missing:
        issues.append(f'{label}: missing checks {sorted(missing)}')


def records(path, issues):
    if not path.exists():
        return
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except (ValueError, TypeError) as error:
                issues.append(f'{path.name}:{number}: malformed JSON record: {error}')
                return


def identity(path, expected, issues, label, expected_bytes=None):
    if not path.is_file():
        issues.append(f'{label}: missing {path}')
        return False
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        issues.append(f'{label}: file size differs')
    if not isinstance(expected, str) or sha(path) != expected:
        issues.append(f'{label}: SHA256 differs or absent')
        return False
    return True


def artifact(folder, entry, issues, mode):
    if not isinstance(entry, dict) or not entry.get('sha256') or not isinstance(entry.get('arrays'), dict):
        issues.append('qualification artifact identity/schema absent')
        return
    name = entry.get('path', '')
    if Path(name).name != name or not name.endswith('.npz'):
        issues.append('qualification artifact must be an immediate NPZ in its worker folder')
        return
    if mode == 'full':
        identity(folder/name, entry['sha256'], issues, name, entry.get('bytes'))


def admission_check(value, issues):
    """Validate recorded arithmetic/guards, not reconstruct a giant state JSON."""
    if not isinstance(value, dict):
        issues.append('missing admission metadata')
        return
    neurons, parameters = sum(SIZES[1:]), sum(a*b for a,b in zip(SIZES[:-1], SIZES[1:]))
    expected = dict(tapes_and_returned_spikes=32*128*neurons*24, live_state=32*neurons*32,
                    logits=32*SIZES[-1]*16, parameter_gradient_optimizer=parameters*48,
                    topology=0, mpi_workspace=0, inputs=32*128*(SIZES[0]*8+48)+32*8)
    if value.get('components') != expected or value.get('exact_native_tape_bytes') != sum(expected.values()):
        issues.append('admission native tape arithmetic differs from full B32/T128 dense shape')
    initial = 32*neurons*64 + parameters*48
    if value.get('exact_initial_budget_bytes') != initial:
        issues.append('admission initial state arithmetic differs')
    if value.get('input_limit_bytes') != 64*1024**2 or value.get('hard_max_tape_bytes') != 1024**3 or value.get('plan_max_tape_bytes') != 1024**3:
        issues.append('admission software caps differ from frozen profile')
    wire = value.get('request', {})
    if not isinstance(wire.get('bytes'), int) or wire['bytes'] <= 0 or not isinstance(wire.get('sha256'), str) or len(wire['sha256']) != 64:
        issues.append('admission request byte count/hash absent')
        return
    reasons = []
    if wire['bytes'] > 64*1024**2:
        reasons.append('request_json_exceeds_64_mib')
    if parameters*48 > 1024**3:
        reasons.append('parameter_budget_exceeded')
    if initial > 1024**3:
        reasons.append('initial_state_budget_exceeded')
    if sum(expected.values()) > 1024**3:
        reasons.append('native_tape_budget_exceeded')
    if value.get('reasons') != reasons or value.get('status') != ('budget_rejected' if reasons else 'admitted'):
        issues.append('admission status/reasons do not follow recorded software caps')


def worker_metadata(folder, worker, row, mode):
    issues = row['errors'];atlas = row['view'] == 'atlas'
    steps = worker.get('qualification_steps', [])
    if not worker:
        steps = []
        for index in range(1,4):
            path = folder/f'qualification-step-{index}.json'
            if path.is_file():steps.append(read(path))
            else:break
        row['qualification_recovered_from_partial_files'] = True
    qualified = []
    for index, meta in enumerate(steps, 1):
        path = folder/f'qualification-step-{index}.json'
        if not path.is_file() or read(path) != meta:
            issues.append(f'qualification-step-{index} metadata absent/differs from worker result')
        if atlas:
            admission_check(meta.get('admission'), issues)
            if meta.get('optimizer_step_before') != index-1 or meta.get('index') != index or meta.get('phase') != 'qualification':
                issues.append(f'qualification-step-{index} has incorrect Adam counter/phase/index')
        if meta.get('qualified') is True:
            checks = meta.get('checks', {})
            names = ['loss'] + [f'{key}_{bank}' for key in ['weights','first','second'] for bank in range(3)]
            if atlas:
                names += ['logits','spikes','initial_vjp','final_state','optimizer_counter','native_tape_count'] + [f'gradient_{bank}' for bank in range(3)]
                if meta.get('status') != 'executed' or meta.get('tape_count_matches_runtime') is not True or not positive(meta.get('public_api_ns')):
                    issues.append(f'qualification-step-{index} lacks executed complete native API evidence')
            else:
                if meta.get('step') != index or not positive(meta.get('actual_public_s')):
                    issues.append(f'qualification-step-{index} has invalid step/public duration')
            required(checks, names, issues, f'qualification-step-{index}')
            if not passed(checks):
                issues.append(f'qualification-step-{index} claims qualified with failed checks')
            artifact(folder, meta.get('artifact'), issues, mode)
            qualified.append(index)
    numerical = (qualified == [1,2,3] and worker.get('qualification_status') ==
                 ('passed_three_actual_public_Adam_updates' if atlas else 'passed_three_actual_public_Adam_updates_and_full_diagnostics'))
    if numerical and not atlas:
        path = folder/'diagnostics.json'
        if not path.is_file():
            issues.append('complete competitor qualification lacks diagnostics.json')
        else:
            value = read(path);checks = value.get('checks', {})
            required(checks, ['loss','logits','states','spikes','initial_vjp'] + [f'gradient_{i}' for i in range(3)], issues, 'diagnostics')
            if checks != worker.get('diagnostic_checks') or not passed(checks):
                issues.append('competitor diagnostics checks differ/fail')
            artifact(folder, value.get('artifact'), issues, mode)
    row['numerical_qualification'] = numerical
    row['completed_qualification_updates'] = len(qualified)
    row['qualification_status'] = worker.get('qualification_status')
    if atlas and steps and positive(steps[0].get('public_api_ns')):
        row['cold']['first_qualification_public_api_s'] = steps[0]['public_api_ns']/1e9


def raw_and_resources(run, folder, external, worker, row):
    issues = row['errors'];atlas = row['view'] == 'atlas'
    # Admissions persist before native invocation; retain an in-flight last call.
    admissions = list(records(folder/'atlas-admissions.jsonl', issues)) if atlas else []
    seen = set()
    allowed = [('qualification',i) for i in range(1,4)] + [('warmup' if i<10 else 'measured',i) for i in range(60)]
    for record in admissions:
        admission_check(record.get('admission'), issues)
        key = (record.get('phase'), record.get('index'))
        if key in seen:
            issues.append(f'duplicate admission: {key}')
        seen.add(key)
        expected_step = record.get('index', -99) - (1 if record.get('phase') == 'qualification' else 0)
        if record.get('optimizer_step_before') != expected_step:
            issues.append(f'admission Adam counter differs: {key}')
    if [(r.get('phase'),r.get('index')) for r in admissions] != allowed[:len(admissions)]:
        issues.append('Atlas admission sequence is not a prefix of three qualification calls followed by60 actual updates')
    for index,record in enumerate(admissions):
        if record.get('admission',{}).get('status')=='budget_rejected' and index!=len(admissions)-1:
            issues.append('native admission continued after a rejected original request')
    row['admissions'] = admissions
    rejected = [v for v in admissions if v.get('admission', {}).get('status') == 'budget_rejected']
    row['admission_outcome'] = ('later_optimizer_state_budget_rejected' if rejected and rejected[0].get('optimizer_step_before',0) > 0
                               else 'initial_request_budget_rejected' if rejected else 'no_recorded_preflight_rejection')
    if rejected:
        row['rejection'] = rejected[0]
        if row['effective_status'] not in ('budget_rejected','timeout'):
            issues.append('recorded admission rejection conflicts with effective status')
    elif row['effective_status'] == 'budget_rejected':
        row['admission_outcome'] = 'runtime_or_exception_budget_rejection_without_rejected_preflight'
    elapsed = [];losses = [];warmup = 0;raw_count = 0
    admission_map = {(r.get('phase'),r.get('index')): r for r in admissions}
    for index, raw in enumerate(records(folder/'raw-steps.jsonl', issues)):
        raw_count += 1
        phase = 'warmup' if index < 10 else 'measured'
        if raw.get('index') != index or raw.get('phase') != phase or index >= 60:
            issues.append('raw performance indices/phases must be consecutive 0..59')
        if atlas:
            admission_check(raw.get('admission'), issues)
            previous = admission_map.get((phase,index))
            if previous is None or previous.get('admission') != raw.get('admission'):
                issues.append(f'raw step {index} lacks matching pre-call admission')
            if raw.get('status') != 'executed':
                if index != worker.get('failed_performance_index') and row['effective_status'] != 'timeout':
                    issues.append('nonexecuted raw step not reflected in failed performance index')
                continue
            if raw.get('optimizer_step_before') != index or raw.get('optimizer_step') != index+1 or raw.get('tape_count_matches_runtime') is not True:
                issues.append(f'raw step {index} has incorrect optimizer/tape count')
            seconds = raw.get('public_api_ns', 0)/1e9
        else:
            seconds = raw.get('elapsed_s')
        if not positive(seconds):
            issues.append(f'raw step {index} has invalid complete API latency')
            continue
        loss = raw.get('loss')
        if not isinstance(loss, (int,float)) or not math.isfinite(loss):
            if row['effective_status'] != 'numerical_divergence':
                issues.append(f'raw step {index} nonfinite loss not classified as divergence')
            continue
        losses.append(loss)
        if index < 10:
            warmup += 1
        else:
            elapsed.append(seconds)
    row.update(raw_records=raw_count,completed_warmup_steps=warmup,completed_measured_steps=len(elapsed),
               raw_loss_trajectory=losses,raw_validation='passed' if not issues else 'failed')
    stated = worker.get('measured_seconds', [])
    if worker and (len(stated) != len(elapsed) or any(not near(a,b) for a,b in zip(stated, elapsed))):
        issues.append('reported measured samples differ from complete raw timings')
    if row['effective_status'] == 'completed' and (raw_count != 60 or warmup != 10 or len(elapsed) != 50):
        issues.append('completed result requires exactly 10 warmup and 50 complete measured steps')
    if len(elapsed) == 50 and row['effective_status'] == 'completed':
        row['seed_median_s'] = statistics.median(elapsed)
        if not near(worker.get('median_s'), row['seed_median_s']):
            issues.append('reported process median differs from raw complete samples')
    elif elapsed:
        row['partial_measured_median_diagnostic_only_s'] = statistics.median(elapsed)
    peak = threads = count = 0;last = -1.
    name = f'{row["view"]}-seed-{row["seed"]}'
    for sample in records(run/f'{name}-resources.jsonl', issues):
        count += 1
        rss = sample.get('aggregate_rss');values = sample.get('rss_by_pid', {})
        if not isinstance(rss, int) or rss < 0 or not all(isinstance(v,int) and v>=0 for v in values.values()) or sum(values.values()) != rss:
            issues.append('resource sample aggregate differs from contemporaneous process tree RSS')
            continue
        clock = sample.get('elapsed_s', -1)
        if not isinstance(clock,(int,float)) or not math.isfinite(clock) or clock < last:
            issues.append('resource sample time not monotonic/finite')
        last = clock;peak = max(peak,rss);threads = max(threads,sample.get('root_threads',0))
    row.update(resource_samples=count,root_thread_count_max=threads,sampled_job_peak_rss_bytes=peak if count else None)
    if external.get('command'):
        if external.get('peak_contemporaneous_job_rss') != peak or external.get('max_observed_root_threads') != threads:
            issues.append('supervisor RSS/thread peak differs from raw resource samples')


def validate(root, run, mode):
    errors = [];limitations = []
    out = dict(schema='e1-large-evidence-validation-r1',mode=mode,root=str(root),run=str(run),errors=errors,
               limitations=limitations,raw_verified=mode=='full',rows=[],views={},scope='E1-large 512-1024-1024-20 B32 T128 CPU FP64')
    if not (run/'freeze.json').is_file():
        out['status']='evidence_invalid';errors.append('missing freeze.json');return out
    freeze = read(run/'freeze.json');out['freeze_sha256']=sha(run/'freeze.json')
    rule = freeze.get('rule', {})
    if rule.get('sizes') != SIZES or rule.get('B') != 32 or rule.get('T') != 128 or rule.get('seeds') != list(SEEDS):
        errors.append('frozen full model/batch/time/seed denominator differs')
    if freeze.get('warmup_steps') != 10 or freeze.get('measured_steps') != 50 or freeze.get('per_seed_budget_s') != 360 or freeze.get('per_view_budget_s') != 1800:
        errors.append('frozen step counts or 360/1800 second budgets differ')
    if freeze.get('no_microbatch') is not True or freeze.get('no_batch_or_time_reduction') is not True:
        errors.append('frozen no-reduction policies absent')
    plan = freeze.get('order', []);planned = [(p['view'][0],p['seed']) for p in plan]
    expected = {(v,s) for v in VIEWS for s in SEEDS}
    if len(planned) != 35 or set(planned) != expected:
        errors.append('frozen order must contain exactly seven views times five seeds')
    terminal_path = run/'terminal.json'
    terminal = read(terminal_path) if terminal_path.is_file() else read(run/'progress.json') if (run/'progress.json').is_file() else {}
    out['supervisor_completed']=terminal.get('supervisor_completed',False)
    if not out['supervisor_completed']:
        limitations.append('Queue terminal absent/not completed; unresolved slots remain pending and cannot rank')
    ledger = {}
    for record in terminal.get('records', []):
        key = (record.get('view'),record.get('seed'))
        if key in ledger:errors.append(f'duplicate terminal slot {key}')
        ledger[key]=record
    manifest_path = root/'fixtures/e1-large/manifest.json'
    manifest = read(manifest_path) if mode=='full' and manifest_path.is_file() else None
    fixture_ref = read(run/'fixture-identity.json') if (run/'fixture-identity.json').is_file() else {}
    out['fixture_manifest_sha256']=fixture_ref.get('manifest_sha256')
    preparation = terminal.get('preparation', {})
    preparation_failed = bool(preparation) and (preparation.get('exit_code') != 0 or preparation.get('termination_reason')=='timeout')
    out['preparation']=preparation
    if mode=='full':
        for rel,h in freeze.get('scripts',{}).items():identity(root/rel,h,errors,'frozen source '+rel)
        identity(root/'runtime/b2-train',freeze.get('runtime_sha256'),errors,'frozen Atlas runtime')
        source_manifest = root/'sources/snapshot-manifest.json'
        if identity(source_manifest,freeze.get('source_manifest_sha256'),errors,'source snapshot manifest'):
            for rel,h in read(source_manifest).get('source_hashes',{}).items():identity(root/'snapshot'/rel,h,errors,'snapshot '+rel)
        for name,h in freeze.get('environment_locks',{}).items():identity(root/'environment'/name,h,errors,'environment lock '+name)
        identity(root/'environment/hardware.json',freeze.get('hardware_sha256'),errors,'hardware metadata')
        if not preparation_failed:
            if not fixture_ref.get('manifest_sha256') or manifest is None:
                errors.append('successful preparation requires full common-array manifest and queue identity')
            else:
                identity(manifest_path,fixture_ref['manifest_sha256'],errors,'common-array manifest')
                if manifest.get('rule') != rule or set(manifest.get('files',{})) != {f'seed-{s}.json' for s in SEEDS}:
                    errors.append('common-array manifest rule/seed denominator differs')
                if manifest.get('generator_sha256') != freeze.get('scripts',{}).get('tools/generate_dense_large.py'):
                    errors.append('common arrays came from another generator')
                for name,meta in manifest.get('files',{}).items():identity(manifest_path.parent/name,meta.get('sha256'),errors,'shared arrays '+name,meta.get('bytes'))
        for view,gate in freeze.get('gates',{}).items():
            if not gate.get('qualified'):continue
            path=Path(gate.get('path',''))
            if not identity(path,gate.get('report_sha256'),errors,view+' Q0 report'):continue
            q=read(path)
            if q.get('dense_qualification_status')!='passed' or q.get('engine')!=VIEWS[view][0]:errors.append(view+' Q0 engine/status differs')
            expected_q0={'base_negative_count_input','batch_duplicate','initial_threshold_boundary','single_sample_no_carry'}
            if set(q.get('cases',{}))!=expected_q0 or not q.get('identities') or any(c.get('passed') is not True or not passed(c.get('checks',{})) for c in q.get('cases',{}).values()):
                errors.append(view+' Q0 lacks all four passing declared cases and source identities')
            for rel,h in q.get('identities',{}).items():identity(root/rel,h,errors,view+' Q0 source '+rel)
            for name in q.get('cases',{}):
                qraw_path=path.parent/(name+'.json')
                if not qraw_path.is_file():errors.append(view+' Q0 raw missing: '+name);continue
                raw=read(qraw_path).get('raw',{})
                if view=='atlas':
                    if raw.get('runner_sha256')!=freeze.get('runtime_sha256') or raw.get('training_api_sha256')!=sha(root/'snapshot/brian2-rust/python/brian2_rust/training.py'):
                        errors.append('Atlas Q0 runtime/API differs from frozen large run')
                elif view not in JAX and (raw.get('engine')!=VIEWS[view][0] or raw.get('compile') is not VIEWS[view][1]):
                    errors.append(view+' Q0 raw engine/compile differs')
    spent={v:0. for v in VIEWS}
    for view,seed in planned:
        folder=run/f'{view}-seed-{seed}';external=ledger.get((view,seed),{})
        worker=read(folder/'result.json') if (folder/'result.json').is_file() else {}
        status='timeout' if external.get('termination_reason')=='timeout' else external.get('status',worker.get('status','pending'))
        row=dict(view=view,seed=seed,effective_status=status,worker_status=worker.get('status'),errors=[],
                 termination_reason=external.get('termination_reason'),exit_code=external.get('exit_code'),
                 reason=external.get('reason',worker.get('error')),budget_s=external.get('timeout_s'),
                 supervisor_wall_s=external.get('elapsed_s'),worker_wall_s=worker.get('wall_s'),
                 array_sha256=worker.get('arrays_sha256'),identities=worker.get('identities',{}),
                 numerical_qualification=False,resource_qualification=view not in JAX,strict_ranking_eligible=False,
                 resource_class='requested1thread_no_core_affinity' if view not in JAX else 'thread-budget-unqualified host-config diagnostics',
                 raw_validation='not_requested' if mode=='report-only' else 'pending',
                 rss_scope='50ms sampled complete subprocess-tree job RSS; includes qualification/oracle/compile; not pure training memory or capacity',
                 cold={'constructor_s':worker.get('constructor_ns',0)/1e9 if view=='atlas' else None,
                       'setup_and_first_update_s':worker.get('cold_construction_and_first_update_s'),
                       'scope':'Atlas constructor only; first qualifying call separately recorded' if view=='atlas' else 'setup plus first actual qualification update; excludes earlier interpreter/import/fixture I/O'})
        out['rows'].append(row);issues=row['errors'];remaining=1800-spent[view]
        if external.get('command'):
            command=external['command']
            def arg(flag):
                return command[command.index(flag)+1] if flag in command and command.index(flag)+1<len(command) else None
            engine,compiled,layerwise=VIEWS[view]
            if arg('--engine')!=engine or arg('--seed')!=str(seed) or ('--compile' in command)!=compiled or ('--layerwise' in command)!=layerwise:
                issues.append('launched command differs from requested engine/seed/compile/layerwise')
            if not near(external.get('timeout_s'),min(360,remaining)) or remaining<=0:issues.append('launched seed cap does not equal min(360, remaining per-view1800)')
            duration=external.get('elapsed_s')
            if not positive(duration):issues.append('launched process has invalid supervisor wall duration')
            else:spent[view]+=duration
            if external.get('termination_reason')=='timeout' and positive(duration) and duration+0.1<external.get('timeout_s',0):issues.append('timeout recorded before requested cap')
            if mode=='full':
                perjob=run/f'{view}-seed-{seed}-terminal.json'
                if not perjob.is_file():issues.append('launched process lacks per-job terminal')
                else:
                    for k,value in read(perjob).items():
                        if external.get(k)!=value:issues.append('per-job terminal differs: '+k)
        elif status=='timeout' and not worker and remaining>0:
            issues.append('prelaunch aggregate timeout while positive per-view budget remains')
        if worker:
            engine,compiled,layerwise=VIEWS[view]
            if (worker.get('engine'),worker.get('seed'),worker.get('compile'),worker.get('layerwise'))!=(engine,seed,compiled,layerwise):issues.append('worker view/seed/compile/layerwise identity differs')
            if worker.get('rule')!=rule or worker.get('no_microbatch') is not True:issues.append('worker workload rule/full batch differs')
            if worker.get('identities_unchanged') is not True:issues.append('worker implementation changed during execution')
            if external.get('termination_reason')!='timeout' and external.get('status') not in (None,worker.get('status')):issues.append('worker and terminal scientific statuses disagree')
            for rel,h in worker.get('identities',{}).items():
                if freeze.get('scripts',{}).get(rel)!=h:issues.append('worker source identity differs: '+rel)
            if worker.get('fixture_manifest_sha256') is not None and worker.get('fixture_manifest_sha256')!=fixture_ref.get('manifest_sha256'):issues.append('worker fixture manifest differs')
            if manifest and worker.get('arrays_sha256') is not None and worker.get('arrays_sha256')!=manifest.get('files',{}).get(f'seed-{seed}.json',{}).get('sha256'):issues.append('worker array digest differs from common seed')
            if view=='atlas' and worker.get('runtime_sha256') not in (None,freeze.get('runtime_sha256')):issues.append('worker Atlas runtime differs')
            if external.get('result_sha256') and sha(folder/'result.json')!=external['result_sha256']:issues.append('terminal result hash differs')
            worker_metadata(folder,worker,row,mode)
        elif mode=='full':
            worker_metadata(folder,worker,row,mode)
        if status=='completed':
            if not worker or worker.get('performance_run') is not True or not row['numerical_qualification'] or external.get('exit_code')!=0:issues.append('completed slot lacks full qualification/performance result or exit0')
            if worker.get('warmup_steps')!=10 or worker.get('measured_steps')!=50 or len(worker.get('measured_seconds',[]))!=50:issues.append('completed worker does not declare 10warmup/50measured with50samples')
            if not worker.get('arrays_sha256') or not worker.get('fixture_manifest_sha256') or not worker.get('identities'):issues.append('completed worker lacks source/common-array identity')
            if view=='atlas' and worker.get('runtime_sha256')!=freeze.get('runtime_sha256'):issues.append('completed Atlas worker lacks matching runtime identity')
        if mode=='full':raw_and_resources(run,folder,external,worker,row)
        else:
            row['completed_measured_steps']=len(worker.get('measured_seconds',[]));row['completed_warmup_steps']=None
            if status=='completed' and len(worker.get('measured_seconds',[]))==50 and all(positive(v) for v in worker['measured_seconds']):row['seed_median_s']=statistics.median(worker['measured_seconds'])
        if status=='completed' and not freeze.get('gates',{}).get(view,{}).get('qualified'):issues.append('completed worker lacks matching Q0 gate')
        for issue in issues:errors.append(f'{view}/seed-{seed}: {issue}')
    if out['supervisor_completed']:
        if set(ledger)!=expected:errors.append('terminal must preserve all35 formal slots')
        for view in VIEWS:
            if not near(terminal.get('spent_seconds',{}).get(view),spent[view]):errors.append(view+' aggregate spent time differs from launched records')
    out['recomputed_spent_seconds']=spent
    for seed in SEEDS:
        hashes={r['array_sha256'] for r in out['rows'] if r['seed']==seed and r.get('array_sha256')}
        if len(hashes)>1:errors.append(f'seed-{seed}: views used different common arrays')
    out['denominator']=dict(views=list(VIEWS),seeds=list(SEEDS),expected_slots=35,scheduled_slots=len(planned),expected_warmup=10,expected_measured=50)
    eligible_global=mode=='full' and not errors and out['supervisor_completed']
    by_key={(r['view'],r['seed']):r for r in out['rows']}
    for row in out['rows']:
        row['strict_ranking_eligible']=bool(eligible_global and row['effective_status']=='completed' and row['numerical_qualification'] and row['resource_qualification'] and not row['errors'])
    for view in VIEWS:
        selected=[by_key[(view,s)] for s in SEEDS if (view,s) in by_key]
        complete=[r for r in selected if r['effective_status']=='completed' and r['numerical_qualification'] and not r['errors'] and positive(r.get('seed_median_s'))]
        summary=dict(scheduled_n=len(selected),completed_n=len(complete),complete_five_seed_summary=len(complete)==5,
                     statuses={str(r['seed']):r['effective_status'] for r in selected},resource_qualification=view not in JAX,
                     strict_ranking_eligible=eligible_global and len(complete)==5 and view not in JAX,
                     ratios_validated=eligible_global and view not in JAX,paired_seed_ratios=[])
        if len(complete)==5:
            values=[r['seed_median_s'] for r in complete]
            summary.update(median_of_process_medians_s=statistics.median(values),min_process_median_s=min(values),max_process_median_s=max(values))
        for seed in SEEDS:
            row=by_key.get((view,seed),{});atlas=by_key.get(('atlas',seed),{})
            eligible=row.get('strict_ranking_eligible') is True and atlas.get('strict_ranking_eligible') is True
            summary['paired_seed_ratios'].append(dict(seed=seed,eligible=eligible,ratio_atlas_over_view=atlas['seed_median_s']/row['seed_median_s'] if eligible else None))
        out['views'][view]=summary
    out['all_declared_cases_completed']=len(out['rows'])==35 and all(r['effective_status']=='completed' and not r['errors'] for r in out['rows'])
    out['all_declared_resource_gates_passed']=all(r['resource_qualification'] for r in out['rows'])
    limitations.extend(['No engine/oracle rerun; NPZ artifacts and common arrays are streamed for integrity, not numerical recomputation.',
        'Recorded exact JSON byte count/hash is preserved and cap arithmetic checked; full request payload was not archived, so this validator does not independently recreate current-state JSON bytes.',
        'A partial measured median is diagnostic only. Fewer than50 complete measured steps or fewer than5 complete independent seeds cannot enter verified five-process ranking.',
        'JAX resource qualification remains false; no strict Atlas/JAX ratio. Requested1thread is not same-core affinity.',
        'Whole-job sampled RSS includes oracle/qualification/compiler/instrumentation; not pure training RSS or capacity.',
        'Cold fields have different boundaries and omit full process cold startup; no cold-speed ranking.',
        'Per-process cap is360s and per-view aggregate1800s; supervisor polling/cleanup wall overhead is recorded, not extra training budget.',
        'Cleanup fields record attempted cleanup of owned descendants; this validator does not prove absence of all transient host contention.',
        'Queue/worker evidence absence under timeout is retained as censorship, never silently replaced with a reduced batch or sequence.'])
    if mode!='full':limitations.append('Report-only: raw timings, NPZ/common-array bytes, runtime/source contents and resources remain unverified; all strict eligibility is false.')
    out['status']='evidence_invalid' if errors else 'passed_full_evidence_checks' if mode=='full' else 'passed_report_checks_raw_unverified'
    return out


def report(value):
    lines=['# E1-large 证据核验', '', value['scope'], '', '状态：'+value['status'], '']
    if value['errors']:
        lines += ['证据存在不一致，拒绝生成已验证性能排名。', ''] + ['- '+e for e in value['errors']]
    else:
        lines += ['每个视图保留五个正式 seed。只有完整 10 warmup + 50 测量、完整数值资格及资源资格的结果可以进入严格统计；拒绝/超时不是速度优势。', '',
                  '完整性核验通过不代表所有科学任务成功，也不代表所有资源门槛通过。', '']
        for title,names in [('请求单线程组（无同核 affinity）',[v for v in VIEWS if v not in JAX]),('JAX 主机配置观测（不满足单线程门槛）',JAX)]:
            lines += ['## '+title,'','| 视图 | 完成/5 | 五进程中位数 ms | min ms | max ms | 严格资格 |','| --- | --- | --- | --- | --- | --- |']
            for view in names:
                item=value['views'][view]
                fields=[view,str(item['completed_n'])+'/5']+[format(item[k]*1000,'.8g') if k in item else '—' for k in ['median_of_process_medians_s','min_process_median_s','max_process_median_s']]+[str(item['strict_ranking_eligible'])]
                lines.append('| '+' | '.join(fields)+' |')
    lines += ['', '## 全部 slot 与准入/预算结果', '', '| 视图 | seed | 状态 | 准入阶段 | 资格步数 | warmup/测量 | 严格资格 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for r in value.get('rows',[]):
        fields=[r['view'],str(r['seed']),r['effective_status'],r.get('admission_outcome','not_requested'),str(r.get('completed_qualification_updates',0)),f'{r.get("completed_warmup_steps")}/{r.get("completed_measured_steps")}',str(r['strict_ranking_eligible'])]
        lines.append('| '+' | '.join(fields)+' |')
    lines += ['', '## 解释边界', '', 'JAX 只属于主机配置观测，资源资格始终为 false；没有 strict 比值。初次请求拒绝与 Adam 更新后完整精度 optimizer-state JSON 超限分开保留；二者均不称为 OOM。冷启动与整作业采样 RSS 的范围不同于纯训练成本。完整 JSON 包含全部五个 seed 的成对比值（无资格时 null）、每步准入元数据和各种未完成状态。', '']
    lines += ['- '+text for text in value['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mode',choices=['full','report-only'],default='report-only')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path)
    args=p.parse_args()
    try:value=validate(args.root.resolve(),args.run.resolve(),args.mode)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
        value=dict(schema='e1-large-evidence-validation-r1',status='evidence_invalid',mode=args.mode,raw_verified=False,
                   root=str(args.root.resolve()),run=str(args.run.resolve()),scope='E1-large 512-1024-1024-20 B32 T128 CPU FP64',
                   rows=[],views={},errors=[f'Cannot complete evidence parsing: {type(error).__name__}: {error}'],limitations=['Parsing failure; no verified ranking.'])
    value['validator_sha256']=sha(__file__)
    with args.output.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    if args.report:
        with args.report.open('x') as stream:stream.write(report(value))
    print(json.dumps({'status':value['status'],'errors':len(value['errors']),'complete':value.get('all_declared_cases_completed'),
                      'rows':len(value.get('rows',[])),'output':str(args.output)},ensure_ascii=False))
    return 2 if value['errors'] else 0


if __name__=='__main__':raise SystemExit(main())
