#!/usr/bin/env python3
"""Read-only, stdlib E1 evidence validation; never runs an engine or an oracle.

report-only: small order/freeze/terminal/worker reports; raw remains unverified.
full: run on the remote evidence root; stream complete raw step records and
verify available frozen executables/sources. No raw data are copied anywhere.
Optional --output is created exclusively; existing reports are never replaced.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

SEEDS = [11, 23, 37, 51, 71]
VIEWS = ['atlas','sj-layerwise','snn-layerwise','spyx','brainstate','sj-compile','snn-compile']


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def all_checks_pass(checks):
    return bool(checks) and all(isinstance(v, dict) and v.get('passed') is True for v in checks.values())


def near(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered)-1)*fraction
    lower = int(math.floor(index)); upper = int(math.ceil(index))
    return ordered[lower] + (ordered[upper]-ordered[lower])*(index-lower)


def raw_steps(path, atlas):
    opener = gzip.open if path.suffix == '.gz' else open
    indices, measured, warmup, finite_losses, losses = [], [], 0, True, []
    with opener(path, 'rt') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            index = row['index']
            expected_phase = 'warmup' if index < 10 else 'measured'
            if row.get('phase') != expected_phase:
                raise ValueError(f'incorrect phase for index {index}')
            indices.append(index)
            if atlas:
                if row.get('status') != 'executed':
                    continue
                seconds = row['public_api_ns']/1e9
                loss = row['result']['loss']
                if row.get('tape_count_matches_runtime') is not True:
                    raise ValueError(f'Atlas tape count mismatch at {index}')
            else:
                seconds, loss = row['elapsed_s'], row['loss']
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f'invalid latency at {index}')
            finite_losses = finite_losses and math.isfinite(float(loss))
            losses.append(float(loss))
            if index < 10:
                warmup += 1
            else:
                measured.append(seconds)
    return dict(indices=indices, warmup_steps=warmup, measured_s=measured,
                finite_losses=finite_losses, losses=losses)


def validate(root, run, mode):
    errors, limitations = [], []
    result = dict(schema='e1-evidence-validation-r1', mode=mode, root=str(root), run=str(run),
                  validation_scope='evidence identity/count/timing/statistics; numerical oracle is not rerun',
                  raw_verified=mode == 'full', errors=errors, limitations=limitations, rows=[])
    for filename in ['order.json', 'freeze.json']:
        if not (run/filename).is_file():
            errors.append(f'missing required {filename}')
    if errors:
        result['status'] = 'evidence_invalid'
        return result
    order, freeze = read(run/'order.json'), read(run/'freeze.json')
    result['freeze_sha256'] = sha256(run/'freeze.json')
    declared_views = [item[0] for item in freeze['views']]
    if set(declared_views) != set(VIEWS):
        errors.append('this frozen E1 revision requires all seven views; no favorable subset may replace the denominator')
    expected = {(name, seed) for name in declared_views for seed in SEEDS}
    scheduled = [(view[0], seed) for seed, view in order]
    if len(scheduled) != len(set(scheduled)) or set(scheduled) != expected:
        errors.append('order must contain each declared view and each of five seeds exactly once')
    if len(declared_views) != len(set(declared_views)):
        errors.append('duplicate declared views')
    result['denominator'] = dict(views=declared_views, seeds=SEEDS, scheduled_slots=len(scheduled),
                                 expected_slots=len(expected), expected_warmup=10, expected_measured=50)
    terminal_path = run/'terminal.json'
    if terminal_path.exists():
        terminal = read(terminal_path)
        result['supervisor_completed'] = terminal.get('supervisor_completed',False)
        records = terminal.get('records',[])
    else:
        result['supervisor_completed'] = False
        records = read(run/'progress.json') if (run/'progress.json').exists() else []
        limitations.append('supervisor terminal.json absent; run may still be in progress')
    ledger = {}
    for record in records:
        if 'seed' in record:
            key = (record['name'], record['seed'])
        else:
            name, _, seed = record['name'].rpartition('-seed-')
            if not seed.isdigit():
                errors.append(f'unrecognized terminal record name: {record["name"]}')
                continue
            key = (name,int(seed))
        if key in ledger:
            errors.append(f'duplicate terminal record: {key}')
        ledger[key] = record
    for name, seed in scheduled:
        stem = f'{name}-seed-{seed}'
        worker_path = run/stem/'result.json'
        external = ledger.get((name,seed),{})
        # Per-job terminal survives before final progress/terminal flush.
        if not external and (run/(stem+'-terminal.json')).exists():
            external = read(run/(stem+'-terminal.json'))
        worker = read(worker_path) if worker_path.exists() else {}
        row = dict(view=name,seed=seed,scheduled=True,worker_result_present=bool(worker),
                   worker_status=worker.get('status'),termination_reason=external.get('termination_reason'),
                   exit_code=external.get('exit_code'),qualification_status=worker.get('qualification_status'),
                   raw_validation='not_requested' if mode=='report-only' else 'pending',errors=[])
        result['rows'].append(row)
        issues = row['errors']
        if external.get('termination_reason') == 'timeout':
            status = 'timeout'
        elif external.get('execution_status'):
            status = external['execution_status']
        elif worker:
            status = worker.get('status','missing_worker_status')
        elif external:
            status = 'worker_result_missing'
        else:
            status = 'pending'
        row['effective_status'] = status
        row['reason'] = external.get('reason') or worker.get('reason') or worker.get('error')
        row['worker_wall_s'] = worker.get('process_wall_seconds',worker.get('wall_s'))
        row['supervisor_wall_s'] = external.get('elapsed_s')
        row['budget_s'] = external.get('timeout_s')
        row['sampled_job_peak_rss_bytes'] = external.get('peak_job_rss_same_sample')
        row['rss_scope'] = '50ms sampled process-tree RSS across import/oracle/JIT/qualification/training; shared pages may be counted multiple times'
        row['resource_class'] = 'Torch/Atlas requested single compute thread' if name not in ('spyx','brainstate') else 'JAX worker-pool size/use not constrained or proved single-thread'
        row['resource_qualification'] = name not in ('spyx','brainstate')
        row['resource_qualification_reason'] = ('r2 requested1thread setting present; macOS core affinity not set'
            if row['resource_qualification'] else 'thread-budget-unqualified host-config diagnostics; r2 initial1thread requirement not met/proved')
        row['strict_ranking_eligible'] = row['resource_qualification'] and status == 'completed'
        if row['budget_s'] is not None and not 0 < row['budget_s'] <= 360:
            issues.append('per-seed budget outside (0,360] seconds')
        atlas = name == 'atlas'
        required_q = 'qualified' if atlas else 'passed_E1_dense'
        recorded_checks = worker.get('qualification_checks') if atlas else worker.get('checks')
        row['numerical_qualification'] = worker.get('qualification_status') == required_q and all_checks_pass(recorded_checks)
        times = ([v/1e9 for v in worker.get('measured_public_api_ns',[])] if atlas
                 else worker.get('measured_seconds',[]))
        row['completed_measured_steps'] = len(times)
        row['completed_warmup_steps'] = None
        if any(not isinstance(t,(int,float)) or not math.isfinite(t) or t <= 0 for t in times):
            issues.append('invalid reported measured latency')
        if times and not issues:
            median = statistics.median(times)
            row.update(seed_median_s=median,step_p10_s=percentile(times,.1),step_p90_s=percentile(times,.9))
            stated = worker.get('median_public_api_ns')
            stated = stated/1e9 if stated is not None else worker.get('median_s')
            if stated is not None and not near(stated,median):
                issues.append('reported median differs from reported samples')
        if status == 'completed':
            if len(times) != 50 or worker.get('warmup_steps') != 10 or worker.get('measured_steps') != 50:
                issues.append('completed worker must declare10 warmup and have50 measured steps')
            if external.get('exit_code') not in (None,0):
                issues.append('completed worker has nonzero exit code')
            if not row['numerical_qualification']:
                issues.append('completed worker lacks passing actual-case qualification')
            if not atlas and worker.get('performance_run') is not True:
                issues.append('completed competitor is not marked a performance run')
        row['identities'] = worker.get('identities',{})
        for rel,h in worker.get('identities',{}).items():
            if rel in freeze.get('scripts',{}) and freeze['scripts'][rel] != h:
                issues.append(f'worker identity differs from freeze: {rel}')
        if atlas and worker:
            mappings = [('benchmark_sha256','tools/benchmark_e1.py'),('adapter_sha256','adapters/atlas_adapter.py'),
                        ('oracle_sha256','adapters/oracle.py')]
            for field,rel in mappings:
                if worker.get(field) != freeze.get('scripts',{}).get(rel):
                    issues.append(f'worker identity differs from freeze: {rel}')
            if worker.get('runtime_sha256') != freeze.get('runtime_sha256'):
                issues.append('Atlas runtime differs from freeze')
            if worker.get('freeze_sha256') != result['freeze_sha256']:
                issues.append('Atlas worker points to another freeze')
        row['array_sha256'] = worker.get('case_sha256',worker.get('arrays_sha256'))
        row['cold'] = ({'first_constructor_s':worker.get('first_constructor_ns',0)/1e9,
                        'first_public_api_s':worker.get('first_public_api_ns',0)/1e9,
                        'scope':'Atlas constructor and public API separately; excludes process start/oracle/fixture'} if atlas else
                       {'setup_and_first_update_s':worker.get('cold_construction_and_first_update_s'),
                        'scope':'post-oracle setup plus first actual update; excludes earlier imports/process start/fixture'})
        if mode == 'full':
            raw = run/stem/('raw-steps.jsonl.gz' if atlas else 'raw-steps.jsonl')
            if raw.exists():
                try:
                    detail = raw_steps(raw,atlas)
                    row['completed_warmup_steps'] = detail['warmup_steps']
                    row['raw_loss_trajectory'] = detail['losses']
                    row['raw_validation'] = 'passed'
                    if detail['indices'] != list(range(len(detail['indices']))):
                        issues.append('raw step indices are not consecutive from zero')
                    if status == 'completed':
                        if detail['indices'] != list(range(60)) or detail['warmup_steps'] != 10:
                            issues.append('completed worker raw does not contain exactly10 warmup plus50 measured')
                        if not detail['finite_losses']:
                            issues.append('completed worker raw contains nonfinite loss')
                        if len(times)!=len(detail['measured_s']) or any(not near(a,b) for a,b in zip(times,detail['measured_s'])):
                            issues.append('raw measured times differ from worker report')
                except Exception as error:
                    row['raw_validation']='unreadable_or_truncated'
                    message=f'{stem}: raw read {type(error).__name__}: {error}'
                    (issues if status=='completed' else limitations).append(message)
            elif status == 'completed':
                issues.append('completed worker lacks full raw steps')
                row['raw_validation']='missing'
            else:
                row['raw_validation']='not_available_for_incomplete_case'
            resource = run/(stem+'-resource.json')
            if resource.exists():
                samples = read(resource).get('samples',[])
                row['root_thread_count_max'] = max((s.get('root_threads',0) for s in samples),default=None)
                row['resource_samples'] = len(samples)
        if issues:
            row['strict_ranking_eligible'] = False
            errors.extend(f'{stem}: {message}' for message in issues)
    # Paired fixture identity must be equal across every view for each seed.
    for seed in SEEDS:
        hashes = {row['array_sha256'] for row in result['rows'] if row['seed']==seed and row.get('array_sha256')}
        if len(hashes)>1:
            errors.append(f'views used different common arrays for seed{seed}')
    # Additional post-run consistency audit, distinct from the predeclared
    # three-update numerical qualification. It never reruns or replaces a seed.
    result['paired_loss_trajectory_audit'] = []
    if mode == 'full':
        for seed in SEEDS:
            reference = next((row for row in result['rows'] if row['view']=='atlas' and row['seed']==seed), {})
            expected_losses = reference.get('raw_loss_trajectory', [])
            for row in result['rows']:
                if row['seed'] != seed or row['view'] == 'atlas':
                    continue
                actual_losses = row.get('raw_loss_trajectory', [])
                comparable = len(expected_losses)==60 and len(actual_losses)==60
                mismatches = ([i for i,(a,b) in enumerate(zip(actual_losses,expected_losses))
                               if not math.isfinite(a) or abs(a-b)>1e-10+1e-8*abs(b)] if comparable else [])
                result['paired_loss_trajectory_audit'].append(dict(seed=seed,view=row['view'],
                    compared_steps=60 if comparable else 0, mismatching_step_indices=mismatches,
                    status=('different' if mismatches else 'consistent') if comparable else 'not_comparable',
                    scope='Post-run same-fixture CE trajectory audit; this does not establish every hidden state or parameter is identical.'))
                if mismatches:
                    issue='post-run paired CE trajectory differs from Atlas beyond frozen FP64 tolerance'
                    row['errors'].append(issue)
                    row['strict_ranking_eligible']=False
                    errors.append(f'{row["view"]}-seed-{seed}: {issue}')
    result['views'] = {}
    atlas_by_seed = {row['seed']:row for row in result['rows'] if row['view']=='atlas'}
    for name in declared_views:
        rows = [row for row in result['rows'] if row['view']==name]
        complete = [row for row in rows if row['effective_status']=='completed' and not row['errors']]
        summary = dict(scheduled_n=len(rows),completed_n=len(complete),independent_n=len(complete),
                       statuses={str(row['seed']):row['effective_status'] for row in rows},
                       supervisor_wall_s=sum(row.get('supervisor_wall_s') or 0 for row in rows),
                       complete_five_seed_summary=len(complete)==5)
        summary['resource_qualification'] = name not in ('spyx','brainstate')
        summary['numerically_qualified_n'] = sum(row['numerical_qualification'] for row in rows)
        summary['strict_ranking_eligible'] = len(complete)==5 and summary['resource_qualification']
        summary['timing_class'] = ('requested1thread_no_core_affinity' if summary['resource_qualification']
            else 'thread-budget-unqualified host-config diagnostics; numerical qualification remains separately valid')
        if len(complete)==5:
            values=[row['seed_median_s'] for row in complete]
            summary.update(median_of_process_medians_s=statistics.median(values),min_process_median_s=min(values),max_process_median_s=max(values))
        pairs=[]
        for row in rows:
            ref=atlas_by_seed.get(row['seed'],{})
            eligible=(row['effective_status']=='completed' and not row['errors'] and
                      ref.get('effective_status')=='completed' and not ref.get('errors') and
                      row['resource_qualification'] and ref.get('resource_qualification'))
            pairs.append(dict(seed=row['seed'],eligible=eligible,
                              ratio_atlas_over_view=ref['seed_median_s']/row['seed_median_s'] if eligible else None))
        summary['paired_seed_ratios']=pairs
        summary['paired_complete_n']=sum(pair['eligible'] for pair in pairs)
        summary['ratio_scope']='strict ratios only within numerically qualified requested1thread group; macOS no core affinity; JAX strict ratios prohibited'
        result['views'][name]=summary
    if mode=='full':
        for rel,h in freeze.get('scripts',{}).items():
            if not (root/rel).is_file() or sha256(root/rel)!=h:
                errors.append(f'live remote source missing/differs from benchmark freeze: {rel}')
        runtime=root/'runtime/b2-train'
        if not runtime.is_file() or sha256(runtime)!=freeze.get('runtime_sha256'):
            errors.append('live remote Atlas runtime missing/differs from freeze')
        manifest_path=root/'sources/snapshot-manifest.json'
        if not manifest_path.exists() or sha256(manifest_path)!=freeze.get('source_manifest'):
            errors.append('source snapshot manifest missing/differs from freeze')
        else:
            for rel,h in read(manifest_path).get('source_hashes',{}).items():
                source=root/'snapshot'/rel
                if not source.is_file() or sha256(source)!=h:
                    errors.append(f'frozen snapshot source missing/changed: {rel}')
        qfolder=run.parent/'remote-qualify-v1/q0-atlas'
        qreport=qfolder/'report.json'
        if any(row['view']=='atlas' and row['effective_status']=='completed' for row in result['rows']):
            if not qreport.is_file():
                errors.append('full mode requires original Atlas Q0 evidence to bind runtime/API identity')
            else:
                api=root/'snapshot/brian2-rust/python/brian2_rust/training.py'
                api_hash=sha256(api) if api.is_file() else None
                for case in read(qreport).get('cases',{}):
                    path=qfolder/(case+'.json')
                    if not path.is_file():
                        errors.append(f'Atlas Q0 raw evidence missing: {case}')
                        continue
                    raw=read(path).get('raw',{})
                    if raw.get('runner_sha256')!=freeze.get('runtime_sha256') or raw.get('training_api_sha256')!=api_hash:
                        errors.append(f'Atlas Q0 runtime/API identity differs from benchmark: {case}')
    else:
        limitations.append('Report-only mode: no raw steps, runtime binaries, source files or sample resources verified')
    limitations.extend([
        'Numerical oracle is not rerun; validator checks recorded qualification outcomes and identities only',
        'Cold fields have different boundaries; empty compilation cache was not established by this validator',
        'RSS includes setup/qualification/instrumentation and is not pure training capacity',
        'JAX pool utilization is not constrained or verified as single-thread'])
    if errors:
        for row in result['rows']:
            row['strict_ranking_eligible'] = False
        for summary in result.get('views',{}).values():
            summary['strict_ranking_eligible'] = False
            summary['ratios_validated'] = False
    else:
        for summary in result.get('views',{}).values():
            summary['ratios_validated'] = mode=='full' and summary['resource_qualification']
    result['status']='evidence_invalid' if errors else ('passed_full_evidence_checks' if mode=='full' else 'passed_report_checks_raw_unverified')
    result['all_declared_cases_completed']=all(row['effective_status']=='completed' and not row['errors'] for row in result['rows'])
    result['all_declared_resource_gates_passed']=all(row['resource_qualification'] for row in result['rows'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,default=Path('evidence/remote-benchmark-v1'))
    parser.add_argument('--mode',choices=['report-only','full'],default='report-only')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    run=args.run.resolve() if args.run.is_absolute() else root/args.run
    result=validate(root,run,args.mode)
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as out:out.write(text)
        print(json.dumps(dict(status=result['status'],errors=len(result['errors']),output=str(args.output))))
    else:
        print(text,end='')
    return 1 if result['errors'] else 0


if __name__=='__main__':
    raise SystemExit(main())
