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


def _validate_original(root, run, mode):
    errors, limitations = [], []
    result = dict(schema='e1-small-arm64-evidence-validation-r1', mode=mode, root=str(root), run=str(run),
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
        row['sampled_job_peak_rss_bytes'] = external.get('peak_contemporaneous_rss',external.get('peak_job_rss_same_sample'))
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
        runtime=root/'runtime/arm64-r1/b2-train'
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
        qfolder=root/'evidence/arm64-r1/q0/atlas'
        qreport=qfolder/'report.json'
        if any(row['view']=='atlas' and row['effective_status']=='completed' for row in result['rows']):
            if not qreport.is_file():
                errors.append('full mode requires the new ARM64 Atlas Q0 evidence to bind runtime/API identity')
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


ARM64_RUNTIME='runtime/arm64-r1/b2-train'
ARM64_Q0='evidence/arm64-r1/q0/atlas/report.json'
PRODUCER='evidence/runtime-arm64-build-r1/terminal.json'
BUILD_CONTRACT='evidence/runtime-arm64-preparation-r1/build-contract.json'
DISPATCHER='tools/run_e1_small_arm64_r1.py'
FIXED_PREEXECUTION={'adapters/atlas_adapter.py': '89cdb5ac9db93e0d9110cac1782dc4720c8698b460ddd2c5cd7aa9290ee6c7ed', 'adapters/jax_adapter.py': '286cfa6df151d8ae3d05a937c0e9ca4545736db2434308da7c0ed02378624831', 'adapters/oracle.py': '80dd94c5be77bdf1857c2f22a2c7853eec871f47a1bee7caab2f946311d72bdb', 'adapters/torch_adapter.py': '8c49b9718578885aa14b3a539a665a3bf00d3555bb30b9d6771a3d38494b6f48', 'environment/cpu-lock.txt': 'd0b31ade6227e70132c886c420dd6a355378b5fc81588bdf34828de0ede1fce4', 'environment/hardware.json': 'f95a16795042edbb9b9f29f857875d6452bb0a42492b0834152ba1ba34e58018', 'environment/jax-lock.txt': '52eb289987704af80b6e59f96b5a91884e48ec71a68d8cfc9c8b68632470e9c3', 'evidence/remote-qualify-v1/freeze.json': 'a50a9abb7f9697b050c2e10ce30e77162e374a89b892f4db69f1aff162acec53', 'evidence/remote-qualify-v1/q0-brainstate-terminal.json': '4781ddcba290ca6305719cd554e27947b6a09a7c0b12b11c3e857726f1ad5d9d', 'evidence/remote-qualify-v1/q0-brainstate/base_negative_count_input.json': '98c89d0eae8949ba5f3c6d7760509ae9bdac6bf7a405495c0a5ebe1f7d071baa', 'evidence/remote-qualify-v1/q0-brainstate/batch_duplicate.json': '483340057621ff00585a94fad6287919e72cb35a18462450bca1aad131d42605', 'evidence/remote-qualify-v1/q0-brainstate/initial_threshold_boundary.json': '94715b02ff853a0a583b15c0e290580a852b685ff2b038702adacb95878d1672', 'evidence/remote-qualify-v1/q0-brainstate/report.json': '01267a7353a26a7087e49c16dab154c5195a7ebfc95c3300411d768af6ad5664', 'evidence/remote-qualify-v1/q0-brainstate/single_sample_no_carry.json': '723b7dcd9d0b5f6b9ff7921ea5c44f86c2ae29367b9893b04c5b631208853ca5', 'evidence/remote-qualify-v1/q0-sj-compile-terminal.json': '4310441a6d29d4a4c831d5e7d99a6acb25a9c778b5237431c74c0a0d39cfdb55', 'evidence/remote-qualify-v1/q0-sj-compile/base_negative_count_input.json': '09a13f727b5c76463b0fa10ad2cc797911ff45d4c2a3cc4493c0994196e8b7bb', 'evidence/remote-qualify-v1/q0-sj-compile/batch_duplicate.json': '401d398218fa65e60793c4bfc4b38b68b27dc511405f8d1cd223a30f10fd103d', 'evidence/remote-qualify-v1/q0-sj-compile/initial_threshold_boundary.json': '0dcbc4d9ccc8e31682bfedbd00254daef9668afcfa963e44851fe1b0bd25f1b4', 'evidence/remote-qualify-v1/q0-sj-compile/report.json': '51c70866b6055e9793b95625e9fb2afbf13ed4c30aea026ec081c5fb87e7d57b', 'evidence/remote-qualify-v1/q0-sj-compile/single_sample_no_carry.json': '9724f58dc0cc04d4944e9803a84001990572f4b3dba5f32ed92806b77efeb544', 'evidence/remote-qualify-v1/q0-sj-layerwise-terminal.json': 'aea06d9bd1dae3cc6e456f7e4ad49dee63e27ff457329030f212450671ef92b8', 'evidence/remote-qualify-v1/q0-sj-layerwise/base_negative_count_input.json': '12581d29bc9b911e8e7f4b7e80440043bda24fca3daad5b2b1657431283bb5dd', 'evidence/remote-qualify-v1/q0-sj-layerwise/batch_duplicate.json': '21fd01bc7b8779989a3bcd1520b2d8afa93c301377a2b2a5a36e3fff173e12b5', 'evidence/remote-qualify-v1/q0-sj-layerwise/initial_threshold_boundary.json': '038cfa0b87e24666b0d4610944167060a32db69eb7b6dd5849aeaebd19f46a83', 'evidence/remote-qualify-v1/q0-sj-layerwise/report.json': '27d572b5ecc148d39403d50da7b9e8c9abe4b0bb18ce8b0c790a9546ebd7397b', 'evidence/remote-qualify-v1/q0-sj-layerwise/single_sample_no_carry.json': '525a96e0093e9654a260465dfe8143dfd76e3605c396de7e28ef5735a9914d27', 'evidence/remote-qualify-v1/q0-snn-compile-terminal.json': '8c70ec22bff1710c76770769c3eadbaef86c9a565a14187157577b26cc6dfeab', 'evidence/remote-qualify-v1/q0-snn-compile/base_negative_count_input.json': 'a19f27b753abc5203a32b4b5c31ccf548f2bd3ddae95731928dc6aa99a369e1e', 'evidence/remote-qualify-v1/q0-snn-compile/batch_duplicate.json': '33d5772935e1abcb7445f0acc015543fce42f7befc5b78d728c91f1d8756beb2', 'evidence/remote-qualify-v1/q0-snn-compile/initial_threshold_boundary.json': '69cb669e71c5c095bb6b21fe578ab83736057bda825e82fa4164f926c096e967', 'evidence/remote-qualify-v1/q0-snn-compile/report.json': '133c2e8a6b8d911e3dcfc2e7c61e2029cf6e0ffec527a83ccf0ec08eb7cabf05', 'evidence/remote-qualify-v1/q0-snn-compile/single_sample_no_carry.json': 'f699d9e140218cb132f95995de4d8d2f141a02afaf39a83cc7fb6c69ec18905f', 'evidence/remote-qualify-v1/q0-snn-layerwise-terminal.json': '2018f57b6ee2bf8970f1a93287cd4b979dfe82209211c070358e71c08fbf377c', 'evidence/remote-qualify-v1/q0-snn-layerwise/base_negative_count_input.json': '7ac36bc7a29765bbb9f13b54cc1ea2d973f4577ba5dd5e7c2168a11a84f767b0', 'evidence/remote-qualify-v1/q0-snn-layerwise/batch_duplicate.json': '43502501a7ce9480dbc78ba3d2bebc0e895ac694562b6420544d37cb73245db3', 'evidence/remote-qualify-v1/q0-snn-layerwise/initial_threshold_boundary.json': '5d78ae8846e1f7b62a5c2149f87f8b9a36446d07a6ae464cc0082d58089b69c1', 'evidence/remote-qualify-v1/q0-snn-layerwise/report.json': 'cb0d2b066452bc333ffb80389fdc82cfcbc2585202dad6377ab1a727927c6961', 'evidence/remote-qualify-v1/q0-snn-layerwise/single_sample_no_carry.json': '825008d3f7505a63ca0f8d8dd95ccf6d3106ecebe508f0c571f6b1661981f66d', 'evidence/remote-qualify-v1/q0-spyx-terminal.json': '433237435ad63e4932bed0062f87b71feee73a7b59881b4c3a92abc2b1d4ec5b', 'evidence/remote-qualify-v1/q0-spyx/base_negative_count_input.json': 'dd00b7f27bf3870fa1b24d9f0184b117925c5d2146c819f2ddd07804fb343f60', 'evidence/remote-qualify-v1/q0-spyx/batch_duplicate.json': '6972d9ff246f96798315a42a7fe620a19b6d97bf486104fcb17e7b2c0d14e2fe', 'evidence/remote-qualify-v1/q0-spyx/initial_threshold_boundary.json': 'f88bcc02a8bd929c6b63e2c3caf43cff2a9b8dbf2c09ea17cca53c90a02b3346', 'evidence/remote-qualify-v1/q0-spyx/report.json': '7e0ab5076656f8ab7be0d87970ba2a7a29cca275b674ebb60aa527495d535e39', 'evidence/remote-qualify-v1/q0-spyx/single_sample_no_carry.json': '6e57bf3442bc32de4bd65e1bf17f81b77084b16210f6c03f837276083b4ca896', 'evidence/runtime-arm64-preparation-r1/build-contract.json': '6a4add6ba4c1eca58ad8b813b303d47e98db3d5c41c9ea250720a691e04cd1d9', 'fixtures/e1-small/manifest.json': '146a4691acefeea9128eac1360a61c0e6d9c130c28e6a5588de4ce8acd25e3b3', 'fixtures/e1-small/seed-11.json': '10e39bbdc76f3291264520415cc1f70b39b8a177bd7f9d916c5e78af371348dc', 'fixtures/e1-small/seed-23.json': '6a8c237f745d442cda4f1e817f37f3be0efd02839cd09d1a8c3b8bf844cc5c3f', 'fixtures/e1-small/seed-37.json': '74217ec216579b15e55916237b6e606188c88ab87ffa424e78684a9893ef961f', 'fixtures/e1-small/seed-51.json': '3a9cd2e91fc85a105af40e3bf2e01ac3d3f984db399cdde84f4d0c26f1d81da6', 'fixtures/e1-small/seed-71.json': '7f8854a105610ea32087d7130384d51c697e642d06c88375ee1990178de1e695', 'fixtures/q0.json': '625df75c9161212a19f72067f89f28dc4a4e279c29eafdb1f829c2f96a96eee9', 'protocol/execution-plan-r2.json': 'cc3dc9f0219db5d1ab3fdb16760cecce7114903244e62294110aa7b64b16d118', 'sources/snapshot-manifest.json': '9362676928272655074e0c83138cbdbd47cea8d8fc740123ca11802a6dabf7e1', 'tools/arm64_artifact_gate_r1.py': '37589818b19c5d17e89bbe547148034b4b908c181b7328804676c3b2ffddf20e', 'tools/benchmark_competitor.py': '2321b1ef829c8a9fe4820d558ddc7acdf88cc5080911e7ac95a3f10909789505', 'tools/benchmark_e1.py': '4810063fdf19a706031e64d66a0a2a9d36b9904216ad2662509d271e69200a85', 'tools/build_arm64_runtime_r1.py': '930a5bd5366351a66344c9b588e545f537d29335261ecf2e07569be1439752ed', 'tools/generate_dense_large.py': '910033835413339d5544d8341f6deeb5a96f571e19fb75bcd748596b2f7c316c', 'tools/generate_recurrent_cases.py': '5da896fa7bce9c4ad294535439ccdbc18281d1b61c30640c4859e7488e137b75', 'tools/qualify_dense.py': '35016f98c480de65d244cc7de81a2a21f7539a0d406a77fd62fcd9aa9e43af19', 'tools/qualify_dense_arm64_r1.py': '2dcabf162327a7ae0d1ae215d03097ac2cf0ce4a90744827d69f6ef9ac80b178', 'tools/qualify_jax.py': '954a072a26d630f7c56b56527b76bf4751a68ede8a8323797070bbf27268194a', 'tools/run_recurrent_queue.py': '2f29be1bdc9548866587a46eb7255ddf597c29a04330d585ea89ae8409106f3f', 'tools/validate_e1_run.py': '5f2ef025910d2a90d47b68bc4f9a2b4cff2437ca34f94b4550451143d1a04b77'}
VIEW_SPECS=[('atlas','cpu','atlas',[]),('sj-layerwise','cpu','spikingjelly_frontier',['--layerwise']),
 ('snn-layerwise','cpu','snntorch_fp64',['--layerwise']),('spyx','jax','spyx',[]),
 ('brainstate','jax','brainx_state',[]),('sj-compile','cpu','spikingjelly_frontier',['--layerwise','--compile']),
 ('snn-compile','cpu','snntorch_fp64',['--layerwise','--compile'])]
Q0_CASES={'base_negative_count_input','batch_duplicate','initial_threshold_boundary','single_sample_no_carry'}


def expected_order(root):
    views=[[name,str(root/f'environment/{environment}/bin/python'),engine,flags] for name,environment,engine,flags in VIEW_SPECS]
    return [[seed,view] for offset,seed in enumerate(SEEDS) for view in views[offset:]+views[:offset]]


def expected_command(root,run,seed,view):
    name,python,engine,flags=view;output=run/f'{name}-seed-{seed}'
    if name=='atlas':
        return [python,str(root/'tools/benchmark_e1.py'),'--worker','--base',str(root),'--source-root',str(root/'snapshot/brian2-rust'),
                '--runner',str(root/ARM64_RUNTIME),'--output',str(output),'--seed',str(seed)]
    return [python,str(root/'tools/benchmark_competitor.py'),'--engine',engine,'--seed',str(seed),'--output',str(output),*flags]


def no_ranking(result):
    for row in result.get('rows',[]):row['strict_ranking_eligible']=False
    for summary in result.get('views',{}).values():
        summary.update(strict_ranking_eligible=False,ratios_validated=False,paired_complete_n=0)
        for pair in summary.get('paired_seed_ratios',[]):pair.update(eligible=False,ratio_atlas_over_view=None)


def policy_errors(policy):
    required=dict(warmup_steps=10,measured_steps=50,per_seed_s=360,per_view_s=1800,finite_slots=35,
                  outer_recommended_cap_s=13000,validator_recommended_cap_s=600,rss_guard_bytes=64*1024**3,
                  no_old_seed_splicing=True)
    return ['frozen policy differs: '+key for key,value in required.items() if policy.get(key)!=value]


def phase_receipt(root,run,path,mode):
    errors=[];limitations=[];receipt={}
    if path is not None and path.is_file():
        receipt=read(path)
        if receipt.get('name')!='arm64-e1-small':errors.append('outer receipt stage name differs')
        command=receipt.get('command',[])
        if str(root/DISPATCHER) not in command or '--allow-run' not in command:errors.append('outer receipt did not launch the ARM64 dispatcher explicitly')
        if receipt.get('remaining_owned_processes')!=[]:errors.append('outer receipt does not attest complete owned-descendant cleanup')
        if not isinstance(receipt.get('timeout_s'),(int,float)) or not 0<receipt['timeout_s']<=13000:errors.append('outer phase cap outside (0,13000]')
        if receipt.get('termination_reason') not in ('exited','timeout','resource_limit'):errors.append('outer termination reason is not closed')
        if not isinstance(receipt.get('exit_code'),int):errors.append('outer receipt lacks an exit code')
        if not isinstance(receipt.get('elapsed_s'),(int,float)) or not math.isfinite(receipt['elapsed_s']) or receipt['elapsed_s']<0:errors.append('outer receipt lacks finite elapsed wall')
        receipt=dict(receipt,path=str(path),sha256=sha256(path))
    elif mode=='full':errors.append('full mode requires the independent outer phase terminal receipt')
    else:limitations.append('Outer phase receipt absent; closure/resource supervision not fully verified')
    return receipt,errors,limitations


def closed_prerequisite_rejection(root,run,mode,phase):
    errors=[];limitations=[]
    result=dict(schema='e1-small-arm64-evidence-validation-r1',mode=mode,root=str(root),run=str(run),
        errors=errors,limitations=limitations,rows=[],views={},raw_verified=False,native_profile='arm64-r1',
        prior_timing_used=False,supervisor_completed=False,all_declared_cases_completed=False,
        all_declared_resource_gates_passed=False,denominator=dict(views=VIEWS,seeds=SEEDS,scheduled_slots=35,
        expected_slots=35,expected_warmup=10,expected_measured=50),freeze_present=False)
    needed=['order.json','preflight.json','progress.json','terminal.json']
    for name in needed:
        if not (run/name).is_file():errors.append('missing no-execution closure evidence: '+name)
    if errors and phase.get('termination_reason') in ('timeout','resource_limit'):
        # An outer kill may interrupt even the tiny preflight/closure writes.
        # Retain the predeclared denominator without inventing a successful freeze.
        errors.clear()
        if (run/'order.json').is_file() and read(run/'order.json')!=expected_order(root):errors.append('outer-limited preflight order differs')
        for seed,view in expected_order(root):
            name=view[0];stem=f'{name}-seed-{seed}'
            if (run/stem).exists() or (run/(stem+'-terminal.json')).exists():errors.append('worker artifacts exist without a launch freeze: '+stem)
            result['rows'].append(dict(view=name,seed=seed,effective_status='not_launched_outer_'+phase['termination_reason'],
                worker_result_present=False,strict_ranking_eligible=False,numerical_qualification=False,resource_qualification=name not in ('spyx','brainstate'),errors=[]))
        for name in VIEWS:
            result['views'][name]=dict(scheduled_n=5,completed_n=0,independent_n=0,complete_five_seed_summary=False,
                strict_ranking_eligible=False,ratios_validated=False,paired_complete_n=0,
                paired_seed_ratios=[dict(seed=seed,eligible=False,ratio_atlas_over_view=None) for seed in SEEDS])
        limitations.append('Outer resource/timeout terminated before a complete freeze/closure; all 35 declared slots remain unlaunched, no model success or timing inferred')
        result.update(status='evidence_invalid' if errors else 'passed_no_execution_evidence_checks',scientific_status='outer_terminated_before_freeze')
        return result
    if errors:
        result.update(status='evidence_invalid',scientific_status='not_established')
        return result
    order=read(run/'order.json');pre=read(run/'preflight.json');terminal=read(run/'terminal.json')
    rows=terminal.get('records',[])
    if order!=expected_order(root):errors.append('no-execution order differs from all 35 new slots')
    if terminal.get('schema')!='e1-small-arm64-terminal-r1' or terminal.get('status')!='preflight_rejected':errors.append('missing freeze is allowed only for explicit preflight rejection')
    if pre.get('schema')!='e1-small-arm64-preflight-r1' or pre.get('status')!='rejected' or not pre.get('error_type') or not pre.get('error'):errors.append('preflight rejection lacks a recorded gate error')
    if pre.get('stage') not in ('exclusive_idle_gate','producer_source_runtime_q0_fixtures'):errors.append('unrecognized prerequisite rejection stage')
    if pre.get('producer_path')!=PRODUCER or pre.get('atlas_q0_path')!=ARM64_Q0:errors.append('rejection references incorrect ARM64 prerequisites')
    if pre.get('expected_fixed_identities')!=FIXED_PREEXECUTION:errors.append('preflight expected source/fixture/competitor identities differ')
    if terminal.get('preflight_sha256')!=sha256(run/'preflight.json') or terminal.get('order_sha256')!=sha256(run/'order.json'):errors.append('terminal preflight/order identity mismatch')
    if terminal.get('closed_slots')!=35 or len(rows)!=35 or read(run/'progress.json')!=rows:errors.append('no-execution closure must retain exactly 35 matching progress/terminal rows')
    expected=[(view[0],seed) for seed,view in expected_order(root)]
    actual=[(row.get('name'),row.get('seed')) for row in rows]
    if actual!=expected:errors.append('preflight rejection lost or reordered a declared seed/view')
    errors.extend(policy_errors(terminal.get('policy',{})))
    for row in rows:
        stem=f"{row.get('name')}-seed-{row.get('seed')}"
        if row.get('slot_name')!=stem or row.get('execution_status')!='not_launched_prerequisite' or row.get('performance_run') is not False:
            errors.append('rejected prerequisite slot is not explicitly unlaunched: '+stem)
        if any(key in row for key in ('command','exit_code','elapsed_s','supervisor_receipt_sha256')):errors.append('rejected prerequisite slot has unexpected launch evidence: '+stem)
        if (run/stem).exists() or (run/(stem+'-terminal.json')).exists():errors.append('worker artifacts exist despite whole-queue prerequisite rejection: '+stem)
        result['rows'].append(dict(view=row.get('name'),seed=row.get('seed'),effective_status='not_launched_prerequisite',
            reason=row.get('reason'),worker_result_present=False,strict_ranking_eligible=False,numerical_qualification=False,
            resource_qualification=row.get('name') not in ('spyx','brainstate'),errors=[]))
    for name in VIEWS:
        result['views'][name]=dict(scheduled_n=5,completed_n=0,independent_n=0,complete_five_seed_summary=False,
            strict_ranking_eligible=False,ratios_validated=False,paired_complete_n=0,
            statuses={str(seed):'not_launched_prerequisite' for seed in SEEDS},
            paired_seed_ratios=[dict(seed=seed,eligible=False,ratio_atlas_over_view=None) for seed in SEEDS])
    if mode=='full':
        if pre.get('dispatcher_sha256')!=sha256(root/DISPATCHER) or terminal.get('dispatcher_sha256')!=sha256(root/DISPATCHER):errors.append('rejection dispatcher identity changed')
        for relative,entry in pre.get('observed_prerequisites',{}).items():
            if relative not in (PRODUCER,ARM64_Q0):errors.append('unexpected observed prerequisite path');continue
            if entry.get('exists') and (not (root/relative).is_file() or sha256(root/relative)!=entry.get('sha256')):
                errors.append('recorded prerequisite changed after rejection: '+relative)
    result.update(preflight_sha256=sha256(run/'preflight.json'),terminal_sha256=sha256(run/'terminal.json'),
        scientific_status='no_model_started_prerequisite_rejected',status='evidence_invalid' if errors else 'passed_no_execution_evidence_checks')
    limitations.append('Closed prerequisite rejection is valid no-execution evidence, never qualification/completion/timing evidence; no missing runtime or Q0 is inferred as passed')
    return result


def check_resources(path,receipt,completed):
    issues=[];notes=[];count=0;peak=0;last=-1.;threads=0
    if not path.is_file():return (['completed job lacks owned-tree resource samples'] if completed else []),[],{}
    try:
        with path.open() as stream:
            for line in stream:
                if not line.strip():continue
                sample=json.loads(line);elapsed=sample['elapsed_s'];rss=sample['rss_by_pid']
                if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<last:raise ValueError('resource clock is not finite/monotonic')
                if any(not isinstance(v,int) or v<0 for v in rss.values()) or sum(rss.values())!=sample['aggregate_rss']:raise ValueError('same-sample RSS sum differs')
                last=elapsed;count+=1;peak=max(peak,sample['aggregate_rss']);threads=max(threads,max(sample.get('threads_by_pid',{}).values(),default=0))
        if receipt.get('peak_contemporaneous_rss')!=peak:issues.append('owned-tree resource peak differs from supervisor receipt')
    except Exception as error:
        (issues if completed else notes).append('resource samples unreadable/truncated: '+repr(error))
    return issues,notes,dict(resource_samples=count,observed_process_thread_count_max=threads,sampled_job_peak_rss_bytes=peak)


def check_frozen_bindings(root,freeze,mode):
    errors=[]
    if freeze.get('schema')!='e1-small-arm64-freeze-r1' or freeze.get('native_profile')!='arm64-r1':errors.append('freeze is not independent ARM64 E1-small')
    if freeze.get('runtime_path')!=ARM64_RUNTIME or freeze.get('atlas_q0_path')!=ARM64_Q0:errors.append('freeze runtime/Q0 path points outside the new ARM64 profile')
    if freeze.get('views')!=[v for _,v in expected_order(root)[:7]]:errors.append('freeze view commands or ordering differ')
    if freeze.get('jax_resource_qualified') is not False or freeze.get('prior_timing_used') is not False:errors.append('freeze changed JAX resource gate or claims old timing reuse')
    errors.extend(policy_errors(freeze.get('policy',{})))
    for relative,expected in FIXED_PREEXECUTION.items():
        if freeze.get('identities',{}).get(relative)!=expected:errors.append('freeze omits/changes prepared identity: '+relative)
    if mode!='full':return errors
    for relative,expected in freeze.get('identities',{}).items():
        if not (root/relative).is_file() or sha256(root/relative)!=expected:errors.append('live file differs from ARM64 freeze: '+relative)
    from arm64_artifact_gate_r1 import gate_artifact
    try:
        runtime=gate_artifact(root/ARM64_RUNTIME,'runner',freeze.get('runtime_sha256'))
        if runtime!=freeze.get('runtime'):errors.append('actual ARM64 runner architecture identity differs from freeze')
        producer=read(root/PRODUCER);contract=read(root/BUILD_CONTRACT)
        binding=freeze.get('producer',{})
        if (binding.get('path')!=PRODUCER or binding.get('sha256')!=sha256(root/PRODUCER) or
            binding.get('build_contract_path')!=BUILD_CONTRACT or binding.get('build_contract_sha256')!=sha256(root/BUILD_CONTRACT) or
            binding.get('runner')!=runtime):errors.append('producer binding differs from frozen ARM64 runner/contract')
        if (producer.get('schema')!='runtime-arm64-build-r1' or producer.get('status')!='built_unqualified' or
            producer.get('source_integrity_after_exit') is not True or producer.get('runtime_path')!=ARM64_RUNTIME or
            producer.get('runtime_sha256')!=runtime['sha256'] or producer.get('runtime_architecture')!='arm64-only'):
            errors.append('producer lacks successful build/unqualified ARM64/source-integrity attestation')
        if any(producer.get(k)!=contract['source_identities'] for k in ('sources_before','sources_after','sources_final')):errors.append('producer source-before/after/final receipts differ')
        if sha256(root/'runtime/b2-train')!=contract['original_runtime_sha256']:errors.append('preserved original runtime changed')
        for name in ('cpu','jax'):
            if gate_artifact(root/f'environment/{name}/bin/python','runner')!=freeze.get('python_interpreters',{}).get(name):errors.append(name+' interpreter ISA/SHA differs')
        for name,_,engine,flags in VIEW_SPECS:
            gate=freeze.get('q0',{}).get(name,{})
            relative=ARM64_Q0 if name=='atlas' else f'evidence/remote-qualify-v1/q0-{name}/report.json'
            if gate.get('path')!=relative or gate.get('sha256')!=sha256(root/relative) or gate.get('flags')!=flags or gate.get('engine')!=engine:
                errors.append('prior Q0 path/hash/flags/engine mismatch: '+name)
            q=read(root/relative)
            if q.get('dense_qualification_status')!='passed' or q.get('engine')!=engine or set(q.get('cases',{}))!=Q0_CASES or any(v.get('passed') is not True or not all_checks_pass(v.get('checks')) for v in q.get('cases',{}).values()):errors.append('prior Q0 is incomplete/unqualified: '+name)
            for rel,h in q.get('identities',{}).items():
                if not (root/rel).is_file() or sha256(root/rel)!=h:errors.append('prior Q0 source identity differs: '+rel)
            for case in Q0_CASES:
                raw_path=(root/relative).parent/(case+'.json')
                if gate.get('identities',{}).get(str(raw_path.relative_to(root)))!=sha256(raw_path):errors.append('Q0 raw file missing from freeze: '+name+'/'+case)
                raw=read(raw_path)
                if raw.get('checks')!=q['cases'][case].get('checks'):errors.append('Q0 raw/report check bank differs: '+name+'/'+case)
                if name=='atlas':
                    for entry in (raw.get('raw',{}),raw.get('raw',{}).get('sgd',{})):
                        if entry.get('runner_sha256')!=runtime['sha256'] or entry.get('training_api_sha256')!=sha256(root/'snapshot/brian2-rust/python/brian2_rust/training.py'):errors.append('ARM64 Q0 raw runtime/API mismatch: '+case)
            if name=='atlas' and (q.get('schema')!='dense-Q0-arm64-r1' or q.get('execution_status')!='completed' or q.get('runner')!=runtime):errors.append('new Atlas Q0 is not completed ARM64 evidence')
        manifest=read(root/'fixtures/e1-small/manifest.json');fixtures=freeze.get('fixtures',{})
        if fixtures.get('manifest_path')!='fixtures/e1-small/manifest.json' or fixtures.get('manifest_sha256')!=sha256(root/'fixtures/e1-small/manifest.json'):errors.append('common array manifest binding differs')
        expected=[dict(seed=seed,path=f'fixtures/e1-small/seed-{seed}.json',**manifest['files'][f'seed-{seed}.json']) for seed in SEEDS]
        if fixtures.get('seed_files')!=expected:errors.append('common arrays are not all five frozen seed files')
        for entry in expected:
            if sha256(root/entry['path'])!=entry['sha256'] or (root/entry['path']).stat().st_size!=entry['bytes']:errors.append('common array identity/bytes differs: '+entry['path'])
    except Exception as error:errors.append('ARM64 producer/ISA/Q0/fixture binding audit failed: '+repr(error))
    return errors


def validate(root,run,mode,phase_terminal=None):
    phase,phase_errors,phase_notes=phase_receipt(root,run,phase_terminal,mode)
    if not (run/'freeze.json').is_file():
        result=closed_prerequisite_rejection(root,run,mode,phase)
        result['errors'].extend(phase_errors);result['limitations'].extend(phase_notes)
        result['outer_phase']=phase
        if result['errors']:result['status']='evidence_invalid'
        no_ranking(result)
        return result
    try:
        read(run/'freeze.json')
    except (OSError,ValueError) as error:
        if phase.get('termination_reason') in ('timeout','resource_limit'):
            result=closed_prerequisite_rejection(root,run,mode,phase)
            result['errors'].extend(phase_errors);result['limitations'].extend(phase_notes)
            result['limitations'].append('Outer limit interrupted freeze publication: '+repr(error))
            result['outer_phase']=phase
            if result['errors']:result['status']='evidence_invalid'
            no_ranking(result);return result
        raise
    result=_validate_original(root,run,mode)
    errors=result['errors'];limitations=result['limitations']
    errors.extend(phase_errors);limitations.extend(phase_notes)
    freeze=read(run/'freeze.json');result.update(native_profile='arm64-r1',prior_timing_used=False,outer_phase=phase)
    errors.extend(check_frozen_bindings(root,freeze,mode))
    if read(run/'order.json')!=expected_order(root):errors.append('ARM64 order must be all 35 new rotated slots, unchanged seeds')
    terminal_path=run/'terminal.json'
    terminal=read(terminal_path) if terminal_path.is_file() else {}
    records=terminal.get('records',[]) if terminal else (read(run/'progress.json') if (run/'progress.json').exists() else [])
    if len(records)!=35 or [(r.get('name'),r.get('seed')) for r in records]!=[(v[0],seed) for seed,v in expected_order(root)]:errors.append('preallocated ledger must retain all 35 seeds/views in order')
    if terminal:
        if terminal.get('schema')!='e1-small-arm64-terminal-r1' or terminal.get('closed_slots')!=35:errors.append('terminal schema or closed slot denominator differs')
        if terminal.get('freeze_sha256')!=sha256(run/'freeze.json') or terminal.get('order_sha256')!=sha256(run/'order.json'):errors.append('terminal freeze/order identity differs')
        if read(run/'progress.json')!=records:errors.append('terminal and final progress disagree')
        if terminal.get('preflight_sha256')!=sha256(run/'preflight.json'):errors.append('terminal preflight identity differs')
    elif phase.get('termination_reason') not in ('timeout','resource_limit'):
        errors.append('queue terminal missing without an observed outer resource/timeout closure')
    pre=read(run/'preflight.json')
    if pre.get('status')!='passed' or pre.get('freeze_sha256')!=sha256(run/'freeze.json'):errors.append('launch freeze lacks successful prerequisite receipt')
    if mode=='full' and pre.get('dispatcher_sha256')!=sha256(root/DISPATCHER):errors.append('dispatcher identity differs from successful preflight')
    ledger={(r.get('name'),r.get('seed')):r for r in records}
    view_spent={name:0. for name in VIEWS}
    for row,(seed,view) in zip(result['rows'],expected_order(root)):
        name=view[0];stem=f'{name}-seed-{seed}';external=ledger.get((name,seed),{})
        row['slot_elapsed_s']=external.get('slot_elapsed_s');row['slot_cap_s']=external.get('slot_cap_s')
        if external.get('slot_cap_s') is not None and not near(external.get('view_spent_before_s',math.nan),view_spent[name]):errors.append('per-view spent-before accounting differs: '+stem)
        if (row['view'],row['seed'])!=(name,seed):errors.append('row/order identity mismatch: '+stem)
        if external.get('slot_name')!=stem:errors.append('slot name mismatch: '+stem)
        launched='command' in external
        completed=row['effective_status']=='completed'
        receipt_path=run/(stem+'-terminal.json')
        if launched:
            if external['command']!=expected_command(root,run,seed,view):errors.append('worker command differs from frozen ARM64 path/options: '+stem)
            if not receipt_path.is_file():
                if phase.get('termination_reason') not in ('timeout','resource_limit'):errors.append('launched job lacks owned-tree supervisor receipt: '+stem)
            else:
                receipt=read(receipt_path)
                interrupted=phase.get('termination_reason') in ('timeout','resource_limit') and external.get('execution_status')=='launching'
                if external.get('supervisor_receipt_sha256')!=sha256(receipt_path):
                    (limitations if interrupted and not external.get('supervisor_receipt_sha256') else errors).append('supervisor receipt SHA not yet committed or differs: '+stem)
                for key in ('command','timeout_s','exit_code','elapsed_s','termination_reason','remaining_owned_processes','process_identities','peak_contemporaneous_rss'):
                    if receipt.get(key)!=external.get(key):
                        (limitations if interrupted and key not in external else errors).append('ledger/supervisor '+key+' missing or differs: '+stem)
                if receipt.get('name')!=stem or receipt.get('remaining_owned_processes')!=[]:errors.append('owned descendant cleanup incomplete: '+stem)
                if not receipt.get('process_identities') or any(not isinstance(p.get('pid'),int) or not isinstance(p.get('create_time'),(int,float)) for p in receipt.get('process_identities',[])):errors.append('owned PID/create_time identity ledger missing: '+stem)
                if receipt.get('rss_guard_bytes')!=64*1024**3:errors.append('resource guard differs: '+stem)
                if mode=='full':
                    resources=run/(stem+'-resources.jsonl')
                    if external.get('resource_samples_sha256') and resources.is_file() and external['resource_samples_sha256']!=sha256(resources):errors.append('resource sample identity differs: '+stem)
                    found,notes,stats=check_resources(resources,receipt,completed)
                    errors.extend(stem+': '+e for e in found);limitations.extend(stem+': '+e for e in notes);row.update(stats)
            elapsed=external.get('slot_elapsed_s');cap=external.get('slot_cap_s');child_elapsed=external.get('elapsed_s');child_cap=external.get('timeout_s')
            valid=all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in (elapsed,cap,child_elapsed,child_cap))
            if not valid:
                if completed:errors.append('completed job lacks finite complete wall accounting: '+stem)
            else:
                if not 0<cap<=min(360.,max(0.,1800.-view_spent[name]))+1e-9 or child_cap>cap:errors.append('seed/remaining-view declared cap differs: '+stem)
                strict=elapsed<=cap and child_elapsed<=child_cap and external.get('termination_reason') not in ('timeout','resource_limit')
                row['strict_wall_passed']=strict
                if not strict:
                    row['effective_status']='resource_limit' if external.get('termination_reason')=='resource_limit' else 'timeout'
                    row['strict_ranking_eligible']=False
                if completed and external.get('exit_code')!=0:errors.append('completed job lacks zero supervisor exit: '+stem)
                view_spent[name]+=elapsed
                if elapsed+1e-9<child_elapsed:errors.append('whole-slot wall shorter than supervised child wall: '+stem)
            if completed and mode=='full' and not external.get('resource_samples_sha256'):errors.append('completed job has no resource sample identity: '+stem)
        else:
            if external.get('execution_status') not in ('timeout','not_launched_prerequisite','not_launched_gate_error','not_launched_pending'):
                errors.append('unlaunched slot lacks explicit budget/gate status: '+stem)
            if external.get('slot_elapsed_s') is not None:view_spent[name]+=external['slot_elapsed_s']
            if (run/stem/'result.json').exists():errors.append('worker result exists in an unlaunched slot: '+stem)
        worker=read(run/stem/'result.json') if (run/stem/'result.json').is_file() else {}
        if worker:
            if worker.get('seed')!=seed:errors.append('worker seed differs: '+stem)
            expected_engine='Atlas' if name=='atlas' else view[2]
            if worker.get('engine')!=expected_engine:errors.append('worker engine differs: '+stem)
            expected_array=next((e['sha256'] for e in freeze.get('fixtures',{}).get('seed_files',[]) if e.get('seed')==seed),None)
            if row.get('array_sha256') and row['array_sha256']!=expected_array:errors.append('worker common array hash differs from exact frozen seed: '+stem)
            if row.get('effective_status')=='completed':
                times=([n/1e9 for n in worker.get('measured_public_api_ns',[])] if name=='atlas' else worker.get('measured_seconds',[]))
                wall=worker.get('process_wall_seconds',worker.get('wall_s'))
                if not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall<0:errors.append('completed worker lacks finite wall: '+stem)
                elif sum(times)>wall+1e-6 or wall>external.get('elapsed_s',math.inf)+1e-6:errors.append('measured-step/worker/supervisor wall ordering differs: '+stem)

    result['strict_view_wall_s']=view_spent
    if terminal:
        for name in VIEWS:
            if not near(terminal.get('view_spent_s',{}).get(name,math.nan),view_spent[name]):errors.append('terminal per-view accounting differs: '+name)
    # Rebuild summaries from final strict statuses; never retain partial or old-seed ratios.
    atlas={r['seed']:r for r in result['rows'] if r['view']=='atlas'}
    outer_ok=(phase.get('termination_reason')=='exited' and phase.get('exit_code')==0 and phase.get('elapsed_s',math.inf)<=phase.get('timeout_s',0))
    if phase and sum(view_spent.values())>phase.get('elapsed_s',math.inf)+1e-6:errors.append('sum of complete per-view slot walls exceeds outer phase wall')
    if phase and not outer_ok:limitations.append('Outer phase did not exit successfully; its limit/failure takes priority over timing summaries')
    for name in VIEWS:
        rows=[r for r in result['rows'] if r['view']==name];summary=result['views'].setdefault(name,{})
        complete=[r for r in rows if r['effective_status']=='completed' and not r['errors'] and r.get('strict_wall_passed')]
        whole=len(complete)==5 and view_spent[name]<=1800 and outer_ok
        summary.update(scheduled_n=5,completed_n=len(complete),independent_n=len(complete),complete_five_seed_summary=whole,
            supervisor_wall_s=view_spent[name],statuses={str(r['seed']):r['effective_status'] for r in rows},
            strict_ranking_eligible=whole and name not in ('spyx','brainstate'),ratios_validated=False)
        if not whole:
            for key in ('median_of_process_medians_s','min_process_median_s','max_process_median_s'):summary.pop(key,None)
        pairs=[]
        atlas_whole=all(s in atlas and atlas[s]['effective_status']=='completed' and not atlas[s]['errors'] and atlas[s].get('strict_wall_passed') for s in SEEDS) and view_spent['atlas']<=1800
        eligible=whole and atlas_whole and name not in ('spyx','brainstate') and mode=='full' and not errors
        for seed in SEEDS:
            row=next(r for r in rows if r['seed']==seed)
            pairs.append(dict(seed=seed,eligible=eligible,ratio_atlas_over_view=atlas[seed]['seed_median_s']/row['seed_median_s'] if eligible else None))
        summary.update(paired_seed_ratios=pairs,paired_complete_n=5 if eligible else 0,ratios_validated=eligible)
    result.update(actual_numerical_protocol=freeze.get('actual_numerical_protocol'),
        all_declared_cases_completed=all(s.get('complete_five_seed_summary') for s in result['views'].values()),
        status='evidence_invalid' if errors else ('passed_full_evidence_checks' if mode=='full' else 'passed_report_checks_raw_unverified'))
    limitations.append('ARM64 changes only executable ISA: Atlas full-shape qualification still checks one Adam update, with tiny dense Q0 three Adam; competitors retain their original full-shape timing-path three-Adam checks')
    if errors or mode!='full' or not outer_ok:no_ranking(result)
    return result


def markdown(result):
    lines=['# ARM64 E1-small 证据审查','',f"状态：{result['status']}。独立 ARM64 profile；未重跑模型或数值 oracle。",'',
           '固定分母：7 views × 5 seeds = 35 槽；每进程 10 次预热 + 50 次计时。',
           'Atlas 保留原 full-shape 1 Adam + tiny Q0 3 Adam；对手保留实际计时路径 3 Adam。',
           'JAX 不具备单线程资源资格；RSS 包含资格、编译与诊断缓存。','',
           '| View | 完成种子 | 完整五种子 | 可验证配对 |','|---|---:|---|---|']
    for name in VIEWS:
        value=result.get('views',{}).get(name,{})
        lines.append(f"| {name} | {value.get('completed_n',0)}/5 | {value.get('complete_five_seed_summary',False)} | {value.get('ratios_validated',False)} |")
    if result['errors']:lines+=['','错误：']+['- '+e for e in result['errors']]
    lines+=['','限制：']+['- '+e for e in result.get('limitations',[])]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,default=Path('evidence/arm64-r1/e1-small'))
    parser.add_argument('--mode',choices=['report-only','full'],default='report-only')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--phase-terminal',type=Path,default=Path('evidence/extended-followup-r3/arm64-e1-small-terminal.json'))
    args=parser.parse_args();root=args.root.resolve()
    run=args.run.resolve() if args.run.is_absolute() else root/args.run
    phase_terminal=args.phase_terminal if args.phase_terminal.is_absolute() else root/args.phase_terminal
    result=validate(root,run,args.mode,phase_terminal)
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as out:out.write(text)
        print(json.dumps(dict(status=result['status'],errors=len(result['errors']),output=str(args.output))))
    else:
        print(text,end='')
    if args.report:
        with args.report.open('x') as out:out.write(markdown(result))
    return 1 if result['errors'] else 0


if __name__=='__main__':
    raise SystemExit(main())
