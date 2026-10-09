#!/usr/bin/env python3
"""Seal existing A1-v4 fixtures only after its closed, successful full audit.

No model, oracle, held-out payload or package is loaded. The exact frozen
stdlib auditor's fixture checker is reused after an independent receipt gate.
A current manifest alone is not provenance: the preparation stdout digest and
available original worker result/progress hashes must agree with it first.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import runpy
import socket
import sys

AUDITOR = 'tools/validate_a1_v4.py'
AUDITOR_SHA = 'e4371d0d1518c1bd2fd8ebb876b08a6c99b63d4abdde04a2c9d48b091c31314d'
WAITER = 'tools/run_a1_v4_validation_when_ready.py'
CONTRACT = 'protocol/A1-execution-addendum-v4.json'
GENERATOR = 'tools/run_a1_mnist_v4.py'
SEEDS = [11, 23, 37, 51, 71]
VIEWS = [('atlas','atlas',False), ('sj-layerwise','spikingjelly_frontier',False),
         ('snn-layerwise','snntorch_fp64',False), ('sj-compile','spikingjelly_frontier',True),
         ('snn-compile','snntorch_fp64',True)]
SUCCESS = 'evidence_consistent_full_checks_with_limits'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def allowed(path):
    path = Path(path)
    for name in [path.name.lower(), path.resolve().name.lower()]:
        check(not any(token in name for token in ('t10k-', 'test-label', 'test-image')), 'held-out payload reads are prohibited')
    return path


def path_in(root, rel):
    path = (root/rel).resolve()
    path.relative_to(root.resolve())
    return allowed(path)


def sha(path):
    h = hashlib.sha256()
    with allowed(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            check(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON token: '+value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return loads(allowed(path).read_text())


def expected_plan():
    return [dict(view=name,engine=engine,compiled=compiled,seed=seed)
            for offset,seed in enumerate(SEEDS) for name,engine,compiled in VIEWS[offset:]+VIEWS[:offset]]


def gate(report, job, launch, queue, freeze):
    check(report.get('schema') == 'a1-v4-evidence-validation-v1' and report.get('mode') == 'full', 'a full A1-v4 independent audit is required')
    check(report.get('status') == SUCCESS and report.get('errors') == [], 'full audit is not error-free and consistent')
    check(report.get('validator_sha256') == AUDITOR_SHA, 'audit was produced by a different validator revision')
    check(job.get('status') == 'validator_exited' and job.get('exit_code') == 0, 'audit process did not exit0 normally')
    check(job.get('validator_sha256') == AUDITOR_SHA and launch.get('validator_sha256') == AUDITOR_SHA, 'audit job source binding differs')
    check(job.get('validation_status') == SUCCESS and job.get('validation_errors') == [], 'audit job terminal disagrees with report')
    denominator = report.get('denominator', {})
    for key, want in [('formal_slots',25),('retained_rows',25),('recorded_outer_slots',25),('formal_seeds',SEEDS),('complete_terminal_ledger',True)]:
        check(denominator.get(key) == want, 'audit denominator is not closed: '+key)
    check(denominator.get('views') == [row[0] for row in VIEWS], 'audit formal views differ')
    check(queue.get('status') == 'finite_a1_queue_exited' and queue.get('finite_slots') == 25, 'v4 queue has not closed all25 formal slots')
    check(freeze.get('schema') == 'a1-queue-v4' and freeze.get('order') == expected_plan(), 'v4 frozen order differs')
    records = queue.get('records', [])
    check(len(records) == 25, 'v4 queue terminal must contain25 outcomes')
    for record, slot in zip(records, expected_plan()):
        check(all(record.get(key) == value for key,value in slot.items()), 'v4 queue record identity/order differs')
    rows = report.get('rows', [])
    check(len(rows) == 25 and {(r.get('view'),r.get('seed')) for r in rows} == {(v[0],s) for v in VIEWS for s in SEEDS}, 'audit rows omit/duplicate a formal slot')
    check(all(row.get('errors') == [] for row in rows), 'audit has slot-level evidence errors')
    # Scientific failures are deliberately allowed; fixture provenance does not
    # require successful model fitting or attainment of a quality threshold.
    return {f"{row['view']}-seed-{row['seed']}":row for row in rows}


def bind_existing_manifest(manifest, current_sha, preparation_sha, observations):
    check(current_sha == preparation_sha, 'current manifest differs from original v4 preparation receipt; refusing self-consistent recertification')
    check(manifest.get('schema') == 'a1-shared-arrays-v4' and manifest.get('seeds') == SEEDS, 'existing shared manifest schema/seeds differ')
    expected = {f'seed-{seed}.npz' for seed in SEEDS}
    check(set(manifest.get('files', {})) == expected, 'existing manifest must bind all five NPZ files')
    seen = {seed:[] for seed in SEEDS}
    for row in observations:
        seed = row['seed']
        check(seed in seen, 'unknown observed seed')
        check(row['shared_manifest_sha256'] == preparation_sha, 'a v4 worker used another shared manifest')
        check(row['common_arrays_sha256'] == manifest['files'][f'seed-{seed}.npz']['sha256'], 'a v4 worker used another NPZ for its seed')
        check(row['contract_sha256'] == manifest['contract_sha256'], 'a v4 worker used another base contract')
        seen[seed].append(row['path'])
    return seen


def seal(args):
    check(socket.gethostname().split('.')[0] == 'rock-mac-studio-1', 'fixture payload validation is restricted to100.90.28.27')
    root = args.root.resolve()
    run = path_in(root, args.run)
    audit_path = path_in(root, args.audit_report)
    job_dir = path_in(root, args.audit_job)
    report, job, launch = read(audit_path), read(job_dir/'terminal.json'), read(job_dir/'launch.json')
    queue, freeze = read(run/'terminal.json'), read(run/'freeze.json')
    rows = gate(report, job, launch, queue, freeze)
    check(Path(report['root']).resolve() == root and Path(report['run']).resolve() == run, 'full audit refers to another evidence root')
    check(sha(root/AUDITOR) == AUDITOR_SHA, 'frozen auditor source changed')
    check(job['output_sha256'] == sha(audit_path), 'full audit output differs from job terminal digest')
    check(launch['waiter_sha256'] == sha(root/WAITER), 'audit waiter differs from its launch receipt')
    barrier = read(job_dir/'barrier.json')
    check(barrier.get('status') == 'released' and barrier.get('prerequisite_sha256') == sha(run/'terminal.json'), 'audit was not bound to this closed v4 terminal')
    identities = {}
    def bind(path):
        path = allowed(path).resolve()
        relative = str(path.relative_to(root))
        identities[relative] = sha(path)
        return identities[relative]
    for path in [audit_path, job_dir/'terminal.json', job_dir/'launch.json', job_dir/'barrier.json',
                 root/AUDITOR, root/WAITER, Path(__file__), run/'terminal.json', run/'freeze.json']:
        bind(path)
    # Detect sources or Q0 evidence changed since the full audit, not merely
    # the helper/auditor hash. No model code is imported by these byte reads.
    for rel, expected in freeze['files'].items():
        check(bind(path_in(root, rel)) == expected, 'v4 frozen source/runtime/Q0 changed: '+rel)
    prep = read(run/'preparation.json')
    check(prep == read(run/'shared-arrays-terminal.json'), 'shared-array launch receipt differs from coordinator copy')
    check(prep.get('name') == 'shared-arrays' and prep.get('exit_code') == 0 and prep.get('termination_reason') == 'exited', 'v4 array preparation did not finish normally')
    check(not prep.get('remaining_owned_processes'), 'array preparation cleanup was not verified')
    for path in [run/'preparation.json', run/'shared-arrays-terminal.json', run/'shared-arrays.log']:
        bind(path)
    receipts = []
    for line in (run/'shared-arrays.log').read_text().splitlines():
        try:
            value = loads(line)
            if isinstance(value, dict) and value.get('status') == 'prepared': receipts.append(value)
        except ValueError:
            continue
    check(len(receipts) == 1, 'exactly one original shared-array preparation stdout receipt is required')
    manifest_path = root/'fixtures/a1-v4/manifest.json'
    check(Path(receipts[0]['manifest']).resolve() == manifest_path, 'v4 preparation stdout names another manifest')
    manifest = read(manifest_path)
    manifest_sha = bind(manifest_path)
    check(manifest['contract_sha256'] == freeze['files'][CONTRACT], 'manifest base contract differs from v4 freeze')
    check(manifest['generator_sha256'] == freeze['files'][GENERATOR], 'manifest generator differs from v4 freeze')
    observations, incomplete = [], []
    for slot in expected_plan():
        name = f"{slot['view']}-seed-{slot['seed']}"
        directory = run/name
        for filename in ['result.json','progress.json']:
            path = directory/filename
            if not path.is_file(): continue
            bind(path)
            try:
                value = read(path)
            except (OSError,ValueError) as error:
                check(rows[name]['effective_status'] in ('timeout','resource_limit','censored_at_cap'), 'unreadable worker identity without an audited interruption')
                incomplete.append(dict(path=str(path.relative_to(root)),reason=str(error)))
                continue
            if value.get('common_arrays_sha256') is None: continue
            check(value.get('schema') == 'a1-mnist-run-v4' and value.get('seed') == slot['seed'] and value.get('engine') == slot['engine'] and value.get('compile') is slot['compiled'], 'worker fixture identity record belongs to another slot')
            observations.append(dict(seed=slot['seed'], path=str(path.relative_to(root)),
                shared_manifest_sha256=value.get('shared_manifest_sha256'),common_arrays_sha256=value['common_arrays_sha256'],
                contract_sha256=value.get('contract_sha256')))
        # Preserve the original supervising receipts that identify those reports.
        for path in [directory/'terminal.json',directory/'launch.json',run/(name+'-terminal.json')]:
            if path.is_file(): bind(path)
    seen = bind_existing_manifest(manifest,manifest_sha,receipts[0]['sha256'],observations)
    # The frozen full auditor did not expose its expected fixture hashes in its
    # output. Repeat exactly its reviewed stdlib fixture checks after binding
    # the manifest to the original preparation receipt and worker observations.
    module = runpy.run_path(str(root/AUDITOR),run_name='frozen_fixture_checker_only')
    repeated = module['Audit']('fixture reseal')
    checked_manifest = module['check_shared'](root,freeze,repeated,True,True)
    check(not repeated.errors and checked_manifest == manifest, 'repeated frozen fixture validation failed: '+json.dumps(repeated.errors))
    seed_files = []
    for seed in SEEDS:
        path = root/'fixtures/a1-v4'/f'seed-{seed}.npz'
        actual = bind(path)
        entry = manifest['files'][path.name]
        check(actual == entry['sha256'] and path.stat().st_size == entry['bytes'], 'NPZ identity differs after fixture validation')
        seed_files.append(dict(seed=seed,path=str(path.relative_to(root)),sha256=actual,bytes=path.stat().st_size))
    split_files = []
    for filename, expected in manifest['split_sha256'].items():
        path = path_in(root/'data/mnist/processed',filename)
        actual = bind(path)
        check(actual == expected, 'split/preprocessing identity differs after validation')
        split_files.append(dict(path=str(path.relative_to(root)),sha256=actual,bytes=path.stat().st_size))
    # Pin the raw training sources too: future readers can verify the exact
    # encoding inputs without opening any held-out source. The frozen checker
    # already validated these expected hashes against preprocessing metadata.
    metadata = read(root/'data/mnist/processed/preprocess.json')
    for filename in ('train-images-idx3-ubyte','train-labels-idx1-ubyte'):
        source = metadata['raw_sources'][filename]
        path = path_in(root/'data',source['path'])
        check(bind(path) == source['sha256'], 'raw training source changed after fixture validation')
    # All sources and receipts are immutable observations for this seal; catch
    # changes while rereading/validating instead of emitting a mixed snapshot.
    check(all(sha(path_in(root,rel)) == expected for rel,expected in identities.items()), 'bound file changed while sealing fixture receipt')
    return dict(schema='a1-v4-fixture-receipt-v1',status='sealed_after_closed_full_audit',
        sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),performance_run=False,
        audit=dict(report_path=str(audit_path.relative_to(root)),sha256=sha(audit_path),validator_path=AUDITOR,validator_sha256=AUDITOR_SHA,
            job_terminal_path=str((job_dir/'terminal.json').relative_to(root)),job_terminal_sha256=sha(job_dir/'terminal.json'),
            waiter_path=WAITER,waiter_sha256=sha(root/WAITER),exit_code=0,mode='full',closed_slots=25),
        fixtures=dict(manifest_path=str(manifest_path.relative_to(root)),manifest_sha256=manifest_sha,
            contract_path=CONTRACT,contract_sha256=manifest['contract_sha256'],generator_path=GENERATOR,generator_sha256=manifest['generator_sha256'],
            seed_files=seed_files,split_files=split_files),
        identities=identities,
        provenance=dict(original_preparation_manifest_sha256=receipts[0]['sha256'],
            worker_bindings_by_seed={str(seed):paths for seed,paths in seen.items()},worker_identity_observations=observations,
            incomplete_worker_json=incomplete,
            repeat_check='Exact frozen stdlib auditor.check_shared, after original preparation/worker hash cross-binding',
            repeated_fixture_checks='passed', scientific_success_not_required=True),
        limitations=[*repeated.limitations,
            'This seals fixture identity and provenance, not scientific success, attained quality, model numerical equivalence or a new held-out test permission.',
            'Seeds lacking a worker array field rely on the original preparation manifest receipt plus repeated frozen fixture validation; they are not inferred from a new current manifest alone.',
            'No held-out payload, test label, prediction, model or numerical oracle is read or executed.'])


def self_test():
    empty = 'a'*64
    report=dict(schema='a1-v4-evidence-validation-v1',mode='full',status=SUCCESS,errors=[],validator_sha256=AUDITOR_SHA,
        denominator=dict(formal_slots=25,retained_rows=25,recorded_outer_slots=25,formal_seeds=SEEDS,views=[v[0] for v in VIEWS],complete_terminal_ledger=True),
        rows=[dict(**slot,errors=[],effective_status='budget_rejected') for slot in expected_plan()])
    job=dict(status='validator_exited',exit_code=0,validator_sha256=AUDITOR_SHA,validation_status=SUCCESS,validation_errors=[])
    launch=dict(validator_sha256=AUDITOR_SHA)
    queue=dict(status='finite_a1_queue_exited',finite_slots=25,records=expected_plan())
    freeze=dict(schema='a1-queue-v4',order=expected_plan())
    tests={}
    tests['closed25_scientific_failures_allowed']=len(gate(report,job,launch,queue,freeze))==25
    def rejects(fn):
        try:fn();return False
        except ValueError:return True
    tests['nonzero_audit_exit_rejected']=rejects(lambda:gate(report,{**job,'exit_code':1},launch,queue,freeze))
    tests['report_only_rejected']=rejects(lambda:gate({**report,'mode':'report-only'},job,launch,queue,freeze))
    tests['incomplete25_rejected']=rejects(lambda:gate(report,job,launch,{**queue,'records':expected_plan()[:-1]},freeze))
    manifest=dict(schema='a1-shared-arrays-v4',seeds=SEEDS,contract_sha256='c'*64,files={f'seed-{s}.npz':{'sha256':str(s)*32} for s in SEEDS})
    observations=[dict(seed=s,path=f'worker-{s}',shared_manifest_sha256=empty,common_arrays_sha256=str(s)*32,contract_sha256='c'*64) for s in SEEDS]
    tests['original_manifest_and_workers_agree']=len(bind_existing_manifest(manifest,empty,empty,observations))==5
    tests['joint_current_replacement_rejected']=rejects(lambda:bind_existing_manifest(manifest,'b'*64,empty,observations))
    bad=[{**observations[0],'common_arrays_sha256':'wrong'},*observations[1:]]
    tests['worker_array_mismatch_rejected']=rejects(lambda:bind_existing_manifest(manifest,empty,empty,bad))
    tests['missing_worker_array_requires_original_preparation_anchor']=len(bind_existing_manifest(manifest,empty,empty,[]))==5
    tests['held_out_open_guard']=rejects(lambda:allowed('/tmp/t10k-labels-idx1-ubyte'))
    return dict(schema='a1-v4-fixture-receipt-synthetic-v1',status='passed' if all(tests.values()) else 'failed',tests=tests,
                real_evidence_read=False,model_imports=False,held_out_read=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--run',default='evidence/a1-queue-v4')
    parser.add_argument('--audit-report',default='evidence/a1-v4-validation-r1.json')
    parser.add_argument('--audit-job',default='evidence/a1-v4-validation-job-r1')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    try:
        if args.self_test:
            result=self_test()
            check(result['status']=='passed','synthetic checks failed')
        else:
            if args.root is None or args.output is None:parser.error('--root and a fresh --output are required')
            result=seal(args)
        if args.output:
            with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
            print(json.dumps(dict(status=result['status'],path=str(args.output),sha256=sha(args.output))))
        else:print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
        print(json.dumps(dict(status='receipt_not_sealed',error_type=type(error).__name__,error=str(error))),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
