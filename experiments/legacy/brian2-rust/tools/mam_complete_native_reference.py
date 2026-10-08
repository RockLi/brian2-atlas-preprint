"""Publish diagnostic completion only after existing terminal and science gates.

This is an offline control-record audit, not a simulation, raw reread or
scientific acceptance test. Missing terminal analysis returns before writes.
"""
import argparse
import datetime
import json
from pathlib import Path

from mam_launch_native_full_reference import CAMPAIGN, PROTOCOL_SHA, label_for, protocol_gate
from mam_launch_native_analysis import analysis_paths, command, guard_ok, prerequisites
from mam_primary_analysis_pipeline import check, read, sha

STAGES = ['activity', 'cell', 'correlation', 'series', 'fc', 'lags']
REPORTS = dict(activity='activity.json', cell='paper-cell-metrics.json',
               correlation='correlation.json', series='time-series.json',
               fc='fc.json', lags='lags.json')


def validate_analysis(directory, gate, seed):
    label = label_for(seed)
    required = ['controller-complete.json', 'report.json', 'summary.json',
                'raw-guard.json', 'science-guard.json', 'raw-controller.json',
                'science-controller.json', 'science-budget.json', 'admission.json']
    required += [stage+'/'+name for stage, report in REPORTS.items()
                 for name in ['catalog.json', report]]
    for name in required:
        p = directory/name
        check(p.is_file() and not p.is_symlink() and p.stat().st_size < 2*2**20,
              'missing or oversized terminal evidence: '+name)
    complete, report, raw, budget, admission = [read(directory/n) for n in [
        'controller-complete.json', 'report.json', 'summary.json',
        'science-budget.json', 'admission.json']]
    for value in [complete, report, budget, admission]:
        check(value['label'] == label and value['seed'] == seed
              and value['protocol_sha256'] == PROTOCOL_SHA, 'completion identity differs')
    check(raw['label'] == label and raw['seed'] == seed and raw['passed'] is True
          and raw['raw_output_audit_passed'] is True and raw['audit_guard_passed'] is True
          and raw['terminal_resource_audit_passed'] is True
          and raw['all_first_2500ms_event_prefixes_exact'] is True,
          'guarded raw audit is incomplete')
    check(raw['resource_report_sha256'] == admission['resource_report_sha256']
          == gate['resource_report_sha256']
          and raw['collection_report_sha256'] == admission['collection_report_sha256']
          == gate['collection_report_sha256'], 'resource/collection provenance differs')
    for phase in ['raw', 'science']:
        controller = read(directory/(phase+'-controller.json'))
        cmd = controller['command']
        check(cmd.count('--timeout') == 1, 'ambiguous analysis deadline')
        timeout = int(cmd[cmd.index('--timeout')+1])
        check(cmd == command(phase, timeout, reference_seed=seed)
              and type(controller['returncode']) is int and controller['returncode'] == 0
              and controller['error'] is None, 'analysis controller failed or changed')
        guard_ok(directory/(phase+'-guard.json'), phase, timeout, reference_seed=seed)
    check(complete['ready'] is True and complete['analysis_complete'] is True
          and report['analysis_complete'] is True
          and complete['scientific_acceptance'] is False
          and report['scientific_acceptance'] is False
          and complete['performance_cost_acceptance'] is False
          and report['performance_cost_acceptance'] is False,
          'engineering completion cannot confer scientific/performance acceptance')
    check(complete['required_stages'] == report['required_stages'] == STAGES
          and complete['interarea_analysis_complete'] is True
          and report['interarea_analysis_complete'] is True
          and set(report['catalogs']) == set(STAGES), 'all six scientific stages required')
    check(complete['raw_summary_sha256'] == report['raw_summary_sha256']
          == budget['summary_sha256'] == sha(directory/'summary.json')
          and raw['audit_guard_sha256'] == report['raw_guard_sha256']
          == budget['raw_guard_sha256'] == sha(directory/'raw-guard.json'),
          'raw publication hashes differ')
    check(complete['shared_analysis_seconds'] == report['shared_budget_seconds']
          == budget['shared_analysis_budget_seconds'] == 10800
          and 0 < complete['elapsed_seconds'] < 10800
          and 0 < report['total_accounted_seconds'] < 10800
          and report['previous_analysis_seconds'] == budget['previous_analysis_seconds']
          and report['total_accounted_seconds'] == report['previous_analysis_seconds']+report['science_seconds']
          and budget['automatic_retry'] is False, 'analysis budget provenance differs')
    rows = report['stages']
    check([(r['stage'], r['state']) for r in rows]
          == [(s, state) for s in STAGES for state in ['started', 'complete']],
          'stage execution ledger is incomplete')
    check(all(0 < rows[i+1]['seconds'] <= rows[i]['timeout_seconds']
              for i in range(0, len(rows), 2)), 'stage deadline exceeded')
    for stage, name in REPORTS.items():
        path = directory/stage/'catalog.json'
        check(sha(path) == report['catalogs'][stage], 'stage catalog digest differs')
        catalog = read(path)
        row = catalog[name]
        p = directory/stage/name
        check(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'],
              'stage report differs from validated catalog')
    return required


def run(evidence, t7, seed):
    label = label_for(seed)
    paths = analysis_paths(seed)
    source = t7/'artifacts'/paths['output'].name
    if not (source/'controller-complete.json').exists():
        return dict(ready=False, completion_published=False,
                    reason='Terminal raw audit and all six scientific stages are required')
    check(protocol_gate(evidence, seed) is not None, 'diagnostic protocol gate closed')
    gate = prerequisites(t7, reference_seed=seed)
    check(gate is not None, 'audited resource and collection records required')
    required = validate_analysis(source, gate, seed)
    root = evidence/CAMPAIGN/f'seed{seed}'
    backup = t7/'artifacts'/CAMPAIGN/f'seed{seed}'
    check(Path('/Volumes/T7').is_mount() and backup.resolve().is_relative_to(Path('/Volumes/T7').resolve()),
          'mounted T7 backup required')
    # The finished controller has already synchronized immutable JSON records.
    # Never copy a growing log, fill a missing report, or modify a failed run.
    for name in required:
        check((root/'analysis'/name).read_bytes() == (source/name).read_bytes(),
              'local terminal analysis evidence differs: '+name)
    resource = root/'resources/report.json'
    check(resource.read_bytes() == (gate['controls']/'report.json').read_bytes()
          and sha(resource) == gate['resource_report_sha256'], 'local resource report differs')
    reports = dict(resources=resource, raw=root/'analysis/summary.json',
                   analysis=root/'analysis/report.json')
    result = dict(schema='b2-mam-full-native-diagnostic-completion-v1',
        label=label, seed=seed, protocol_sha256=PROTOCOL_SHA,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        terminal_resource_audit_passed=True, raw_output_audit_passed=True,
        analysis_complete=True, scientific_acceptance=False, performance_cost_acceptance=False,
        reports={n:dict(path=str(p.relative_to(root)), sha256=sha(p)) for n,p in reports.items()},
        analysis_controller_sha256=sha(source/'controller-complete.json'),
        science_guard_sha256=sha(source/'science-guard.json'),
        collection_report_sha256=gate['collection_report_sha256'],
        remote_artifact_hashes_verified_by_guarded_pipeline=True,
        independent_raw_backup=False, automatic_next_run=False,
        scope='Finite diagnostic execution and artifact validation; not scientific equivalence or cost acceptance')
    targets = [backup/'completion.json', root/'completion.json']
    check(all(p.parent.is_dir() and not p.exists() for p in targets),
          'completion already exists or parent missing')
    payload = (json.dumps(result, indent=2)+'\n').encode()
    for target in targets:
        with target.open('xb') as stream: stream.write(payload)
        check(target.read_bytes() == payload, 'completion write differs')
    return dict(ready=True, completion_published=True, seed=seed, label=label,
                scientific_acceptance=False, automatic_next_run=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--t7', type=Path, required=True)
    p.add_argument('--reference-seed', type=int, choices=[1730,1731], required=True)
    args = p.parse_args()
    result = run(args.evidence, args.t7, args.reference_seed)
    print(json.dumps(result))
    raise SystemExit(0 if result['ready'] else 2)
