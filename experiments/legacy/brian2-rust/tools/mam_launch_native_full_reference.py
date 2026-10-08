"""One admitted full-duration diagnostic NEST seed; no automatic second run.

The tested primary source/runtime/layout are reused without staging or edits.
Historical primary gates retain their default seed1729 identity. New run
identities are explicit, and reserved confirmation seeds are refused.
"""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import time

from mam_launch_native_primary import (
    BASE, PROJECT, NODES, PARAMETERS, LAYOUT_SHA, check, read, sha,
    launch_options, preflight_code, remote, rust_gate,
)

PROTOCOL_SHA = '8f45dce3c1863371e32b6e00b863ed6dfbf3ffc80e3032ceada6ceb0a775b18b'
CAMPAIGN = 'native-full-reference-v1'
BUILD = '/data/brick2/brian2-mpi-region-20260907'


def label_for(seed):
    check(type(seed) is int and seed in (1730, 1731), 'only diagnostic seeds1730/1731 are admitted')
    return f'nest-mam-full-reference-v1-metastable-seed{seed}-100500ms'


def protocol_gate(evidence, seed):
    label_for(seed)  # Refuse confirmation seeds before any file/network access.
    path = evidence/CAMPAIGN/'protocol.json'
    check(sha(path) == PROTOCOL_SHA, 'frozen diagnostic protocol changed')
    protocol = read(path)
    for row in protocol['pinned_inputs']:
        path = evidence.parent/row['path']
        check(path.resolve().is_relative_to(evidence.parent.resolve())
              and path.is_file() and not path.is_symlink()
              and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
              'pinned diagnostic input changed: '+row['path'])
    if seed == 1731:
        # A subsequent audited completion record will explicitly pin all three
        # first-run gates. A terminal launcher alone can never unlock seed1731.
        gate_path = evidence/CAMPAIGN/'seed1730/completion.json'
        if not gate_path.exists():
            return None
        gate = read(gate_path)
        check(gate['schema'] == 'b2-mam-full-native-diagnostic-completion-v1'
              and gate['label'] == label_for(1730) and gate['seed'] == 1730
              and gate['protocol_sha256'] == PROTOCOL_SHA
              and gate['terminal_resource_audit_passed'] is True
              and gate['raw_output_audit_passed'] is True and gate['analysis_complete'] is True,
              'first diagnostic seed is incomplete')
        check(set(gate['reports']) == {'resources', 'raw', 'analysis'}, 'first-run gate coverage')
        flags = {'resources':'terminal_resource_audit_passed',
                 'raw':'raw_output_audit_passed', 'analysis':'analysis_complete'}
        for name, row in gate['reports'].items():
            path = evidence/CAMPAIGN/'seed1730'/row['path']
            check(path.resolve().is_relative_to((evidence/CAMPAIGN/'seed1730').resolve())
                  and sha(path) == row['sha256'], 'first-run completion source changed')
            report = read(path)
            check(report['label'] == label_for(1730) and report[flags[name]] is True,
                  'first-run completion report identity or outcome differs')
            if name == 'analysis':
                required = ['activity','cell','correlation','series','fc','lags']
                check(report.get('required_stages') == required
                      and set(report.get('catalogs', {})) == set(required)
                      and report.get('interarea_analysis_complete') is True
                      and report.get('seed') == 1730 and report.get('protocol_sha256') == PROTOCOL_SHA,
                      'all six full-reference scientific summaries are required before seed1731')
    return protocol


def leader_preflight_code():
    return f'''import os,json,subprocess,datetime
from pathlib import Path
p=Path({BUILD!r})
assert os.uname().nodename=='hk-prod-model-ae02-23'
assert Path('/data/brick2').is_mount() and p.resolve().is_relative_to(Path('/data/brick2'))
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5)
assert not units.strip(),units
s=os.statvfs(p);free=s.f_bavail*s.f_frsize
assert free>=1792*2**30
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),active_own_units=[],volume='/data/brick2',free_bytes=free,campaign_additional_allowance_bytes=512*2**30,runtime_reserve_bytes=1280*2**30)))'''


def publish(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def run(evidence, t7, seed):
    label = label_for(seed)
    protocol = protocol_gate(evidence, seed)
    if protocol is None:
        return dict(ready=False, launch_started=False,
                    reason='seed1730 terminal resources, raw audit and complete analysis are required before seed1731')
    prerequisite = rust_gate(evidence, t7/'artifacts/primary-terminal-resources-v1')
    check(prerequisite is not None, 'completed Rust prerequisite missing')
    native = read(evidence/'primary-native-resources/terminal-report.json')
    analysis = read(evidence/'primary-native-postrun/run/report.json')
    raw_path = evidence/'primary-native-postrun/run/summary.json'
    raw = read(raw_path)
    check(native['terminal_resource_audit_passed'] is True
          and native['construction_ledger_passed'] is True
          and analysis['analysis_complete'] is True
          and raw['raw_output_audit_passed'] is True
          and sha(raw_path) == analysis['raw_summary_sha256'], 'completed primary NEST prerequisite missing')
    layout_dir = evidence/'primary-native-layout'
    staged = read(layout_dir/'stage.json')
    layout = read(layout_dir/'layout.json')
    inventory = read(layout_dir/'inventory.json')['nodes']
    check([r['host'] for r in staged] == [r['host'] for r in layout['hosts']]
          == [r['host'] for r in inventory] == NODES, 'six-host source/layout coverage')
    check(all(r['source_catalog']['mam_nest_reference.py']['sha256']
              == protocol['condition']['producer_sha256'] for r in staged), 'producer identity changed')
    smoke = read(layout_dir/'smoke-report.json')
    check(smoke['all_rank_event_bytes_exact'] is True
          and smoke['all_32_workers_on_distinct_single_cpus'] is True, 'binding validation missing')
    volume = Path('/Volumes/T7')
    check(volume.is_mount() and t7.resolve().is_relative_to(volume.resolve()), 'T7 is required')
    stat = os.statvfs(t7)
    check(stat.f_bavail*stat.f_frsize >= 128*2**30, 'T7 reserve')
    directory = evidence/CAMPAIGN/f'seed{seed}'
    output = t7/'logs'/label
    backup = t7/'artifacts'/CAMPAIGN/f'seed{seed}'
    check(not directory.exists() and not output.exists() and not backup.exists(),
          'diagnostic experiment already admitted or attempted; no retry')
    started = time.monotonic()
    # Seven bounded read-only checks: collect every outcome before admission.
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        futures = [pool.submit(remote, NODES[i], preflight_code(i, staged[i],
            layout['hosts'][i]['selected_topology'], inventory[i]['mpi_sha256'],
            output_label=label)) for i in range(6)]
        leader_future = pool.submit(remote, 'hk-prod-model-ae02-23', leader_preflight_code())
        rows = []; failures = []
        for index, future in enumerate(futures):
            try: rows.append(future.result())
            except Exception as exc: failures.append(dict(host=NODES[index], error=str(exc)))
        try: leader = leader_future.result()
        except Exception as exc: failures.append(dict(host='hk-prod-model-ae02-23', error=str(exc)))
    check(not failures, 'fresh resource admission failed: '+json.dumps(failures))
    check(time.monotonic()-started < 60 and [r['host'] for r in rows] == NODES,
          'resource admission stale or incomplete')
    options = launch_options(output, label=label, seed=seed)
    sources = {name: sha(Path(__file__).parent/name) for name in [
        'mam_launch_native_full_reference.py', 'mam_launch_native_primary.py',
        'mam_primary_resources.py', 'mpi_teleport_launch.py',
        'mam_native_primary_resources.py', 'mam_native_rank_audit.py',
        'mam_collect_native_primary_resources.py']}
    admission = dict(schema='b2-mam-native-primary-admission-v1', label=label,
        seed=seed, campaign=CAMPAIGN, protocol_sha256=PROTOCOL_SHA,
        admitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prerequisite=prerequisite, rust_leader_terminal=dict(rust_primary_active_units=[]),
        collection_host_preflight=leader, preflight=rows,
        layout_sha256=LAYOUT_SHA, parameters_sha256=PARAMETERS,
        source_project=PROJECT, source_project_is_retained_primary=True,
        controller_source_sha256=sources,
        experiment_budget=dict(runs=1, automatic_retry=False, wall_seconds=54000,
            collection_seconds=7200, analysis_stage_seconds=10800),
        launch_options={k:str(v) if isinstance(v,Path) else v for k,v in options.items()},
        admitted=True, scientific_acceptance=False, performance_cost_acceptance=False)
    directory.mkdir(exist_ok=False)
    backup.mkdir(parents=True, exist_ok=False)
    publish(directory/'admission.json', admission)
    publish(backup/'admission.json', admission)
    publish(backup/'protocol.json', protocol)
    (backup/'controller-sources').mkdir()
    for name, digest in sources.items():
        source = Path(__file__).parent/name
        check(sha(source) == digest, 'controller source changed before launch')
        with (backup/'controller-sources'/name).open('xb') as stream:
            stream.write(source.read_bytes())
    print(json.dumps(dict(event='native_full_diagnostic_admitted', label=label,
        nodes=NODES, wall_limit_seconds=54000, automatic_next_run=False)), flush=True)
    from mpi_teleport_launch import launch
    try:
        result = launch(**options)
        publish(directory/'launch.json', result)
        publish(backup/'launch.json', result)
        from mam_native_primary_resources import terminal_launch
        terminal_launch(admission, result, label=label, seed=seed)
    except BaseException as exc:
        failure = dict(label=label, error=str(exc), automatic_retry=False,
                       remote_terminal_state_must_be_verified=True)
        publish(directory/'failure.json', failure)
        publish(backup/'failure.json', failure)
        raise
    return dict(ready=True, launch_started=True, launcher_terminal_success=True,
        label=label, raw_resource_scientific_audits_required=True, automatic_next_run=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--t7', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[1730,1731], required=True)
    args = parser.parse_args()
    result = run(args.evidence, args.t7, args.seed)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['ready'] else 2)
