"""Admit the single selected NEST 100.5-second performance target.

Selection, sources, model, recording and resource limits remain frozen. This
entry point cannot run another tuning case, retry, or launch the Rust target.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time

import mam_launch_native_primary as old
import mam_launch_performance_tuning as tuning

SELECTION_SHA = '21e42afc0d765f409f651f57f76a1f4636ed9f343d93b3548d4801b9022c2084'
CASE_ID = 'nest-selected-target'


def selected_layout(evidence):
    root = evidence/'performance-runs-v1'
    old.check(old.sha(root/'selection.json') == SELECTION_SHA, 'frozen selection changed')
    selection = old.read(root/'selection.json')
    old.check(selection['protocol_sha256'] == tuning.PROTOCOL_SHA
              and selection['finite_tuning_complete'] is True, 'finite selection missing')
    walls = []
    for case_id, selected in zip(['nest48x4-tuning','nest96x2-tuning'], selection['cases'], strict=True):
        directory = root/case_id
        old.check(selected['case_id'] == case_id
                  and old.sha(directory/'completion.json') == selected['completion_sha256'], 'pilot completion changed')
        completion = old.read(directory/'completion.json')
        old.check(completion['terminal_resource_audit_passed'] is True
                  and completion['raw_output_audit_passed'] is True
                  and completion['protocol_sha256'] == tuning.PROTOCOL_SHA, 'pilot not accepted')
        for name, digest in completion['input_sha256'].items():
            old.check(old.sha(directory/name) == digest, 'pilot input changed: '+name)
        terminal = old.read(directory/'terminal/report.json')
        for item in terminal['collection_files']:
            old.check(old.sha(directory/'terminal'/item['file']) == item['sha256'], 'collected pilot controls changed')
        old.check(terminal['launch_wall_seconds'] == selected['launch_wall_seconds'], 'selection wall mismatch')
        walls.append(terminal['launch_wall_seconds'])
    chosen = 'nest96x2' if walls[1] < .95*walls[0] else 'nest48x4'
    old.check(chosen == selection['selected_layout'], 'selection rule differs')
    return chosen


def options_for(protocol, selected, output):
    pilot, old_label, options = tuning.options_for(protocol, selected+'-tuning', output)
    case = dict(next(c for c in protocol['cases'] if c['id'] == CASE_ID))
    old.check(case['purpose'] == 'target' and case['duration_ms'] == 100500
              and case['wall_seconds'] == 54000 and case['max_spikes_per_rank'] == 268435456,
              'target observation or budget changed')
    case.update(layout=selected, ranks=pilot['ranks'], threads=pilot['threads'])
    label = old_label.replace('-2500ms', '-100500ms')
    app = [value.replace(old_label, label) for value in options['application']]
    app[app.index('--duration-ms')+1] = '100500'
    app[app.index('--max-spikes-per-rank')+1] = str(case['max_spikes_per_rank'])
    options.update(application=app, timeout=case['wall_seconds'])
    return case, label, options


def admission_for(protocol, case, label, layout, options, checks):
    result = tuning.admission_for(protocol, case, label, layout, options, checks)
    result['identity']['duration_ms'] = case['duration_ms']
    result['limits']['wall_seconds'] = case['wall_seconds']
    result['selection_sha256'] = SELECTION_SHA
    return result


def run(args):
    phase = args.evidence/'performance-protocol-v1'
    old.check(old.sha(phase/'protocol.json') == tuning.PROTOCOL_SHA, 'frozen protocol differs')
    protocol = old.read(phase/'protocol.json')
    chosen = selected_layout(args.evidence)
    case, label, options = options_for(protocol, chosen, args.t7/'logs'/CASE_ID)
    staged = old.read(phase/'staging.json')
    old.check(staged['all_hosts_staged'] and [r['host'] for r in staged['rows']] == old.NODES,
              'source host coverage')
    for name, item in protocol['sources'].items():
        path = Path(__file__).parent/name
        old.check(path.stat().st_size == item['bytes'] and old.sha(path) == item['sha256'], 'source changed: '+name)
    layout_path = phase/(chosen+'-layout.json')
    old.check(old.sha(layout_path) == protocol['native']['layouts'][chosen]['sha256'], 'layout changed')
    layout = old.read(layout_path)
    old.check(Path('/Volumes/T7').is_mount() and args.t7.resolve().is_relative_to(Path('/Volumes/T7')), 'T7 not mounted')
    s = os.statvfs(args.t7)
    old.check(s.f_bavail*s.f_frsize >= 512*2**30, 'T7 start reserve')
    out = args.evidence/'performance-runs-v1'/CASE_ID
    old.check(not options['output'].exists(), 'target log directory already exists')
    out.mkdir(exist_ok=False)
    tuning.write(out/'intent.json', dict(case_id=CASE_ID, protocol_sha256=tuning.PROTOCOL_SHA,
        selection_sha256=SELECTION_SHA, maximum_neural_runs=1, automatic_retry=False,
        launch_options={k:str(v) if isinstance(v,Path) else v for k,v in options.items()}))
    inventory = old.read(args.evidence/'primary-native-layout/inventory.json')['nodes']
    leader_code = '''import os,json,subprocess,datetime
from pathlib import Path
assert os.uname().nodename=='hk-prod-model-ae02-23'
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5)
assert not units.strip(),units
p=Path('/data/brick2');s=os.statvfs(p);free=s.f_bavail*s.f_frsize
assert os.stat(p).st_dev!=os.stat('/').st_dev and free>=2048*2**30
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,active_own_units=[])))'''
    def check_host(index):
        code = old.preflight_code(index, staged['rows'][index]['result'],
            layout['hosts'][index]['selected_topology'], inventory[index]['mpi_sha256'], output_label=label)
        code = code.replace(old.PROJECT, protocol['native']['project'])
        return old.remote(old.NODES[index], 'import resource,signal\nsignal.alarm(40)\nresource.setrlimit(resource.RLIMIT_AS,(768*2**20,768*2**20))\nresource.setrlimit(resource.RLIMIT_CPU,(30,30))\n'+code)
    start = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
            leader = pool.submit(old.remote, 'hk-prod-model-ae02-23', leader_code)
            futures = [pool.submit(check_host,i) for i in range(6)]
            checks = dict(node23=leader.result(), native=[f.result() for f in futures])
        old.check(time.monotonic()-start < 60, 'preflight stale')
        admission = admission_for(protocol, case, label, layout, options, checks)
        admission['launcher_sources'] = {n:old.sha(Path(__file__).parent/n) for n in
            ['mam_launch_performance_target.py','mam_launch_performance_tuning.py','mam_launch_native_primary.py','mpi_teleport_launch.py']}
        tuning.write(out/'admission.json', admission)
        tuning.write(args.t7/'artifacts/performance-protocol-v1'/(CASE_ID+'-admission.json'), admission)
        old.check(time.monotonic()-start < 60, 'preflight expired before launch')
        print(json.dumps(dict(event='admitted',case_id=CASE_ID,label=label,selected_layout=chosen,
            admission_sha256=old.sha(out/'admission.json'),duration_ms=100500,wall_limit_seconds=54000)),flush=True)
        from mpi_teleport_launch import launch
        result = launch(**options)
        tuning.write(out/'launch.json',result)
        print(json.dumps(dict(event='launcher_terminal',case_id=CASE_ID,wall_seconds=result['wall_seconds'],audits_required=True)),flush=True)
    except BaseException as error:
        if (options['output']/'launch.json').exists() and not (out/'launch.json').exists():
            tuning.write(out/'launch.json',old.read(options['output']/'launch.json'))
        tuning.write(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--t7',type=Path,required=True)
    run(parser.parse_args())
