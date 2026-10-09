"""Read-only bounded observer of an already launched run; never starts jobs."""
from pathlib import Path
import argparse, importlib.util, json, time

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(); p.add_argument('case'); a = p.parse_args()
spec = importlib.util.spec_from_file_location('capacity_control', HERE / 'control.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
started = time.monotonic()
while time.monotonic() - started < 2100:
    c.monitor()
    latest = sorted(HERE.glob('monitor-*.json'))[-1]
    rows = json.loads(latest.read_text())
    workers = [g for r in rows for g in r['active_guards'] if '-proxy-' in g['guard']]
    peaks = [int(g['live']['memory.peak']) / 2**30 for g in workers]
    print(json.dumps({'snapshot': latest.name, 'active_worker_hosts': len(workers),
                      'peak_gib_minmax': [min(peaks), max(peaks)] if peaks else None,
                      'oom_seen': any('oom 0' not in g['live']['memory.events'] for g in workers),
                      'launch_finished': (HERE / (a.case + '-launch-result.json')).exists()}), flush=True)
    if (HERE / (a.case + '-launch-result.json')).exists(): break
    time.sleep(35)
