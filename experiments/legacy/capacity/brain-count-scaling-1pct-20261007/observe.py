"""Compact read-only view of the active sweep's local observations."""
from pathlib import Path
import datetime
import json
import time

HERE = Path(__file__).resolve().parent
state = json.loads((HERE/'sweep-state.json').read_text())
record = {k:state.get(k) for k in ['status','action','case','hosts','accepted_hosts']}
record['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
record['observer_age_seconds'] = round(time.time()-state['last_update_epoch'],1)
if state.get('error'):
    record['error'] = state['error']
stage = HERE / (str(state.get('case'))+'-status.json')
if stage.exists() and state.get('action') == 'status':
    data = json.loads(stage.read_text())
    guard = data.get('guard') or {}
    live = data.get('live') or guard.get('after') or {}
    record.update(current_gib=int(live.get('memory.current',0))/2**30,
                  peak_gib=int(live.get('memory.peak',0))/2**30,
                  memory_events=live.get('memory.events'),
                  terminal_returncode=guard.get('returncode'),
                  stage_log=data.get('log','')[-800:])
    cpu = dict(line.split() for line in live.get('cpu.stat','').splitlines())
    record['stage_cpu_seconds'] = int(cpu.get('usage_usec',0))/1e6
if state.get('action') == 'launch':
    launch_dir = HERE / (state['case']+'-launch')
    record['proxy_log_files'] = len(list(launch_dir.glob('proxy-*.log')))
print(json.dumps(record))
