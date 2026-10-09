"""Read-only host availability snapshot; not a replacement for launch admission."""
from pathlib import Path
import concurrent.futures
import json
import time
import control as c

HERE = Path(__file__).resolve().parent
c.configure(30)
code = '''from pathlib import Path
import json,os
keys=['MemAvailable','MemTotal','AnonPages','Cached','SwapFree']
m={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in keys}
print(json.dumps({'host':os.uname().nodename,'memory':m}))'''
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(lambda node: c.remote(node, code), c.NODES))
for row, cap in zip(rows, c.resource_caps('weak1pct-h30-v2', 30)):
    row.update(requested_cap_gib=cap, reserve_gib=64,
               available_gib=row['memory']['MemAvailable']/2**30)
record = dict(utc_epoch=time.time(), rows=rows,
              scope='Read-only memory snapshot; active pilot/preparation may affect availability. Launch performs separate full admission.')
(HERE / f'resource-snapshot-{int(record["utc_epoch"])}.json').write_text(json.dumps(record, indent=2)+'\n')
below = [dict(host=r['host'],available_gib=round(r['available_gib'],2),required_gib=r['requested_cap_gib']+64)
         for r in rows if r['available_gib'] <= r['requested_cap_gib']+64]
print(json.dumps(dict(below_threshold=below, all_memory_thresholds_met=not below)))
