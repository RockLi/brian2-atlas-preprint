"""Read-only compact observer for the active bounded sweep."""
from pathlib import Path
import json,time
import control as c
p=Path(__file__).resolve().parent
s=json.loads((p/'sweep-state.json').read_text());c.configure(s['hosts'])
case=s['case'] or ''
for prefix in ['prepare-','reference-','audit-']:
    if case.startswith(prefix):
        case=case.removeprefix(prefix)
        break
code=f'''from pathlib import Path
import json
p=Path({c.BASE!r})/{case!r}
ranks=list((p/'mpi').glob('instance.rank-*.bin'))
print(json.dumps({{'rank_input_files':len(ranks),'rank_input_bytes':sum(f.stat().st_size for f in ranks),'compiled':(p/'mpi/b2-mpi').exists(),'full_result_report':(p/'result/mpi-runtime.json').exists(),'audit_present':(p/'audit.json').exists()}}))'''
r=c.remote(c.NODES[0],code)
print(json.dumps(dict(state={k:s[k] for k in ['status','action','case','hosts','accepted_hosts']},progress=r)))
