"""Compare two bounded metrics against inspected, pinned official functions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_mam_paper_cell_metrics import cell_metrics

SOURCE_SHA = '82a3265e2752ef9ccaa2a211089529779e43a28cbdf75779629a58409c84fa8d'
COMMIT = '0a658be40bef3249cbe452f38809edf7d2f524ba'


def validate(source, output):
    raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    tree=ast.parse(raw)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['pop_rate','pop_LvR']]
    assert len(functions)==2
    # Only the two reviewed, hash-pinned pure numerical definitions run. No imports,
    # model construction, filesystem or module initialization from upstream.
    ns={'np':np}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),ns)
    cases=[(np.array([],dtype=int),np.array([],dtype=int),4),
           (np.array([4999,5000,5010,5030,24999,25000]),np.array([0,0,0,0,1,0]),5),
           (np.array([5000,5020,5040,5060]),np.array([0,0,0,0]),1)]
    rng=np.random.default_rng(98413)
    for _ in range(100):
        neurons=int(rng.integers(1,25));ticks=[];ids=[]
        for cell in range(neurons):
            train=rng.choice(np.arange(4990,25010),int(rng.integers(0,80)),replace=False)
            ticks.extend(train);ids.extend([cell]*len(train))
        cases.append((np.array(ticks,dtype=int),np.array(ids,dtype=int),neurons))
    errors=[]
    for ticks,ids,n in cases:
        actual,arr=cell_metrics(ticks,ids,n)
        order=np.argsort(ticks,kind='stable');data=np.column_stack([ids[order]+11,ticks[order]*.1])
        rate=ns['pop_rate'](data,500.,2500.,n)
        lvr, values=ns['pop_LvR'](data,2.,500.,2500.,n)
        np.testing.assert_allclose(actual['paper_rate_hz'],rate,rtol=1e-14,atol=0)
        np.testing.assert_allclose(actual['paper_lvr_mean'],lvr,rtol=1e-12,atol=1e-12)
        # Upstream places active IDs first and appends silent zeros; compare the
        # whole value multiset, not a falsely paired physical-ID vector.
        np.testing.assert_allclose(np.sort(arr['cell_lvr']),np.sort(values),rtol=1e-12,atol=1e-12)
        errors.append(abs(actual['paper_lvr_mean']-float(lvr)))
    result=dict(passed=True,cases=len(cases),max_lvr_mean_abs_error=max(errors),
        official_commit=COMMIT,official_source_sha256=SOURCE_SHA,
        source_url=f'https://github.com/INM-6/multi-area-model/blob/{COMMIT}/multiarea_model/analysis_helpers.py',
        checked_functions=['pop_rate(return_stat=False)','pop_LvR'],
        caveat='Small fixture algorithm check; not full model, historical environment or other paper observable acceptance.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();validate(a.source,a.output)
