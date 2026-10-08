"""Compare integer-tick rate bins to an extracted, pinned official helper."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from mam_paper_rate_bins import full_rate_bins

HELPER_SHA='82a3265e2752ef9ccaa2a211089529779e43a28cbdf75779629a58409c84fa8d'


def validate(source, output):
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=HELPER_SHA:
        raise ValueError('unverified official analysis helper')
    tree=ast.parse(raw)
    functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='pop_rate_time_series']
    assert len(functions)==1
    namespace={'np':np}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    reference=namespace['pop_rate_time_series']
    cases=[(np.array([],dtype=np.int64),7,25000),
           (np.array([4999,5000,5001,5004,5005,5006,5014,5015,24999,25000,25001,25005]),7,25000),
           (np.array([5005]*17+[5015]*13+[25000]*11),11,25000)]
    rng=np.random.default_rng(20260908)
    for i in range(40):
        end=5010+10*int(rng.integers(1,3000))
        cases.append((rng.integers(0,end+20,size=i*97,dtype=np.int64),int(rng.integers(1,100001)),end))
    for ticks,n,end in cases:
        times=ticks*.1
        keep=(times>500.) & (times<=end*.1)
        data=np.column_stack([np.zeros(np.count_nonzero(keep)),times[keep]])
        actual=full_rate_bins(ticks,n,end)
        expected=reference(data,n,500.,end*.1,resolution=1.)
        np.testing.assert_array_equal(actual,expected)
    output.write_text(json.dumps(dict(passed=True,fixture_cases=len(cases),all_rate_bins_exact=True,official_helper_sha256=HELPER_SHA,
        source=str(source),numpy=np.__version__,implementation_sha256=hashlib.sha256(Path(__file__).with_name('mam_paper_rate_bins.py').read_bytes()).hexdigest(),
        contract='Official wrapper (500,T] ms followed by helper histogram [500.5,T+0.5] ms, 1 ms bins, physical dt=0.1 ms.',
        scope='Bounded numerical helper fixtures only; full-run rebinning, PSD and historical environment fidelity remain open.'),indent=2)+'\n')
    print(output.read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();validate(a.source,a.output)
