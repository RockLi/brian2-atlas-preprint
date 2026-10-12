"""Retain each real skip's backend parameters and source location for audit."""
import json,os
from pathlib import Path
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item,call):
    outcome=yield;report=outcome.get_result()
    if not report.skipped:return
    path=os.environ.get('B2_SKIP_EVIDENCE')
    if not path:return
    params=getattr(getattr(item,'callspec',None),'params',{})
    devices={k:v for k,v in params.items() if isinstance(v,str) and v in ('cpu','metal','cuda','mpi') and any(t in k.lower() for t in ('backend','engine','device'))}
    location=report.longrepr
    if not isinstance(location,tuple):location=(str(item.path),item.location[1],str(location))
    guards=[]
    if call.excinfo is not None:
        for entry in call.excinfo.traceback:
            source=str(entry.path)
            if 'brian2-rust/tests/' in source:guards.append([source,entry.lineno+1,str(location[2])])
    row=dict(nodeid=item.nodeid,phase=report.when,devices=devices,location=list(location),guard_location=guards[-1] if guards else None,mode=os.environ['B2_FULL_MODE'],runtime_flags={key:os.environ.get(key) for key in ('B2_TEST_GPU','B2_TEST_CUDA','B2_TEST_CUDA_TRAIN','B2_TEST_MPI')})
    with Path(path).open('a') as output:output.write(json.dumps(row)+'\n')
