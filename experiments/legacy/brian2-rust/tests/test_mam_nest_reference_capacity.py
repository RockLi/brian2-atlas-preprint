"""Prevent the observed native NEST connection-index failure before allocation."""
import argparse
import builtins
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mam_nest_reference_capacity', ROOT / 'tools/mam_nest_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def test_full_archived_model_rejected_before_nest_import(tmp_path, monkeypatch):
    raw = gzip.decompress((ROOT / 'mpi-evidence/region/full-parameters.json.gz').read_bytes())
    parameter = tmp_path / 'parameters.json'
    parameter.write_bytes(raw)
    args = argparse.Namespace(parameters=parameter, parameters_sha256=hashlib.sha256(raw).hexdigest(),
                              max_neurons=5000000, max_edges=25000000000, ranks=32, threads=1)
    original = builtins.__import__

    def reject_nest(name, *a, **kw):
        if name == 'nest':
            raise AssertionError('NEST import attempted for an impossible connection layout')
        return original(name, *a, **kw)

    monkeypatch.setattr(builtins, '__import__', reject_nest)
    with pytest.raises(ValueError, match='need at least 180 virtual processes; 32 requested'):
        reference.run(args)
    assert reference.virtual_process_capacity(json.loads(raw), 32, 6)['minimum_virtual_processes'] == 180


def test_observed_failure_and_capacity_boundary():
    p = dict(total_neurons=825885, total_recurrent_synapses=1930117543)
    with pytest.raises(ValueError, match='need at least 15 virtual processes; 4 requested'):
        reference.virtual_process_capacity(p, 4, 1)
    assert reference.virtual_process_capacity(p, 4, 4)['virtual_processes'] == 16
    p = dict(total_neurons=1, total_recurrent_synapses=134217724)
    assert reference.virtual_process_capacity(p, 1, 1)['static_connections'] == 134217726
    p['total_recurrent_synapses'] += 1
    with pytest.raises(ValueError, match='need at least 2 virtual processes'):
        reference.virtual_process_capacity(p, 1, 1)


@pytest.mark.parametrize('duration,accepted',[('2500',True),('10500',True),('10500.1',False),('10499.95',False),('nan',False)])
def test_long_cli_admission_preserves_grid_and_ceiling(tmp_path,monkeypatch,duration,accepted):
    import runpy,sys
    raw=gzip.decompress((ROOT/'mpi-evidence/region/full-parameters.json.gz').read_bytes())
    parameter=tmp_path/'parameters.json';parameter.write_bytes(raw)
    monkeypatch.setattr(sys,'argv',['mam_nest_reference.py','--parameters',str(parameter),
        '--parameters-sha256',hashlib.sha256(raw).hexdigest(),'--output',str(tmp_path/'run'),
        '--ranks','48','--threads','4','--max-neurons','4200000','--max-edges','25000000000',
        '--duration-ms',duration,'--chunk-ms','50'])
    class ReachedAdmittedKernelImport(Exception):pass
    original=builtins.__import__
    def stop_kernel(name,*args,**kwargs):
        if name=='nest':raise ReachedAdmittedKernelImport
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',stop_kernel)
    with pytest.raises(ReachedAdmittedKernelImport if accepted else SystemExit) as result:
        runpy.run_path(str(ROOT/'tools/mam_nest_reference.py'),run_name='__main__')
    if not accepted:assert result.value.code==2
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('duration,limit,accepted',[
    ('100500',None,False),('100500','100500',True),('100500.1','100500',False),
    ('100499.95','100500',False),('inf','100500',False),('nan','100500',False),
    ('100500','100501',False)])
def test_primary_duration_needs_explicit_ceiling_and_preserves_pre_kernel_checks(tmp_path,monkeypatch,duration,limit,accepted):
    import runpy,sys
    raw=gzip.decompress((ROOT/'mpi-evidence/region/full-parameters.json.gz').read_bytes())
    parameter=tmp_path/'parameters.json';parameter.write_bytes(raw)
    argv=['mam_nest_reference.py','--parameters',str(parameter),
          '--parameters-sha256',hashlib.sha256(raw).hexdigest(),'--output',str(tmp_path/'run'),
          '--ranks','48','--threads','4','--max-neurons','4200000','--max-edges','25000000000',
          '--duration-ms',duration,'--chunk-ms','50','--max-chunk-spikes','2000000',
          '--max-spikes-per-rank','268435456']
    if limit is not None:argv+=['--max-duration-ms',limit]
    monkeypatch.setattr(sys,'argv',argv)
    class ReachedAdmittedKernelImport(Exception):pass
    original=builtins.__import__
    def stop_kernel(name,*args,**kwargs):
        if name=='nest':raise ReachedAdmittedKernelImport
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',stop_kernel)
    with pytest.raises(ReachedAdmittedKernelImport if accepted else SystemExit) as result:
        runpy.run_path(str(ROOT/'tools/mam_nest_reference.py'),run_name='__main__')
    if not accepted:assert result.value.code==2
    assert not (tmp_path/'run').exists()
