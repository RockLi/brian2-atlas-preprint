import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from mam_benchmark_raw_audit import scan_rank,host_code


def fixture(tmp_path):
    # More than one 64 KiB read, a delayed recorder event, and the final tick.
    blocks=[[(499,47)]*8200,[(450,47)]]+[[] for _ in range(2007)]+[[(1005000,47)]]
    path=tmp_path/'events.bin'
    path.write_bytes(np.array([r for block in blocks for r in block],dtype=[('tick','<u4'),('cell','<u4')]).tobytes())
    progress=[];chunks=[];memory={};total=0
    for i,block in enumerate(blocks):
        end=(i+1)*50;total+=len(block);mem={'VmRSS':1024};memory[f'after_{end}ms']=mem
        chunk=dict(end_ms=end,spikes=len(block),simulation_seconds=1.,output_seconds=.01)
        chunks.append(chunk);progress.append(dict(**chunk,event='chunk_complete',rank=0,total_spikes=total,event_bytes=8*total,memory=mem))
    return dict(schema='b2-native-nest-mam-benchmark-events-v2',rank=0,ranks=48,duration_ms=100500,
                event_bytes=path.stat().st_size,event_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                local_spikes=total,chunks=chunks,phase_memory=memory),progress,path


def test_full_window_and_multiple_read_blocks(tmp_path):
    report,progress,path=fixture(tmp_path)
    r=scan_rank(report,progress,path,duration_ms=100500,release_cache=False)
    assert len(r['physical_50ms_bin_counts'])==2010
    assert r['physical_50ms_bin_counts'][0]==8201 and r['terminal_tick_events']==1
    assert r['spikes']==8202 and r['maximum_buffer_bytes']==65536


@pytest.mark.parametrize('fault,match',[('missing_chunk','chunk coverage'),('late_tick','physical domain'),('unselected_layout','recording identity')])
def test_target_rejects_incomplete_or_wrong_records(tmp_path,fault,match):
    report,progress,path=fixture(tmp_path)
    if fault=='missing_chunk':progress.pop()
    elif fault=='unselected_layout':report['ranks']=96
    else:
        raw=bytearray(path.read_bytes());raw[-8:-4]=(1005001).to_bytes(4,'little');path.write_bytes(raw)
    with pytest.raises(ValueError,match=match):scan_rank(report,progress,path,duration_ms=100500,release_cache=False)


def test_remote_scan_uses_full_target_budget_and_refuses_implicit_pilot():
    admission=json.loads((ROOT/'mpi-evidence/performance-runs-v1/nest-selected-target/admission.json').read_text())
    terminal={'ranks':[dict(rank=i,report_sha256='0'*64) for i in range(48)]}
    for i in range(6):
        code=host_code(admission,terminal,i,duration_ms=100500);ast.parse(code)
        assert 'signal.alarm(900)' in code and 'duration_ms=100500' in code
        assert 'resource.RLIMIT_CPU,(600,600)' in code
    with pytest.raises(ValueError,match='admitted duration'):host_code(admission,terminal,0)


def test_missing_terminal_controls_never_start_raw_reads(tmp_path,monkeypatch):
    import mam_benchmark_raw_audit as module
    raw=(ROOT/'mpi-evidence/performance-runs-v1/nest-selected-target/admission.json').read_bytes()
    (tmp_path/'admission.json').write_bytes(raw)
    def reject(*args,**kwargs):raise AssertionError('network must not run')
    monkeypatch.setattr(module.subprocess,'run',reject)
    with pytest.raises(FileNotFoundError):module.run(SimpleNamespace(case=tmp_path))
    assert not (tmp_path/'raw').exists()
