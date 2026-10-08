import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from mam_benchmark_raw_audit import scan_rank


def fixture(tmp_path, ranks=48):
    # A later recorder drain can contain an earlier physical time.
    blocks = [[(499, ranks-1)], [(450, ranks-1)]]+[[] for _ in range(47)]+[[(25000,ranks-1)]]
    events = np.array([event for block in blocks for event in block],dtype=[('tick','<u4'),('cell','<u4')])
    path=tmp_path/'events.bin';path.write_bytes(events.tobytes())
    chunks=[];progress=[];memory={};total=0
    for i, block in enumerate(blocks):
        end=(i+1)*50;total+=len(block);mem={'VmRSS':1024};memory[f'after_{end}ms']=mem
        chunk=dict(end_ms=end,spikes=len(block),simulation_seconds=1.,output_seconds=.01)
        chunks.append(chunk)
        progress.append(dict(event='chunk_complete',rank=0,**chunk,total_spikes=total,event_bytes=8*total,memory=mem))
    report=dict(schema='b2-native-nest-mam-benchmark-events-v2',rank=0,ranks=ranks,duration_ms=2500,
                event_bytes=path.stat().st_size,event_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                local_spikes=total,chunks=chunks,phase_memory=memory)
    return report,progress,path


@pytest.mark.parametrize('ranks',[48,96])
def test_delayed_drain_and_terminal_tick(ranks,tmp_path):
    report,progress,path=fixture(tmp_path,ranks)
    result=scan_rank(report,progress,path,release_cache=False)
    assert result['physical_50ms_bin_counts'][0]==2
    assert result['terminal_tick_events']==1
    assert result['spikes']==3


@pytest.mark.parametrize('fault,match',[('owner','rank ownership'),('time','physical domain'),
                                      ('truncated','size or path'),('digest','checksum'),('progress','count ledger')])
def test_corruptions_fail(fault,match,tmp_path):
    report,progress,path=fixture(tmp_path)
    if fault in ['owner','time']:
        data=bytearray(path.read_bytes());start=4 if fault=='owner' else 0
        data[start:start+4]=(48 if fault=='owner' else 501).to_bytes(4,'little');path.write_bytes(data)
    elif fault=='truncated':path.write_bytes(path.read_bytes()[:-1])
    elif fault=='digest':report['event_sha256']='0'*64
    else:progress[1]['total_spikes']+=1
    with pytest.raises(ValueError,match=match):scan_rank(report,progress,path,release_cache=False)
