"""Sparse 2010-chunk fixtures, not synthetic full-model observations."""
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import mam_native_rank_audit as m


def recording(root,name,duration,*,end_boundary=False):
    # Rank zero owns zero-based cells 47,95,... . A recorder may drain a
    # boundary event in a later chunk; the physical histogram must not confuse
    # drain times with spike times.
    chunks=[];progress=[];memory={};values=[];total=0
    for i in range(duration//50):
        if i==0:events=[(1,47),(499,95)]
        elif i==1:events=[(500,47)]
        elif i==duration//50-1 and end_boundary:events=[(duration*10-1,95),(duration*10,47)]
        else:events=[]
        values.extend(events);count=len(events);total+=count;end=(i+1)*50
        c=dict(end_ms=end,spikes=count,simulation_seconds=.1,output_seconds=.01)
        chunks.append(c);memory[f'after_{end}ms']=dict(rss_kib=100)
        progress.append(dict(event='chunk_complete',rank=0,**c,total_spikes=total,event_bytes=total*8,memory=memory[f'after_{end}ms']))
    a=np.array(values,dtype=m.DTYPE);path=root/(name+'.bin');a.tofile(path)
    report=dict(schema='b2-native-nest-mam-v1',rank=0,ranks=48,threads=4,seed=1729,dt_ms=.1,duration_ms=duration,
        nest_version='3.10.0',parameters_sha256='fixture',allowed_cpus=[0],N_scaling=1.,K_scaling=1.,
        populations=[dict(name='fixture',count=96,first_gid=1,cell_start=0,cell_end=96)],
        projection_local_counts=[4,5],local_recurrent_edges=9,local_external_connections=2,local_recording_connections=2,
        local_total_connections=13,local_spikes=total,event_bytes=path.stat().st_size,
        event_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),chunks=chunks,phase_memory=memory,
        min_delay_ms=.1,max_delay_ms=20.,connection_index_capacity={'fixture':True})
    return report,progress,path


@pytest.fixture
def inputs(tmp_path):
    primary,progress,path=recording(tmp_path,'primary',100500,end_boundary=True)
    old,_,prior=recording(tmp_path,'prior',10500)
    args=dict(rank=0,duration_ms=100500,neurons=96,parameters_sha256='fixture',allowed_cpus=[0],
              prefix_report=old,prefix_events=prior,prefix_allowed_cpus=[0],release_cache=False)
    return primary,progress,path,args


def test_full_physical_window_and_exact_prefix(inputs):
    report,progress,path,args=inputs
    result=m.audit_rank(report,progress,path,**args)
    assert result['chunks']==2010 and result['spikes']==5 and result['terminal_tick_events']==1
    assert len(result['physical_50ms_bin_counts'])==2010
    assert result['physical_50ms_bin_counts'][:2]==[2,1]
    assert result['physical_50ms_bin_counts'][-1]==1
    assert result['prefix_bytes']==24 and result['raw_byte_prefix_exact']
    assert not result['scientific_equivalence'] and not result['whole_cluster_resource_audit_passed']


@pytest.mark.parametrize('fault',['progress-count','progress-memory','chunk-time','missing-chunk',
    'wrong-rank','wrong-duration','wrong-cpu','changed-ledger','prefix-ledger','checksum','trailing-bytes'])
def test_bad_native_recording_cannot_pass(inputs,fault):
    report,progress,path,args=inputs
    if fault=='progress-count':progress[0]['total_spikes']+=1
    elif fault=='progress-memory':progress[0]['memory']={'rss_kib':101}
    elif fault=='chunk-time':report['chunks'][1000]['end_ms']-=50
    elif fault=='missing-chunk':report['chunks'].pop()
    elif fault=='wrong-rank':report['rank']=1
    elif fault=='wrong-duration':report['duration_ms']=10500
    elif fault=='wrong-cpu':report['allowed_cpus']=[1]
    elif fault=='changed-ledger':report['local_recurrent_edges']+=1
    elif fault=='prefix-ledger':args['prefix_report']['projection_local_counts']=[5,4]
    elif fault=='checksum':report['event_sha256']='0'*64
    else:
        with path.open('ab') as stream:stream.write(b'0')
    with pytest.raises(ValueError):m.audit_rank(report,progress,path,**args)


@pytest.mark.parametrize('fault',['ownership','physical-time','prefix-byte'])
def test_event_content_checked_even_when_producer_hash_is_updated(inputs,fault):
    report,progress,path,args=inputs
    a=np.fromfile(path,dtype=m.DTYPE)
    if fault=='ownership':a['cell'][-1]=46
    elif fault=='physical-time':a['tick'][-1]+=1
    else:a['tick'][0]=2
    a.tofile(path);report['event_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):m.audit_rank(report,progress,path,**args)


def test_stream_spans_multiple_bounded_reads(inputs,monkeypatch):
    report,progress,path,args=inputs
    # Use a small injected block to test boundary crossings without repeating
    # a large source recording or allocating a full simulated spike train.
    monkeypatch.setattr(m,'BLOCK',1)
    result=m.audit_rank(report,progress,path,**args)
    assert result['maximum_read_block_events']==1 and result['raw_byte_prefix_exact']


@pytest.mark.parametrize('seed',[1730,1731])
def test_full_reference_checks_own_short_prefix_and_refuses_other_seed(tmp_path,seed):
    report,progress,path=recording(tmp_path,'full',100500,end_boundary=True)
    prior,_,previous=recording(tmp_path,'short',2500)
    report['seed']=prior['seed']=seed
    args=dict(rank=0,duration_ms=100500,neurons=96,parameters_sha256='fixture',allowed_cpus=[0],
              prefix_report=prior,prefix_events=previous,prefix_allowed_cpus=[0],release_cache=False)
    with pytest.raises(ValueError):m.audit_rank(report,progress,path,**args)
    result=m.audit_rank(report,progress,path,**args,expected_seed=seed)
    assert result['seed']==seed and result['prefix_duration_ms']==2500
    assert result['prefix_bytes']==24 and result['raw_byte_prefix_exact']
    prior['seed']=1729
    with pytest.raises(ValueError):m.audit_rank(report,progress,path,**args,expected_seed=seed)
    prior['seed']=seed
    values=np.fromfile(previous,dtype=m.DTYPE);values['tick'][0]+=1;values.tofile(previous)
    prior['event_sha256']=hashlib.sha256(previous.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='prefix bytes'):m.audit_rank(report,progress,path,**args,expected_seed=seed)


def test_primary_default_does_not_silently_accept_reference_prefix_duration(tmp_path):
    report,progress,path=recording(tmp_path,'full',100500)
    prior,_,previous=recording(tmp_path,'short',2500)
    with pytest.raises(ValueError,match='prefix durations'):
        m.audit_rank(report,progress,path,rank=0,duration_ms=100500,neurons=96,parameters_sha256='fixture',
                     allowed_cpus=[0],prefix_report=prior,prefix_events=previous,prefix_allowed_cpus=[0],release_cache=False)
