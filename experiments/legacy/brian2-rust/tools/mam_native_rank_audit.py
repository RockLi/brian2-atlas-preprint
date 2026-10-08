"""Bounded native rank recording audit for 10.5/100.5 s observations.

This verifies recorder bytes, chunk ledgers and an exact retained prefix. It
does not independently reproduce the NEST trajectory, validate whole-cluster
resources, or establish scientific equivalence.
"""
from contextlib import ExitStack
import hashlib
import math
import os
from pathlib import Path

import numpy as np

DTYPE=np.dtype([('tick','<u4'),('cell','<u4')])
BLOCK=131072
CACHE_WINDOW=64*2**20
MAX_SPIKES=268435456


def require(ok,message):
    if not ok: raise ValueError(message)


def nonnegative(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def identity(report,rank,duration_ms,neurons,parameters_sha256,allowed_cpus,*,expected_seed=1729):
    require(type(expected_seed) is int and 0<expected_seed<2**31, 'invalid expected native seed')
    require(report['schema']=='b2-native-nest-mam-v1'
            and (report['rank'],report['ranks'],report['threads'],report['seed'],report['dt_ms'],report['duration_ms'])
            ==(rank,48,4,expected_seed,.1,duration_ms), 'native rank identity/duration differs')
    require(0<=rank<48 and duration_ms in (2500,10500,100500), 'unadmitted rank/duration')
    require(report['parameters_sha256']==parameters_sha256 and report['nest_version']=='3.10.0', 'native parameters/version differs')
    require(report['allowed_cpus']==allowed_cpus, 'reported main-thread affinity differs')
    offset=0
    for pop in report['populations']:
        require(type(pop['count']) is int and pop['count']>0
                and (pop['first_gid'],pop['cell_start'],pop['cell_end'])
                ==(offset+1,offset,offset+pop['count']), 'native population ID geometry differs')
        offset+=pop['count']
    require(offset==neurons, 'population neuron total differs')
    counts=report['projection_local_counts']
    require(all(type(v) is int and v>=0 for v in counts)
            and sum(counts)==report['local_recurrent_edges'], 'native construction ledger differs')
    require(report['local_total_connections']==report['local_recurrent_edges']
            +report['local_external_connections']+report['local_recording_connections'], 'connection total differs')
    require(type(report['local_spikes']) is int and 0<=report['local_spikes']<=MAX_SPIKES
            and report['event_bytes']==8*report['local_spikes'], 'native event quota/bytes differ')
    require(len(report['chunks'])==duration_ms//50, 'native chunk coverage differs')


def audit_rank(report,progress,events,*,rank,duration_ms,neurons,parameters_sha256,
               allowed_cpus,prefix_report,prefix_events,prefix_allowed_cpus,
               release_cache=True,expected_seed=1729):
    """No sorting/relabeling: retain and compare native within-chunk order.

    Recorder-drain chunks and physical-time bins are distinct. A record may
    carry a time earlier than the drain chunk due to recorder delivery; only
    the upper physical bound and full independent time histogram are enforced.
    """
    identity(report,rank,duration_ms,neurons,parameters_sha256,allowed_cpus,expected_seed=expected_seed)
    prefix_duration=prefix_report['duration_ms']
    require((duration_ms,prefix_duration) in [(10500,2500),(100500,10500)]
            or ((duration_ms,prefix_duration)==(100500,2500) and expected_seed in (1730,1731)),
            'unadmitted exact prefix durations')
    identity(prefix_report,rank,prefix_duration,neurons,parameters_sha256,prefix_allowed_cpus,expected_seed=expected_seed)
    for key in ['populations','projection_local_counts','local_recurrent_edges','local_external_connections',
                'local_recording_connections','local_total_connections','min_delay_ms','max_delay_ms',
                'connection_index_capacity','N_scaling','K_scaling']:
        require(report[key]==prefix_report[key], 'retained construction field differs: '+key)
    old_chunks=prefix_report['chunks'];chunks=report['chunks']
    require([(c['end_ms'],c['spikes']) for c in chunks[:len(old_chunks)]]
            ==[(c['end_ms'],c['spikes']) for c in old_chunks], 'retained prefix chunks differ')
    require(len(progress)==len(chunks), 'native progress coverage differs')
    if release_cache: require(hasattr(os,'posix_fadvise'), 'Linux cache release required')
    require(events.stat().st_size==report['event_bytes'] and prefix_events.stat().st_size==prefix_report['event_bytes'],
            'native recording file size differs')
    for path in [events,prefix_events]:require(path.is_file() and not path.is_symlink(), 'regular event files required')
    physical=np.zeros(duration_ms//50,dtype=np.int64)
    end_tick=duration_ms*10
    terminal=total=prefix_bytes=0
    digest,old_digest=hashlib.sha256(),hashlib.sha256()
    maximum_block=0
    with ExitStack() as stack:
        current=stack.enter_context(events.open('rb'));prior=stack.enter_context(prefix_events.open('rb'))
        released={current:0,prior:0}
        def release(stream,final=False):
            length=stream.tell()-released[stream]
            if release_cache and length>0 and (final or length>=CACHE_WINDOW):
                os.posix_fadvise(stream.fileno(),released[stream],length,os.POSIX_FADV_DONTNEED)
                released[stream]=stream.tell()
        for index,(chunk,row) in enumerate(zip(chunks,progress,strict=True)):
            count=chunk['spikes'];end=(index+1)*50
            require(type(count) is int and 0<=count<=2000000, 'native chunk spike quota')
            require(chunk['end_ms']==row['end_ms']==end and row['rank']==rank
                    and row['event']=='chunk_complete' and row['spikes']==count, 'chunk/progress identity differs')
            for key in ['simulation_seconds','output_seconds']:
                require(nonnegative(chunk[key]) and row[key]==chunk[key], 'chunk timing differs')
            total+=count
            require(row['total_spikes']==total and row['event_bytes']==8*total
                    and row['memory']==report['phase_memory'][f'after_{end}ms'], 'progress count/memory ledger differs')
            left=count
            while left:
                block=np.fromfile(current,dtype=DTYPE,count=min(left,BLOCK))
                require(len(block)==min(left,BLOCK), 'truncated native chunk')
                maximum_block=max(maximum_block,len(block));left-=len(block)
                require(np.all(block['tick']<=end*10) and np.all(block['cell']<neurons), 'event outside physical/cell bounds')
                require(np.all((block['cell'].astype(np.uint64)+1)%48==rank), 'native event rank ownership differs')
                before=block['tick']<end_tick
                physical+=np.bincount(block['tick'][before]//500,minlength=len(physical))
                terminal+=int(np.count_nonzero(block['tick']==end_tick))
                raw=block.tobytes();digest.update(raw)
                remaining=prefix_report['event_bytes']-prefix_bytes
                if remaining:
                    size=min(len(raw),remaining);previous=prior.read(size)
                    require(len(previous)==size and previous==raw[:size], 'native event prefix bytes differ')
                    old_digest.update(previous);prefix_bytes+=size;release(prior)
                release(current)
        require(not current.read(1) and not prior.read(1), 'trailing native recording bytes')
        release(current,True);release(prior,True)
    require(total==report['local_spikes'] and int(physical.sum())+terminal==total, 'native physical histogram differs')
    require(digest.hexdigest()==report['event_sha256'] and old_digest.hexdigest()==prefix_report['event_sha256'],
            'native recording checksum differs')
    require(prefix_bytes==prefix_report['event_bytes'], 'native prefix coverage incomplete')
    result=dict(rank=rank,duration_ms=duration_ms,rank_event_stream_verified=True,
        physical_50ms_bin_counts=physical.tolist(),terminal_tick_events=terminal,spikes=total,event_bytes=8*total,
        event_sha256=digest.hexdigest(),prefix_duration_ms=prefix_duration,prefix_bytes=prefix_bytes,
        prefix_sha256=old_digest.hexdigest(),raw_byte_prefix_exact=True,construction_ledger_exact=True,
        chunks=len(chunks),maximum_read_block_events=maximum_block,linux_cache_release=release_cache,
        scientific_equivalence=False,whole_cluster_resource_audit_passed=False,
        scope='One native rank recording and retained exact prefix, including independent physical time bins. '
              'Caller must pin parameters, terminal reports, host/rank coverage, global projection totals and guards. '
              'Main-thread affinity is not an observation of every OpenMP worker. NEST dynamics and scientific '
              'equivalence are not independently proved by recorder consistency.')
    if expected_seed!=1729:result['seed']=expected_seed
    return result
