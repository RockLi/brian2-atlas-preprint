from pathlib import Path
from types import SimpleNamespace
import sys
import os
import json
import gzip
import hashlib
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import mam_correlation_stream as s
import mam_paper_correlation as m
import analyze_native_mam_activity as native
import analyze_mam_paper_correlation as cli
LINUX = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='explicit Linux cache-release path')


@pytest.mark.parametrize('size',[1,7,131072])
@pytest.mark.parametrize('offset',[0,1])
def test_streamed_population_preserves_warmup_id_endpoints_and_summaries(size,offset):
    ticks=np.array([100,24999,5000,6000,25000,4999,5001,23000],dtype='<u4')
    cells=np.array([0,2,1,3000,3000,3001,2,1],dtype='<u4')
    # The minimum recorded ID only fires in warmup. Later chunks/rank files
    # need not be globally ordered; selection must still start at zero.
    def blocks():
        for lo in range(0,len(ticks),size):
            if offset:
                raw=ticks[lo:lo+size]-1
                yield raw.astype(np.int64)+1,cells[lo:lo+size]
            else:yield ticks[lo:lo+size],cells[lo:lo+size]
    frozen=np.histogram(ticks[(ticks>=5000)&(ticks<25000)],bins=np.arange(5000,25001,10))[0]
    old,ids,selected=m.summarize_histogram(m.candidate_histogram(ticks,cells,0),0)
    new,samples,observed=s.summarize_population(blocks,neurons=4000,end_tick=25000,expected_raw=len(ticks),frozen=frozen)
    for k in old:
        if k=='mean_pairwise_correlation':assert abs(old[k]-new[k])<=1e-12
        else:assert old[k]==new[k]
    for key,array in dict(selected_ids=ids,selected_spikes=selected.sum(axis=1),selected_histogram_sum=selected.sum(axis=0)).items():
        assert samples[key].dtype==array.dtype;np.testing.assert_array_equal(samples[key],array)
    np.testing.assert_array_equal(observed,frozen)


def test_empty_and_mismatched_recordings_are_not_fabricated():
    empty=lambda:iter([])
    r,a,_=s.summarize_population(empty,neurons=10,end_tick=25000,expected_raw=0,frozen=np.zeros(2000))
    assert r['available'] is False and r['mean_pairwise_correlation'] is None and a=={}
    for n in [-1,2**32,True]:
        with pytest.raises(ValueError):s.summarize_population(empty,neurons=10,end_tick=25000,expected_raw=n,frozen=np.zeros(2000))
    with pytest.raises(ValueError):s.summarize_population(empty,neurons=10,end_tick=999999999,expected_raw=0,frozen=np.zeros(2000))
    with pytest.raises(ValueError,match='count mismatch'):s.summarize_population(empty,neurons=10,end_tick=25000,expected_raw=1,frozen=np.zeros(2000))


@pytest.mark.parametrize('ticks,cells',[(np.array([25001]),np.array([0])),(np.array([5000]),np.array([10])),(np.array([-1]),np.array([0])),(np.zeros(131073,dtype=int),np.zeros(131073,dtype=int))])
def test_bad_blocks_rejected(ticks,cells):
    with pytest.raises(ValueError):s.summarize_population(lambda:iter([(ticks,cells)]),neurons=10,end_tick=25000,expected_raw=len(ticks),frozen=np.zeros(2000))


def test_second_pass_count_or_first_id_change_is_rejected():
    calls=0
    def blocks():
        nonlocal calls
        calls+=1
        yield np.array([5000]),np.array([calls-1])
    with pytest.raises(ValueError,match='changed between'):s.summarize_population(blocks,neurons=10,end_tick=25000,expected_raw=1,frozen=np.zeros(2000))


@LINUX
def test_file_cache_adapters_keep_bytes_views_and_close_owned_fds(tmp_path):
    path=tmp_path/'events';events=np.array([(5000+i%17,i%3) for i in range(131074)],dtype=native.DTYPE);events.tofile(path)
    assert s.file_sha(path)==hashlib.sha256(path.read_bytes()).hexdigest()
    arrays=list(s.native_blocks(path,native.DTYPE))
    assert [len(a[0]) for a in arrays]==[131072,2]
    np.testing.assert_array_equal(np.concatenate([a[1] for a in arrays]),events['cell'])
    mapping=np.memmap(path,mode='r',dtype='u1')
    view=np.ndarray(events.shape,dtype=native.DTYPE,buffer=mapping)
    before=len(list(Path('/proc/self/fd').iterdir()))
    with pytest.raises(RuntimeError,match='forced'):
        with s.mapped_cache(mapping,path) as release:
            release(view['tick']);release(view['cell'])
            with pytest.raises(ValueError):release(np.arange(4))
            raise RuntimeError('forced')
    assert len(list(Path('/proc/self/fd').iterdir()))==before
    np.testing.assert_array_equal(view,events)
    with s.mapped_cache(mapping,path) as release:
        population=dict(spike_ticks=view['tick'],indices=view['cell'])
        got=list(s.rust_blocks(population,release))
        np.testing.assert_array_equal(np.concatenate([x for x,_ in got]),events['tick'].astype(np.int64)+1)
    other=tmp_path/'other';other.write_bytes(path.read_bytes())
    with pytest.raises(ValueError,match='does not match'):
        with s.mapped_cache(mapping,other):pass
    mapping._mmap.close()


@LINUX
def test_bucket_cache_mode_matches_original_across_flush_boundaries(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'CACHE_CHUNK',8)
    p=tmp_path/'raw';a=np.array([(4999,2),(25000,0),(5000,0),(6000,1),(7000,2)],dtype=native.DTYPE);a.tofile(p)
    args=([dict(count=1),dict(count=2)],[p])
    old,n=native.bucket_events(*args,tmp_path/'old',start_tick=0,end_tick=25001)
    new,m=native.bucket_events(*args,tmp_path/'new',start_tick=0,end_tick=25001,release_file_cache=True)
    np.testing.assert_array_equal(n,m)
    for x,y in zip(old,new,strict=True):assert x.read_bytes()==y.read_bytes()
    assert p.read_bytes()==a.tobytes()


@LINUX
@pytest.mark.parametrize('end',[25000,105000])
def test_native_complete_cli_preserves_legacy_report_and_selection(tmp_path,end):
    raw=gzip.decompress((ROOT/'mpi-evidence/region/full-parameters.json.gz').read_bytes());params=json.loads(raw)
    digest=hashlib.sha256(raw).hexdigest();audit=tmp_path/'audit';parameter=audit/'node25/fixture/parameters.json';parameter.parent.mkdir(parents=True);parameter.write_bytes(raw)
    folder=audit/'node25/runs/fixture';folder.mkdir(parents=True)
    # Real population sizes/identity, sparse independent native fixture.
    cells=np.array([0,1,2,0,1,2,0],dtype='<u4');ticks=np.array([100,5000,7000,end,4999,end-1,6000],dtype='<u4')
    events=np.empty(len(cells),dtype=native.DTYPE);events['tick']=ticks;events['cell']=cells
    files={};reports=[]
    for rank in range(48):
        p=folder/f'rank{rank}.events.bin';(events if rank==0 else np.empty(0,dtype=native.DTYPE)).tofile(p)
        h=hashlib.sha256(p.read_bytes()).hexdigest();files[str(p)]=dict(bytes=p.stat().st_size,sha256=h)
        r=dict(rank=rank,seed=1729,ranks=48,threads=4,duration_ms=end//10,dt_ms=.1,nest_version='3.10.0',parameters_sha256=digest,event_sha256=h,event_bytes=p.stat().st_size)
        reports.append(r);p.with_name(f'rank{rank}.json').write_text(json.dumps(r))
    (audit/'summary.json').write_text(json.dumps(dict(passed=True,parameters_sha256=digest,spikes=len(events))))
    order=native.canonical_population_indices(params['populations']);counts=np.zeros((254,(end-5000)//10),dtype=np.int64)
    j=order.index(0);valid=(ticks>=5000)&(ticks<end);np.add.at(counts[j],(ticks[valid]-5000)//10,1)
    baseline=tmp_path/'baseline';baseline.mkdir()
    pops=[dict(name=native.canonical_population_name(params['populations'][k]),neurons=params['populations'][k]['count'],area=params['populations'][k]['area'],group=params['populations'][k]['population'],pairwise_corr_mean=0.) for k in order]
    base=dict(neurons=4129924,populations=pops,window=dict(start_tick=5000,end_tick=end,bin_ticks=10,endpoint='[start,end)',spike_tick_offset=0),parameters_sha256=digest,result_files=files,simulation=native.simulation_identity(reports,duration_ms=end//10))
    (baseline/'activity.json').write_text(json.dumps(base));np.savez_compressed(baseline/'activity-arrays.npz',population_counts=counts)
    catalog={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in baseline.iterdir()};(baseline/'catalog.json').write_text(json.dumps(catalog))
    for mode in [False,True]:cli.analyze(SimpleNamespace(baseline=baseline,output=tmp_path/str(mode),native_audit=audit,model=None,results=None,bounded_memory=mode))
    a=json.loads((tmp_path/'False/correlation.json').read_text());b=json.loads((tmp_path/'True/correlation.json').read_text())
    for key in a:
        if key not in ['populations','analysis_seconds','implementation_sha256','bounded_memory']:assert a[key]==b[key]
    for x,y in zip(a['populations'],b['populations'],strict=True):
        for key in x:
            if key=='mean_pairwise_correlation' and x[key] is not None:assert abs(x[key]-y[key])<=1e-12
            else:assert x[key]==y[key]
    with np.load(tmp_path/'False/selection.npz') as a,np.load(tmp_path/'True/selection.npz') as b:
        assert a.files==b.files
        for key in a.files:assert a[key].dtype==b[key].dtype;np.testing.assert_array_equal(a[key],b[key])
    assert not (tmp_path/'True/scratch-events').exists()
    assert (folder/'rank0.events.bin').read_bytes()==events.tobytes()
