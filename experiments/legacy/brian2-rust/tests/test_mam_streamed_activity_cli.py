from pathlib import Path
import sys,json,gzip,hashlib,struct
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'python')]
import analyze_multi_area_activity as rust
import analyze_native_mam_activity as native
from mam_raster_budget import RasterBudget,RasterSample
LINUX=pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux cache-release mode')


def compare_outputs(a,b,plots):
    x=json.loads((a/'activity.json').read_text());y=json.loads((b/'activity.json').read_text())
    assert y['bounded_memory'] and y['raster_point_budget']['used']==sum(len(r['times']) for r in plots[1])
    for key in x:
        if key!='populations':assert x[key]==y[key],key
    for old,new in zip(x['populations'],y['populations'],strict=True):
        for k in old:
            if k in ['lvr_zero_padded_mean','lvr_eligible_mean','pairwise_corr_mean'] and old[k] is not None:np.testing.assert_allclose(old[k],new[k],rtol=1e-12,atol=1e-12)
            else:assert old[k]==new[k]
    with np.load(a/'activity-arrays.npz') as x,np.load(b/'activity-arrays.npz') as y:
        assert x.files==y.files
        for k in x.files:
            assert x[k].dtype==y[k].dtype
            if k.endswith('_lvr_values'):np.testing.assert_allclose(x[k],y[k],rtol=1e-12,atol=1e-12)
            else:np.testing.assert_array_equal(x[k],y[k])
    assert len(plots[0])==len(plots[1])
    for x,y in zip(plots[0],plots[1],strict=True):
        for k in x:
            if k in ['times','indices']:assert x[k].dtype==y[k].dtype;np.testing.assert_array_equal(x[k],y[k])
            else:assert x[k]==y[k]


@LINUX
@pytest.mark.parametrize('offset',[0,1])
@pytest.mark.parametrize('end',[25000,105000])
def test_rust_bounded_cli_both_timestamp_maps_and_raster(tmp_path,monkeypatch,offset,end):
    model={'definition':{'populations':[dict(name='mam_V1_23E',count=1,dt=struct.pack('>d',.0001).hex(),steps=end,monitor={'window_steps':end})]}}
    model_path=tmp_path/'model.json';model_path.write_text(json.dumps(model));run=tmp_path/'run';run.mkdir()
    events=np.array([(4998,0),(4999,0),(5000,0),(end-2,0),(end-1,0)],dtype=[('tick','<i8'),('cell','<i8')]);events.tofile(run/'results.bin');(run/'events.bin').write_bytes(b'fixture')
    mapping=np.memmap(run/'results.bin',mode='r',dtype=np.uint8);view=np.ndarray(events.shape,dtype=events.dtype,buffer=mapping)
    pop=dict(spike_ticks=view['tick'],indices=view['cell']);monkeypatch.setattr(rust,'load_results',lambda *a,**k:dict(populations=[pop],_dump=mapping))
    plots=[];monkeypatch.setattr(rust,'plot',lambda rows,areas,groups,rates,raster,window,output:plots.append(raster))
    before=len(list(Path('/proc/self/fd').iterdir()))
    for mode in [False,True]:rust.analyze(model_path,run,tmp_path/str(mode),spike_tick_offset=offset,end_tick=end,bounded_memory=mode)
    assert len(list(Path('/proc/self/fd').iterdir()))==before
    compare_outputs(tmp_path/'False',tmp_path/'True',plots)
    assert (run/'results.bin').read_bytes()==events.tobytes();mapping._mmap.close()


def native_fixture(tmp_path,end):
    raw=gzip.decompress((ROOT/'mpi-evidence/region/full-parameters.json.gz').read_bytes());params=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    assert params['populations'][0]['area']=='V1'
    number=native.canonical_population_indices(params['populations']).index(0)
    selected=RasterSample(params['populations'][0]['count'],20260908+10000+number,RasterBudget()).selected
    a,c=map(int,selected[:2]);events=np.array([(4999,a),(5000,a),(5001,c),(5020,a),(5040,c),(5050,a),(end-1,c),(end,a)],dtype=native.DTYPE)
    audit=tmp_path/'audit';p=audit/'node25/fixture/parameters.json';p.parent.mkdir(parents=True);p.write_bytes(raw);folder=audit/'node25/runs/fixture';folder.mkdir(parents=True)
    for rank in range(48):
        p=folder/f'rank{rank}.events.bin';(events if rank==0 else np.empty(0,dtype=native.DTYPE)).tofile(p)
        r=dict(rank=rank,seed=1729,ranks=48,threads=4,duration_ms=end//10,dt_ms=.1,nest_version='3.10.0',parameters_sha256=digest,event_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),event_bytes=p.stat().st_size)
        p.with_name(f'rank{rank}.json').write_text(json.dumps(r))
    bins=np.bincount(events['tick'][events['tick']<end]//500,minlength=end//500)
    (audit/'summary.json').write_text(json.dumps(dict(passed=True,parameters_sha256=digest,spikes=len(events),physical_50ms_bin_counts=bins.tolist())))
    return audit,folder/'rank0.events.bin',events.tobytes()


@LINUX
@pytest.mark.parametrize('end',[25000,105000])
def test_native_bounded_cli_exact_raster_and_baseline(tmp_path,monkeypatch,end):
    audit,path,original=native_fixture(tmp_path,end);plots=[]
    monkeypatch.setattr(native,'plot',lambda rows,areas,groups,rates,raster,window,output:plots.append(raster))
    before=len(list(Path('/proc/self/fd').iterdir()))
    for mode in [False,True]:native.analyze(audit,tmp_path/str(mode),end_tick=end,bounded_memory=mode)
    assert len(list(Path('/proc/self/fd').iterdir()))==before
    compare_outputs(tmp_path/'False',tmp_path/'True',plots)
    assert not (tmp_path/'True/population-events').exists() and path.read_bytes()==original


@LINUX
def test_native_raster_budget_failure_closes_inputs_without_success_report(tmp_path,monkeypatch):
    audit,path,original=native_fixture(tmp_path,25000);plots=[];monkeypatch.setattr(native,'plot',lambda *args:plots.append(args))
    before=len(list(Path('/proc/self/fd').iterdir()))
    with pytest.raises(ValueError,match='raster point budget'):native.analyze(audit,tmp_path/'failed',bounded_memory=True,raster_max_points=1)
    assert len(list(Path('/proc/self/fd').iterdir()))==before
    assert not plots and not (tmp_path/'failed/activity.json').exists() and not (tmp_path/'failed/catalog.json').exists()
    assert path.read_bytes()==original


@pytest.mark.parametrize('simulator',['rust','native'])
def test_custom_raster_budget_cannot_be_silently_ignored_in_legacy_mode(tmp_path,simulator):
    with pytest.raises(ValueError,match='requires bounded_memory'):
        if simulator=='rust':rust.analyze(tmp_path/'missing',tmp_path/'missing',tmp_path/'out',raster_max_points=1)
        else:native.analyze(tmp_path/'missing',tmp_path/'out',raster_max_points=1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('simulator',['rust','native'])
def test_primary_window_requires_bounded_mode_before_io(tmp_path,simulator):
    with pytest.raises(ValueError,match='requires bounded memory'):
        if simulator=='rust':rust.analyze(tmp_path/'missing',tmp_path/'missing',tmp_path/'out',end_tick=1005000,spike_tick_offset=1)
        else:native.analyze(tmp_path/'missing',tmp_path/'out',end_tick=1005000)
    assert not list(tmp_path.iterdir())


def test_primary_rust_requires_physical_grid_before_io(tmp_path):
    with pytest.raises(ValueError,match='physical spike grid'):
        rust.analyze(tmp_path/'missing',tmp_path/'missing',tmp_path/'out',end_tick=1005000,bounded_memory=True)


@LINUX
@pytest.mark.parametrize('model_end,invalid_tick',[(1005000,False),(105000,False),(1005000,True)])
def test_primary_rust_full_statistics_and_separate_raster(tmp_path,monkeypatch,model_end,invalid_tick):
    end=1005000
    model={'definition':{'populations':[dict(name='mam_V1_23E',count=1,dt=struct.pack('>d',.0001).hex(),steps=model_end,monitor={'window_steps':model_end})]}}
    path=tmp_path/'model.json';path.write_text(json.dumps(model));run=tmp_path/'run';run.mkdir()
    physical=[4999,5000,105000,500000,1004999,1005001 if invalid_tick else 1005000]
    events=np.array([(tick-1,0) for tick in physical],dtype=[('tick','<i8'),('cell','<i8')])
    events.tofile(run/'results.bin');(run/'events.bin').write_bytes(b'fixture')
    mapping=np.memmap(run/'results.bin',mode='r',dtype=np.uint8);view=np.ndarray(events.shape,dtype=events.dtype,buffer=mapping)
    pop=dict(spike_ticks=view['tick'],indices=view['cell']);loads=[]
    def load(*a,**k):
        loads.append(True)
        return dict(populations=[pop],_dump=mapping)
    monkeypatch.setattr(rust,'load_results',load)
    plots=[];monkeypatch.setattr(rust,'plot',lambda rows,areas,groups,rates,raster,window,output:plots.append((raster,window)))
    try:
        if model_end!=end or invalid_tick:
            with pytest.raises(ValueError):rust.analyze(path,run,tmp_path/'out',end_tick=end,spike_tick_offset=1,bounded_memory=True)
            assert not (tmp_path/'out/activity.json').exists() and not plots
            if model_end!=end:assert not loads
            return
        rust.analyze(path,run,tmp_path/'out',end_tick=end,spike_tick_offset=1,bounded_memory=True)
        report=json.loads((tmp_path/'out/activity.json').read_text())
        assert report['window']['seconds']==100 and report['window']['raster_end_tick']==105000
        assert report['observed_spikes']==4 and report['mean_rate_hz']==.04
        with np.load(tmp_path/'out/activity-arrays.npz') as data:
            expected=np.zeros(100000,dtype=np.int64);expected[[0,10000,49500,99999]]=1
            np.testing.assert_array_equal(data['population_counts'][0],expected)
        assert len(plots)==1
        np.testing.assert_array_equal(plots[0][0][0]['times'],[.5])
        assert report['raster_point_budget']['used']==1
        assert (run/'results.bin').read_bytes()==events.tobytes()
    finally:mapping._mmap.close()


@LINUX
def test_primary_native_full_window_includes_late_activity(tmp_path,monkeypatch):
    end=1005000
    audit,path,original=native_fixture(tmp_path,end);plots=[]
    monkeypatch.setattr(native,'plot',lambda rows,areas,groups,rates,raster,window,output:plots.append((raster,window)))
    native.analyze(audit,tmp_path/'out',end_tick=end,bounded_memory=True,max_bytes=128*2**30)
    report=json.loads((tmp_path/'out/activity.json').read_text())
    assert report['simulation']['duration_ms']==100500
    assert report['window']['seconds']==100 and report['window']['raster_end_tick']==105000
    assert report['observed_spikes']==6 and report['mean_rate_hz']==6/4129924/100
    with np.load(tmp_path/'out/activity-arrays.npz') as data:
        assert data['population_counts'].shape==(254,100000)
        assert data['population_counts'][:,-1].sum()==1
        assert data['population_counts'].sum()==6
    assert sum(len(r['times']) for r in plots[0][0])==5
    assert all(np.all((r['times']>=.5)&(r['times']<10.5)) for r in plots[0][0])
    assert path.read_bytes()==original and not (tmp_path/'out/population-events').exists()
