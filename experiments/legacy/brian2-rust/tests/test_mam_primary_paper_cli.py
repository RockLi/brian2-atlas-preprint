"""Complete sparse 254-population fixtures on the admitted primary time grid.

MAM_NORMALIZATION_DIR supplies the already verified official normalization
files for the spectrum CLI, without duplicating its 6 MiB generated source.
The Rust fixture injects only the binary reader result; existing reader tests
cover the wire format. All CLI hashing, event adapters and statistics run.
"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import os,sys,json,gzip,hashlib,struct,importlib.util
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'python')]
import brian2_rust.results as reader
import analyze_native_mam_activity as activity
import analyze_mam_paper_cell_metrics as cell
import analyze_mam_paper_correlation as correlation
import analyze_mam_paper_time_series as series
from validate_mam_paper_spectrum import manual_welch
from mam_paper_rate_bins import full_rate_bins
LINUX=pytest.mark.skipif(not sys.platform.startswith('linux'),reason='explicit Linux cache-release path')


def catalog(folder):
    (folder/'catalog.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in folder.iterdir() if p.name!='catalog.json'}))


def fixture(tmp_path,monkeypatch,end):
    raw=gzip.decompress((ROOT/'mpi-evidence/region/full-parameters.json.gz').read_bytes());params=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    audit=tmp_path/'audit';p=audit/'node25/fixture/parameters.json';p.parent.mkdir(parents=True);p.write_bytes(raw)
    folder=audit/'node25/runs/fixture';folder.mkdir(parents=True)
    events=np.array([(100,0),(4999,1),(5000,1),(5004,2),(5005,1),(5015,1),(5050,2),(end//2,1),(end-1000,2),(end-1,1),(end,2)],dtype=activity.DTYPE)
    for rank in range(48):
        p=folder/f'rank{rank}.events.bin';local=events[(events['cell']+1)%48==rank];local.tofile(p)
        report=dict(rank=rank,seed=1729,ranks=48,threads=4,duration_ms=end//10,dt_ms=.1,nest_version='3.10.0',parameters_sha256=digest,event_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),event_bytes=p.stat().st_size,local_spikes=len(local))
        p.with_name(f'rank{rank}.json').write_text(json.dumps(report))
    counts=np.bincount(events['tick'][events['tick']<end]//500,minlength=end//500)
    (audit/'summary.json').write_text(json.dumps(dict(passed=True,parameters_sha256=digest,spikes=len(events),terminal_tick_events=1,physical_50ms_bin_counts=counts.tolist())))
    monkeypatch.setattr(activity,'plot',lambda *args:None)
    baseline=tmp_path/'baseline';activity.analyze(audit,baseline,end_tick=end,bounded_memory=True)
    return params,audit,baseline,events


@contextmanager
def rust_fixture(tmp_path,monkeypatch,params,baseline,events,end):
    base=json.loads((baseline/'activity.json').read_text());pops=base['populations'];names=[p['name'] for p in pops];active=names.index('mam_V1_23E')
    model=dict(definition=dict(populations=[dict(name=p['name'],count=p['neurons'],steps=end,dt=struct.pack('>d',.0001).hex()) for p in pops]),instance=dict(rng_seed=1729))
    model_path=tmp_path/'model.json';model_path.write_text(json.dumps(model));run=tmp_path/'run';run.mkdir()
    data=np.empty(len(events),dtype=activity.DTYPE);data['tick']=events['tick']-1;data['cell']=events['cell'];data.tofile(run/'results.bin');(run/'events.bin').write_bytes(b'reader fixture')
    mapping=np.memmap(run/'results.bin',mode='r',dtype=np.uint8);view=np.ndarray(data.shape,dtype=data.dtype,buffer=mapping)
    populations=[dict(spike_ticks=view['tick'] if i==active else view['tick'][:0],indices=view['cell'] if i==active else view['cell'][:0]) for i in range(254)]
    def load(*args,**kwargs):
        assert kwargs['include_times'] is False
        return dict(populations=populations,_dump=mapping,metadata=dict(spike_count=len(events)))
    monkeypatch.setattr(reader,'load_results',load)
    # The cell CLI loads the same reader from a file spec. Substitute just that
    # module, preserving normal imports and all downstream CLI code.
    original_spec=importlib.util.spec_from_file_location;original_module=importlib.util.module_from_spec
    class Loader:
        def exec_module(self,module):pass
    fake=SimpleNamespace(loader=Loader())
    monkeypatch.setattr(importlib.util,'spec_from_file_location',lambda name,*a,**k:fake if name=='paper_results' else original_spec(name,*a,**k))
    monkeypatch.setattr(importlib.util,'module_from_spec',lambda spec:SimpleNamespace(load_results=load) if spec is fake else original_module(spec))
    base['window']['spike_tick_offset']=1
    base['model_sha256']=hashlib.sha256(model_path.read_bytes()).hexdigest()
    base['result_sha256']={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ['results.bin','events.bin']}
    (baseline/'activity.json').write_text(json.dumps(base));catalog(baseline)
    try:yield model_path,run
    finally:mapping._mmap.close()


def check_complete_outputs(tmp_path,params,baseline,events,end,normalization):
    base=json.loads((baseline/'activity.json').read_text());j=[p['name'] for p in base['populations']].index('mam_V1_23E');n=params['populations'][0]['count']
    expected,per_cell=cell.cell_metrics(events['tick'],events['cell'],n,end=end)
    rows=json.loads((tmp_path/'cell/paper-cell-metrics.json').read_text())['populations']
    for k,v in expected.items():
        if v is not None and isinstance(v,float):assert rows[j][k]==pytest.approx(v,rel=1e-12,abs=1e-12)
        else:assert rows[j][k]==v
    assert rows[j]['half_open_spikes']==8 and rows[j]['strict_spikes']==7
    with np.load(tmp_path/'cell/cell-metrics.npz') as data:
        assert len(data.files)==762
        for key,v in per_cell.items():np.testing.assert_allclose(data[f'p{j}_{key}'],v,rtol=1e-12,atol=1e-12)
    corr=json.loads((tmp_path/'corr/correlation.json').read_text())
    assert corr['raw_events']==len(events) and corr['observation_ms']==[500,end//10]
    bins=np.arange(5000,end+1,10)
    # Independently compute just the two varying candidate rows, including T.
    h=np.array([np.histogram(events['tick'][events['cell']==i],bins=bins)[0] for i in [1,2]])
    assert corr['populations'][j]['mean_pairwise_correlation']==pytest.approx(np.corrcoef(h)[0,1],abs=1e-12)
    with np.load(tmp_path/'corr/selection.npz') as data:np.testing.assert_array_equal(data[f'p{j}_selected_ids'],[1,2])
    report=json.loads((tmp_path/'series/time-series.json').read_text())
    assert report['frozen_histogram_exact'] and report['endpoint_count_identity_exact']
    assert report['raw_events']==len(events) and report['frozen_events']==8
    assert report['shifted_events']==7 and report['included_terminal_events']==1 and report['excluded_lower_half_ms_events']==2
    weights=json.loads(normalization.read_text())['populations'];weight=weights[j]['official_normalization_neurons']
    expected_counts=full_rate_bins(events['tick'],1000,end)
    with np.load(tmp_path/'series/time-series.npz') as data:
        assert data['population_counts'].shape==(254,(end-5000)//10)
        np.testing.assert_array_equal(data['population_counts'][j],expected_counts)
        np.testing.assert_array_equal(data['population_rates_hz'][j],expected_counts/(weight/1000.))
        assert data['population_counts'][j,-1]==2 # end-1 and exact terminal both count
        index=report['area_names'].index('V1');freq,power=manual_welch(data['area_rates_hz'][index])
        np.testing.assert_array_equal(data['frequency_hz'],freq)
        np.testing.assert_allclose(data['power_hz2_per_hz'][index],power,rtol=5e-13,atol=1e-20)
    assert not (tmp_path/'cell/population-events').exists() and not (tmp_path/'corr/scratch-events').exists()


@LINUX
@pytest.mark.parametrize('simulator',['native','rust'])
def test_complete_primary_paper_views(tmp_path,monkeypatch,simulator):
    directory=os.environ.get('MAM_NORMALIZATION_DIR')
    if not directory:pytest.skip('requires retained official normalization via MAM_NORMALIZATION_DIR')
    normalization=Path(directory)/'mam-official-analysis-neuron-sizes-v1.json'
    params,audit,baseline,events=fixture(tmp_path,monkeypatch,1005000)
    monkeypatch.setattr(series,'plot',lambda *a,**k:None)
    from contextlib import nullcontext
    ctx=rust_fixture(tmp_path,monkeypatch,params,baseline,events,1005000) if simulator=='rust' else nullcontext((None,None))
    before=len(list(Path('/proc/self/fd').iterdir()))
    original={p:p.read_bytes() for p in audit.glob('node*/runs/*/rank*.events.bin')}
    with ctx as (model,results):
        args=SimpleNamespace(baseline=baseline,native_audit=audit if simulator=='native' else None,model=model,results=results,bounded_memory=True,normalization=normalization)
        for name,tool in [('cell',cell),('corr',correlation),('series',series)]:
            args.output=tmp_path/name;tool.analyze(args)
        descriptors=len(list(Path('/proc/self/fd').iterdir()))
        def fail_count(*a,**k):raise RuntimeError('forced histogram failure')
        with monkeypatch.context() as patch:
            patch.setattr(series.Counts,'add',fail_count);args.output=tmp_path/'series-failed'
            with pytest.raises(RuntimeError,match='forced histogram'):series.analyze(args)
        assert len(list(Path('/proc/self/fd').iterdir()))==descriptors
        assert not args.output.exists()
    assert len(list(Path('/proc/self/fd').iterdir()))==before
    check_complete_outputs(tmp_path,params,baseline,events,1005000,normalization)
    assert all(p.read_bytes()==raw for p,raw in original.items())


@pytest.mark.parametrize('tool',[cell,correlation,series])
def test_primary_mode_rejected_before_any_raw_or_arrays(tmp_path,tool):
    (tmp_path/'activity.json').write_text(json.dumps(dict(window=dict(start_tick=5000,end_tick=1005000,dt_seconds=.0001,endpoint='[start,end)'))))
    with pytest.raises(ValueError,match='requires bounded memory'):tool.analyze(SimpleNamespace(baseline=tmp_path,bounded_memory=False))
    assert sorted(p.name for p in tmp_path.iterdir())==['activity.json']


@LINUX
@pytest.mark.parametrize('end',[25000,105000])
def test_bounded_time_series_preserves_legacy_complete_outputs(tmp_path,monkeypatch,end):
    directory=os.environ.get('MAM_NORMALIZATION_DIR')
    if not directory:pytest.skip('requires retained official normalization via MAM_NORMALIZATION_DIR')
    normalization=Path(directory)/'mam-official-analysis-neuron-sizes-v1.json'
    params,audit,baseline,events=fixture(tmp_path,monkeypatch,end)
    monkeypatch.setattr(series,'plot',lambda *a,**k:None)
    for bounded in [False,True]:
        series.analyze(SimpleNamespace(baseline=baseline,native_audit=audit,model=None,results=None,normalization=normalization,output=tmp_path/str(bounded),bounded_memory=bounded))
    a=json.loads((tmp_path/'False/time-series.json').read_text());b=json.loads((tmp_path/'True/time-series.json').read_text())
    for key in a:
        if key not in ['analysis_seconds','implementation_sha256']:assert a[key]==b[key],key
    with np.load(tmp_path/'False/time-series.npz') as a,np.load(tmp_path/'True/time-series.npz') as b:
        assert a.files==b.files
        for key in a.files:
            assert a[key].dtype==b[key].dtype
            np.testing.assert_array_equal(a[key],b[key])
