from pathlib import Path
from types import SimpleNamespace
import sys,json,gzip,hashlib
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import analyze_native_mam_activity as native
import analyze_mam_paper_cell_metrics as cli
import mam_correlation_stream as streaming
LINUX=pytest.mark.skipif(not sys.platform.startswith('linux'),reason='explicit Linux cache-release mode')


def native_fixture(tmp_path,monkeypatch,end):
    raw=gzip.decompress((ROOT/'mpi-evidence/region/full-parameters.json.gz').read_bytes());digest=hashlib.sha256(raw).hexdigest()
    audit=tmp_path/'audit';parameters=audit/'node25/fixture/parameters.json';parameters.parent.mkdir(parents=True);parameters.write_bytes(raw)
    run=audit/'node25/runs/fixture';run.mkdir(parents=True)
    events=np.array([(4999,0),(5000,0),(5001,1),(5020,0),(5040,1),(5050,0),(end-1,1),(end,0)],dtype=native.DTYPE)
    for rank in range(48):
        p=run/f'rank{rank}.events.bin';(events if rank==0 else np.empty(0,dtype=native.DTYPE)).tofile(p)
        report=dict(rank=rank,seed=1729,ranks=48,threads=4,duration_ms=end//10,dt_ms=.1,nest_version='3.10.0',parameters_sha256=digest,event_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),event_bytes=p.stat().st_size)
        p.with_name(f'rank{rank}.json').write_text(json.dumps(report))
    bins=np.bincount(events['tick'][events['tick']<end]//500,minlength=end//500)
    (audit/'summary.json').write_text(json.dumps(dict(passed=True,parameters_sha256=digest,spikes=len(events),physical_50ms_bin_counts=bins.tolist())))
    monkeypatch.setattr(native,'plot',lambda *args:None)
    baseline=tmp_path/'baseline';native.analyze(audit,baseline,end_tick=end)
    return audit,baseline,run/'rank0.events.bin',events.tobytes()


@LINUX
@pytest.mark.parametrize('end',[25000,105000])
def test_complete_native_cli_preserves_old_semantics_and_all_cell_arrays(tmp_path,monkeypatch,end):
    audit,baseline,raw,original=native_fixture(tmp_path,monkeypatch,end)
    before=len(list(Path('/proc/self/fd').iterdir()))
    for bounded in [False,True]:
        cli.analyze(SimpleNamespace(baseline=baseline,native_audit=audit,model=None,results=None,output=tmp_path/str(bounded),bounded_memory=bounded))
        assert len(list(Path('/proc/self/fd').iterdir()))==before
    old=json.loads((tmp_path/'False/paper-cell-metrics.json').read_text());new=json.loads((tmp_path/'True/paper-cell-metrics.json').read_text())
    assert new['bounded_memory'] and new['scratch_removed'] and len(new['implementation_sha256'])==5
    for key in old:
        if key not in ['populations','bounded_memory']:assert old[key]==new[key]
    for a,b in zip(old['populations'],new['populations'],strict=True):
        for k in a:
            if k in ['paper_lvr_mean','lvr_eligible_mean'] and a[k] is not None:np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-12)
            else:assert a[k]==b[k]
    with np.load(tmp_path/'False/cell-metrics.npz') as a,np.load(tmp_path/'True/cell-metrics.npz') as b:
        assert a.files==b.files and len(a.files)==762
        for k in a.files:
            assert a[k].dtype==b[k].dtype
            if k.endswith('_cell_lvr'):np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-12)
            else:np.testing.assert_array_equal(a[k],b[k])
    assert not (tmp_path/'True/population-events').exists() and raw.read_bytes()==original
    catalog=json.loads((tmp_path/'True/catalog.json').read_text())
    for name,r in catalog.items():
        p=tmp_path/'True'/name;assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']


@LINUX
def test_failed_regression_closes_baseline_and_leaves_no_success_report(tmp_path,monkeypatch):
    from mam_streamed_cell_metrics import CellMetrics
    audit,baseline,raw,original=native_fixture(tmp_path,monkeypatch,25000)
    old=CellMetrics.summarize
    def corrupt(self):
        summary,arrays=old(self);arrays['half_open_cell_counts'][0]+=1
        return summary,arrays
    monkeypatch.setattr(CellMetrics,'summarize',corrupt)
    before=len(list(Path('/proc/self/fd').iterdir()))
    output=tmp_path/'failed'
    with pytest.raises(AssertionError):cli.analyze(SimpleNamespace(baseline=baseline,native_audit=audit,model=None,results=None,output=output,bounded_memory=True))
    assert len(list(Path('/proc/self/fd').iterdir()))==before
    assert not (output/'paper-cell-metrics.json').exists() and not (output/'catalog.json').exists()
    assert raw.read_bytes()==original


def test_unsupported_cache_mode_fails_before_loading_inputs(monkeypatch,tmp_path):
    def unavailable():raise RuntimeError('unavailable POSIX cache advice')
    monkeypatch.setattr(streaming,'require_cache_release',unavailable)
    with pytest.raises(RuntimeError,match='unavailable POSIX'):
        cli.analyze(SimpleNamespace(bounded_memory=True,baseline=tmp_path/'absent'))
    assert not list(tmp_path.iterdir())
