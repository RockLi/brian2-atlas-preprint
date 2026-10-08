"""Synthetic display/cohort fixtures; never biological reproduction evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def catalog(folder):
    result={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in folder.iterdir() if p.name!='catalog.json'}
    (folder/'catalog.json').write_text(json.dumps(result))


def series(folder,case,bins):
    p=folder/f'mam-paper-series-{case}-v1';p.mkdir(parents=True)
    areas=['V1','V2','MT','FEF','MIP','PITd']+[f'fixture{i}' for i in range(26)]
    rates=np.tile(np.r_[np.ones(bins//2),np.full(bins//2,3.)],(32,1))
    np.savez_compressed(p/'time-series.npz',area_rates_hz=rates,frequency_hz=np.arange(513)*1000/1024,power_hz2_per_hz=np.ones((32,513)))
    report=dict(area_names=areas,population_names=[f'p{i}' for i in range(254)],frozen_histogram_exact=True,endpoint_count_identity_exact=True,
                welch={'fixture':True},unrounded_population_sum=4130056.7202180736,wrapper_window_ms=f'(500,{500+bins}]',helper_histogram_range_ms=[500.5,500.5+bins],bin_ms=1,
                welch_segments=1+(bins-1024)//24,normalization='synthetic fixture',area_weighting='synthetic fixture',identity={'synthetic_fixture':case})
    (p/'time-series.json').write_text(json.dumps(report));catalog(p);return p


def correlation(folder,case,end,available=True):
    p=folder/f'mam-paper-correlation-{case}-v1';p.mkdir(parents=True)
    rows=[dict(name=f'mam_fixture_{i}',available=available,selected_cells=2 if available else 0,raw_events=3,
               frozen_uniform_correlation=.2 if available else None,mean_pairwise_correlation=.1 if available else None) for i in range(254)]
    report=dict(populations=rows,frozen_histograms_exact=True,scratch_removed=True,observation_ms=[500,end],endpoint=f'[500,{end}]',bin_ms=1,selection='fixture',calculation='fixture',helper_sha256='fixture',wrapper_sha256='fixture',toolbox_commit='fixture',available_populations=254 if available else 0,identity={'synthetic_fixture':case})
    (p/'correlation.json').write_text(json.dumps(report));catalog(p);return p


def capture_figures(monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    captured={}
    def save(self,path,**kwargs):
        captured[Path(path).name]=dict(title=self._suptitle.get_text(),visible_axes=sum(ax.get_visible() for ax in self.axes),
             last_x=[float(ax.lines[0].get_xdata()[-1]) for ax in self.axes if ax.lines])
    monkeypatch.setattr(Figure,'savefig',save);return captured


@pytest.mark.parametrize('bins',[2000,10000,100000])
def test_explicit_psd_cohort_uses_true_window_and_half_means(tmp_path,monkeypatch,bins):
    m=module('compare_mam_paper_time_series');figures=capture_figures(monkeypatch)
    cases=['fixture-native','fixture-rust']
    for case in cases:series(tmp_path,case,bins)
    m.compare(tmp_path,tmp_path/'out',cases=cases)
    r=json.loads((tmp_path/'out/comparison.json').read_text());values=r['areas'][0]['runs'][cases[0]]
    assert values['first_second_mean_hz' if bins==2000 else 'first_half_mean_hz']==1
    assert values['second_second_mean_hz' if bins==2000 else 'second_half_mean_hz']==3
    assert r['observation_seconds']==bins/1000 and len(r['cohort'])==2 and not r['scientific_equivalence']
    assert f'{bins/1000:g} s observation' in figures['spectra-full-band.png']['title']
    assert figures['rates-comparison.png']['visible_axes']==2
    assert figures['rates-comparison.png']['last_x']==pytest.approx([.5055+(bins//10-1)*.01]*2)


@pytest.mark.parametrize('tool,create',[('compare_mam_paper_time_series',series),('compare_mam_paper_correlation',correlation)])
def test_mixed_observation_lengths_are_rejected_before_output(tmp_path,tool,create):
    if create is series:
        create(tmp_path,'a',2000);create(tmp_path,'b',10000)
    else:
        create(tmp_path,'a',2500);create(tmp_path,'b',10500)
    with pytest.raises(AssertionError):module(tool).compare(tmp_path,tmp_path/'out',cases=['a','b'])
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('available',[True,False])
@pytest.mark.parametrize('end',[10500,100500])
def test_long_correlation_cohort_retains_unavailable_values(tmp_path,monkeypatch,available,end):
    m=module('compare_mam_paper_correlation');figures=capture_figures(monkeypatch)
    correlation(tmp_path,'fixture-native',end,available)
    m.compare(tmp_path,tmp_path/'out',cases=['fixture-native'])
    r=json.loads((tmp_path/'out/comparison.json').read_text())
    assert r['observation_ms']==[500,end] and r['cases'][0]['available']==(254 if available else 0)
    assert len(r['cases'][0]['unavailable'])==(0 if available else 254)
    assert figures['sampling-comparison.png']['visible_axes']==1
    assert f'{(end-500)/1000:g} s observations' in figures['sampling-comparison.png']['title']


@pytest.mark.parametrize('cases',[[],['a','a'],['../a'],['a']*7])
def test_invalid_cohort_rejected_without_reading_inputs(tmp_path,cases):
    for name in ['compare_mam_paper_time_series','compare_mam_paper_correlation']:
        with pytest.raises(ValueError):module(name).compare(tmp_path/'missing',tmp_path/'out',cases=cases)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('end_tick',[105000,1005000])
def test_native_pair_long_plot_uses_physical_window_and_catalog(tmp_path,monkeypatch,end_tick):
    m=module('compare_native_mam_activity');figures=capture_figures(monkeypatch)
    seconds=(end_tick-5000)/10000
    for simulator,name,offset in [('rust','mam-rust-ground1729-activity-v1',1),('nest','mam-native-ground1729-activity-v1',0)]:
        p=tmp_path/simulator;p.mkdir()
        report=json.loads((ROOT/'mpi-evidence/mam-ground-condition'/name/'activity.json').read_text())
        report['window'].update(end_tick=end_tick,raw_end_tick=end_tick-offset,seconds=seconds)
        # Synthetic zero traces test geometry only; the copied identity fields
        # are fixtures and are not presented as a new simulator observation.
        np.savez_compressed(p/'activity-arrays.npz',area_rates_hz=np.zeros((32,int(seconds*1000))))
        (p/'activity.json').write_text(json.dumps(report));catalog(p)
    m.compare(tmp_path/'rust',tmp_path/'nest',tmp_path/'out')
    result=json.loads((tmp_path/'out/comparison.json').read_text())
    assert result['physical_window']['seconds']==seconds and f'[0.5,{seconds+.5:g})' in result['scope']
    assert f'One {seconds:g} s observation' in figures['selected-area-comparison.png']['title']
    assert figures['selected-area-comparison.png']['last_x']==pytest.approx([seconds+.49]*5)
    (tmp_path/'rust/activity.json').write_text('{}')
    with pytest.raises(AssertionError):m.compare(tmp_path/'rust',tmp_path/'nest',tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
