"""Synthetic admission/metric tests; these fixtures are not primary results."""
import copy
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import compare_mam_primary_lags as m


def catalog(root):
    (root/'catalog.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=m.sha(p))
        for p in root.iterdir() if p.name!='catalog.json'}))


@pytest.fixture
def cases(tmp_path):
    template=json.loads((ROOT/'mpi-evidence/paper-propagation-v1/original/lags.json').read_text())
    areas=template['area_names'];keep=[i for i,a in enumerate(areas) if a!='MDP']
    identifiers={
        'original':dict(simulator='Published original rates',label=m.LABEL,revision=m.REV),
        'rust':dict(simulator='Rust',model_sha256=m.RUST_MODELS[100000],seed=1729),
        'native':dict(simulator='NEST',condition='metastable',parameters_sha256=m.PARAMETERS_SHA,
                      seed=1729,ranks=48,threads=4,duration_ms=100500.,dt_ms=.1,nest_version='3.10.0')}
    roots=[]
    for name,scale in [('original',1.),('rust',1.1),('native',.9)]:
        folder=tmp_path/name;folder.mkdir();roots.append(folder)
        h=np.linspace(-30,30,31);c=(h[:,None]-h[None,:])*scale
        full=np.full((32,32),np.nan);full[np.ix_(keep,keep)]=c
        fitted=m.fit_hierarchy(c);report=copy.deepcopy(template)
        report.update(identity=identifiers[name],observation_seconds=100.,
            levels_ms=fitted['levels_ms'].tolist(),normalized_levels=fitted['normalized_levels'].tolist(),
            ordered_early_to_late=[report['hierarchy_area_names'][i] for i in np.argsort(fitted['levels_ms'],kind='stable')])
        report['implementation_sha256']['mam_paper_propagation.py']=m.sha(ROOT/'tools/mam_paper_propagation.py')
        report['source_sha256']['reference_audit']='shared-fixture-reference'
        (folder/'lags.json').write_text(json.dumps(report))
        np.savez_compressed(folder/'lags.npz',lag_ms=full,retained_lag_ms=c,levels_ms=fitted['levels_ms'],normalized_levels=fitted['normalized_levels'])
        catalog(folder)
    return roots


def test_matched_comparison_does_not_rerun_any_short_windows(cases,tmp_path,monkeypatch):
    import compare_mam_paper_lags as legacy
    monkeypatch.setattr(legacy,'lag_matrix',lambda *a:pytest.fail('short windows rerun'))
    monkeypatch.setattr(m,'plot',lambda *a:None)
    result=m.compare(*cases,tmp_path/'comparison')
    assert result['equal_observation_duration'] and result['observation_seconds']==100.
    assert result['original_short_windows_recomputed'] is False and result['scientific_equivalence'] is False
    # Same normalized ordering can coexist with different absolute lag scales.
    metric=result['metrics']['rust_vs_original']
    assert metric['hierarchy_pearson']==pytest.approx(1.)
    h=np.linspace(-30,30,31);c=h[:,None]-h[None,:];mask=~np.eye(31,dtype=bool)
    assert metric['lag_mae_ms']==pytest.approx(np.mean(np.abs(c[mask])*.1))
    with np.load(tmp_path/'comparison/comparison-arrays.npz') as arrays:
        assert len(arrays.files)==9
        assert not any('block' in key for key in arrays.files)


@pytest.mark.parametrize('fault',['short-window','wrong-rust-model','wrong-seed','native-duration',
    'native-layout','reference','settings','missing-area','order-summary'])
def test_incompatible_artifact_rejected_before_output(cases,tmp_path,monkeypatch,fault):
    folder=cases[2] if fault.startswith('native-') else cases[1]
    path=folder/'lags.json';report=json.loads(path.read_text())
    if fault=='short-window':report['observation_seconds']=10.
    elif fault=='wrong-rust-model':report['identity']['model_sha256']=m.RUST_MODELS[10000]
    elif fault=='wrong-seed':report['identity']['seed']=1730
    elif fault=='native-duration':report['identity']['duration_ms']=10500.
    elif fault=='native-layout':report['identity']['threads']=1
    elif fault=='reference':report['source_sha256']['reference_audit']='different'
    elif fault=='settings':report['settings']['causal_inference']=True
    elif fault=='missing-area':report['area_names'].pop()
    else:report['ordered_early_to_late'].reverse()
    path.write_text(json.dumps(report));catalog(folder)
    monkeypatch.setattr(m,'plot',lambda *a:pytest.fail('plot before rejection'))
    with pytest.raises((ValueError,AssertionError)):m.compare(*cases,tmp_path/'comparison')
    assert not (tmp_path/'comparison').exists()


def test_npz_expansion_rejected_before_array_decode(cases,monkeypatch):
    folder=cases[1]
    with zipfile.ZipFile(folder/'lags.npz','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['lag_ms.npy','retained_lag_ms.npy','levels_ms.npy','normalized_levels.npy']:
            archive.writestr(name,b'0'*(2**20 if name=='levels_ms.npy' else 20))
    catalog(folder)
    monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('expanded array decoded'))
    with pytest.raises(ValueError,match='expansion'):m.read_artifact(folder,'rust')


def test_changed_levels_cannot_relabel_same_lag_matrix(cases):
    folder=cases[1]
    with np.load(folder/'lags.npz') as data:arrays={k:data[k] for k in data.files}
    arrays['levels_ms'][0]+=1
    np.savez_compressed(folder/'lags.npz',**arrays)
    report=json.loads((folder/'lags.json').read_text());report['levels_ms']=arrays['levels_ms'].tolist()
    (folder/'lags.json').write_text(json.dumps(report));catalog(folder)
    with pytest.raises(AssertionError):m.read_artifact(folder,'rust')
