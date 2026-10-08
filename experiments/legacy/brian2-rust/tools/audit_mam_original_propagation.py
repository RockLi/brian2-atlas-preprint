"""Verify all samples of four published original Fig. 7 covariance references."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import time
import types
import numpy as np
from scipy.signal import find_peaks_cwt
from mam_paper_propagation import transforms,time_axis,smoothed_curve,select_peak,SETTINGS
from analyze_mam_paper_fc import checked
from audit_mam_original_interarea import REV,LABEL

PAIRS=[['V1','V1'],['V1','V2'],['V1','FEF'],['MIP','V1']]
SOURCE_HASHES={'helper.py':'9db8c61b7e56c2ab36c819b0059c0618a0eec236ea8fdf7a0073693e5f47a201',
 'correlation_analysis.py':'7b6c89887e0a9031cc9232a464077d5e1a8017722124a6a7094107a6e7f8472c',
 'Fig7_temporal_hierarchy.py':'f9161200ce0b9cef525cea52f5ecc5545857b0607f0a90899e19208602df90a4'}


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def selected_functions(path,names,scope):
    tree=ast.parse(path.read_text())
    body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in body}!=set(names):raise ValueError('source function set mismatch')
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),scope)


def source_oracle(source):
    for name,digest in SOURCE_HASHES.items():
        if sha(source/name)!=digest:raise ValueError('pinned source mismatch: '+name)
    helper={'np':np,'copy':copy};selected_functions(source/'helper.py',['centralize','calculate_fft'],helper)
    scope={'np':np,'cthlp':types.SimpleNamespace(**{name:helper[name] for name in ['centralize','calculate_fft']})}
    selected_functions(source/'correlation_analysis.py',['crossspec','crosscorrfunc'],scope)
    peaks={'np':np,'find_peaks_cwt':find_peaks_cwt};selected_functions(source/'Fig7_temporal_hierarchy.py',['correlation_peak'],peaks)
    return helper,scope,peaks


def audit(rates,reference,source,output):
    start=time.perf_counter();helper,scope,peak_scope=source_oracle(source)
    rc=json.loads((rates/'catalog.json').read_text());pc=json.loads((reference/'catalog.json').read_text())
    if rc['revision']!=REV or rc['labels']['metastable100']!=LABEL or pc['revision']!=REV or pc['label']!=LABEL or pc['failures'] or pc['pairs']!=PAIRS:
        raise ValueError('original reference identity differs')
    expected={'cross_correlation_'+'_'.join(pair)+'.npy' for pair in PAIRS}|{'cross_correlation_time.npy'}
    if set(pc['files'])!=expected:raise ValueError('reference file set differs')
    metadata=json.loads(checked(rates,rc['files'],'metastable100--rate_time_series_full_Parameters.json',2**20).read_text())
    if (metadata['t_min'],metadata['t_max'],metadata['resolution'])!=(500.,100500.,1.):
        raise ValueError('original window differs')
    published_time=np.load(checked(reference,pc['files'],'cross_correlation_time.npy',2**20),allow_pickle=False)
    t=time_axis(100000)
    if published_time.shape!=t.shape:raise ValueError('published lag axis size differs')
    time_error=float(np.max(np.abs(published_time-t)))
    rows=[];plots={};all_numerical=True;all_peaks=True;all_source=True
    for first,second in PAIRS:
        dat=[]
        for area in [first,second]:
            name='metastable100--rate_time_series_full_'+area+'.npy'
            values=np.load(checked(rates,rc['files'],name,2**20),allow_pickle=False)
            if values.shape!=(100000,) or values.dtype!=np.float64 or not np.isfinite(values).all():raise ValueError('invalid complete original rates')
            dat.append(values)
        original=np.load(checked(reference,pc['files'],'cross_correlation_'+first+'_'+second+'.npy',4*2**20),allow_pickle=False)
        if original.shape!=(2,2,100000) or not np.isfinite(original).all():raise ValueError('invalid published covariance')
        ft=transforms(dat);computed=np.array([[smoothed_curve(ft[i],ft[j]) for j in range(2)] for i in range(2)])
        centered=[helper['centralize'](row,units=True) for row in dat]
        freq,cross=scope['crossspec'](centered,1.)
        source_t,source_cross=scope['crosscorrfunc'](freq,cross)
        sigma=2.;tr=np.arange(-5.,5.);kernel=1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(tr**2/(2*sigma**2)))
        source_smoothed=np.array([[np.convolve(kernel,source_cross[i,j],mode='same') for j in range(2)] for i in range(2)])
        source_match=bool(np.allclose(computed,source_smoothed,rtol=1e-12,atol=1e-8) and np.array_equal(t,source_t));all_source&=source_match
        numerical=bool(np.allclose(computed,original,rtol=1e-12,atol=1e-8));all_numerical&=numerical
        peak_scope['cross_correlation']={'time':published_time,first:{second:original}}
        source_peak,source_amplitude=peak_scope['correlation_peak'](first,second,500.,100500.)
        source_peak=float(np.asarray(source_peak).reshape(-1)[0]);source_amplitude=float(np.asarray(source_amplitude).reshape(-1)[0])
        peak,amplitude,status=select_peak(t,computed[0,1],same_area=(first==second))
        peak_match=bool(peak==source_peak);all_peaks&=peak_match
        rows.append(dict(first=first,second=second,all_four_covariance_curves_match=numerical,
            maximum_absolute_covariance_error=float(np.max(np.abs(computed-original))),
            independent_pinned_source_match=source_match,
            peak_lag_ms=peak,source_on_published_lag_ms=source_peak,peak_lag_exact=peak_match,
            peak_amplitude=amplitude,source_on_published_peak_amplitude=source_amplitude,status=status))
        selection=(t>-110)&(t<110)
        plots[first+'_'+second]=computed[0,1,selection]
    report=dict(schema='b2-mam-original-propagation-audit-v1',scientific_equivalence=False,
        published_selected_covariances_reproduced=all_numerical and time_error<=1e-12,
        source_arithmetic_verified=all_source,selected_peak_lags_match=all_peaks,
        reference_pairs=4,total_verified_covariance_samples=1600000,observation_seconds=100.,
        maximum_absolute_time_axis_error=time_error,curve_numerical_rtol=1e-12,curve_numerical_atol=1e-8,
        pairs=rows,settings=SETTINGS,source_hashes=SOURCE_HASHES,
        reference_catalog_sha256=sha(reference/'catalog.json'),rates_catalog_sha256=sha(rates/'catalog.json'),
        implementation_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mam_paper_propagation.py')]},
        analysis_seconds=time.perf_counter()-start,
        scope='All samples of four original pair files and extracted pinned toolbox/figure functions. Peak comparison uses current SciPy on published curves, not recovered historical peak output. All-pair lag matrix, hierarchy, historical dependency identity, simulator/paper equivalence and causal inference remain separate.')
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'selected-curves.npz',time_ms=t[(t>-110)&(t<110)],**plots)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'catalog.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir()},indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not (report['published_selected_covariances_reproduced'] and all_source and all_peaks):raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['rates','reference','source','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.rates,a.reference,a.source,a.output)
