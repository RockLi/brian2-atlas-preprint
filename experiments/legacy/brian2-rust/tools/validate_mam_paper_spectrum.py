"""Check explicit Welch settings against pinned source and independent FFT."""
import argparse
import ast
from copy import copy
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.signal import welch
from mam_paper_spectrum import spectrum,SETTINGS
from validate_mam_paper_rate_bins import HELPER_SHA


def manual_welch(rate):
    x=rate-np.mean(rate)
    w=np.hanning(1025)[:-1]
    segments=np.lib.stride_tricks.sliding_window_view(x,1024)[::24].copy()
    segments-=segments.mean(axis=1,keepdims=True)
    power=np.abs(np.fft.rfft(segments*w,axis=1))**2/(1000*np.sum(w*w))
    power[:,1:-1]*=2
    return np.fft.rfftfreq(1024,.001),power.mean(axis=0)


def validate(helper,wrapper,catalog,output):
    assert hashlib.sha256(helper.read_bytes()).hexdigest()==HELPER_SHA
    c=json.loads(catalog.read_text());key='figures/Schmidt2018_dyn/compute_power_spectrum.py'
    assert c['commit']=='0a658be40bef3249cbe452f38809edf7d2f524ba'
    assert hashlib.sha256(wrapper.read_bytes()).hexdigest()==c['files'][key]['sha256']=='a405ea3fe93f1f9eed852c1f6d70614fccf98793f2aee61dcdd6595448bff7d5'
    tree=ast.parse(wrapper.read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='welch']
    assert len(calls)==1 and {k.arg for k in calls[0].keywords}=={'fs','noverlap','nperseg'}
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    assert {k:assignments[k] for k in ['fs','noverlap','nperseg','window']}==dict(fs=1000.,noverlap=1000,nperseg=1024,window='boxcar')
    funcs=[n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='centralize'];assert len(funcs)==1
    env=dict(np=np,copy=copy);exec(compile(ast.Module(body=funcs,type_ignores=[]),str(helper),'exec'),env)
    rng=np.random.default_rng(20260909);cases=[np.zeros(2000),np.ones(2000)*3.,np.sin(2*np.pi*40*np.arange(2000)/1000),rng.poisson(3,2000).astype(float)]
    cases += [rng.normal(size=n) for n in [1024,1048,2000,5000,10000]]
    max_error=0.
    for rate in cases:
        f,p=spectrum(rate);ef,ep=welch(env['centralize'](rate,units=True),fs=1000.,noverlap=1000,nperseg=1024)
        np.testing.assert_array_equal(f,ef);np.testing.assert_array_equal(p,ep)
        mf,mp=manual_welch(rate);np.testing.assert_array_equal(f,mf)
        np.testing.assert_allclose(p,mp,rtol=5e-13,atol=1e-20)
        max_error=max(max_error,float(np.max(np.abs(p-mp))))
    r=dict(passed=True,fixture_cases=len(cases),explicit_vs_official_call_exact=True,
           independent_fft_max_absolute_error=max_error,independent_fft_tolerance=dict(relative=5e-13,absolute=1e-20,scope='floating arithmetic check, not scientific acceptance margin'),
           scipy=scipy.__version__,numpy=np.__version__,welch_signature=str(inspect.signature(welch)),settings=SETTINGS,
           official_declared_but_unused_window='boxcar',effective_window='periodic Hann',
           helper_sha256=HELPER_SHA,wrapper_sha256=c['files'][key]['sha256'],
           implementation_sha256=hashlib.sha256(Path(__file__).with_name('mam_paper_spectrum.py').read_bytes()).hexdigest(),
           scope='Available pinned modern wrapper under SciPy 1.18.1; historical SciPy environment and paper-level PSD reproduction remain unproven.')
    output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['helper','wrapper','catalog','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();validate(a.helper,a.wrapper,a.catalog,a.output)
