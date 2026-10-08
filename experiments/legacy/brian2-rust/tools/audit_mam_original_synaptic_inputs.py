"""Verify original full population rates -> Fig. 8 current approximation -> FC."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mam_paper_interarea import synaptic_area_inputs, functional_connectivity, SETTINGS
from audit_mam_original_interarea import read_checked, REV, LABEL
from export_mam_interarea_matrices import PARAMETERS_SHA, GENERATED_SHA, K_SHA


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def comparison(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('invalid current comparison arrays')
    delta=actual-expected
    return dict(passed=bool(np.allclose(actual,expected,rtol=1e-12,atol=1e-8)),
        maximum_absolute_error=float(np.max(np.abs(delta))),
        relative_l2_error=float(np.linalg.norm(delta)/np.linalg.norm(expected)),
        relative_mean_error=float(delta.mean()/expected.mean()),
        actual_mean=float(actual.mean()),reference_mean=float(expected.mean()))


def audit(populations, reference, matrices, output):
    start=time.perf_counter()
    pc=json.loads((populations/'catalog.json').read_text());rc=json.loads((reference/'catalog.json').read_text())
    for cat in [pc,rc]:
        if cat['revision']!=REV or cat['label']!=LABEL or cat['failures']:
            raise ValueError('wrong or incomplete original reference catalog')
    meta=json.loads((matrices/'matrices.json').read_text())
    if (meta['schema']!='b2-mam-official-interarea-matrices-v1'
            or meta['parameters_sha256']!=PARAMETERS_SHA or meta['generated_data_sha256']!=GENERATED_SHA
            or meta['stabilized_matrix_sha256']!=K_SHA or not meta['complete_dict_vector_matrices_exact']
            or (matrices/'matrices.npz').stat().st_size!=meta['matrix_bytes']
            or sha(matrices/'matrices.npz')!=meta['matrix_sha256']):
        raise ValueError('full official matrix identity mismatch')
    with np.load(matrices/'matrices.npz',allow_pickle=False) as data:
        w,k,n,indices=[data[name] for name in ['weights','indegrees','neuron_numbers','area_indices']]
    if w.shape!=(254,254) or k.shape!=w.shape or n.shape!=(254,) or indices.shape!=(254,):
        raise ValueError('full 254-population matrices required')
    names=meta['population_names'];areas=meta['area_names']
    if len(names)!=254 or len(set(names))!=254 or len(areas)!=32 or len(set(areas))!=32:
        raise ValueError('population/area names malformed')
    file_names=['rate_time_series_full_'+name.replace('-','_')+'.npy' for name in names]
    if set(pc['files'])!=set(file_names):raise ValueError('original population file set differs')
    rates=np.empty((254,100000),dtype=np.float64)
    for i,name in enumerate(file_names):
        dat=np.load(read_checked(populations,pc,name),allow_pickle=False)
        if dat.shape!=(100000,) or dat.dtype!=np.float64 or not np.isfinite(dat).all() or np.any(dat<0):
            raise ValueError('invalid population rate array: '+name)
        rates[i]=dat
    rp=json.loads(read_checked(reference,rc,'synaptic_input_Parameters.json').read_text())
    if (rp['t_min'],rp['t_max'],rp['resolution'])!=(500.,100500.,1.) or set(rp['areas'])!=set(areas):
        raise ValueError('reference window or area set differs')
    t=time.perf_counter()
    current=synaptic_area_inputs(rates,w,k,n,indices,meta['tau_syn_ms'])
    compute_seconds=time.perf_counter()-t
    original=np.array([np.load(read_checked(reference,rc,'synaptic_input_'+a+'.npy'),allow_pickle=False) for a in areas])
    if original.shape!=(32,100000):raise ValueError('invalid original current arrays')
    rows=[dict(area=a,**comparison(x,y)) for a,x,y in zip(areas,current,original,strict=True)]
    fc,flat=functional_connectivity(current)
    if len(flat):raise ValueError('flat reconstructed input')
    original_fc=np.load(read_checked(reference,rc,'functional_connectivity_synaptic_input.npy'),allow_pickle=False)
    if sha(reference/'functional_connectivity_synaptic_input.npy')!='e7e30a416c20a586f0eb3db40dc7d7eaf4588b7132c5184bcd6875e9cb80ee89':
        raise ValueError('published FC identity mismatch')
    order=[rp['areas'].index(a) for a in areas];original_fc=original_fc[np.ix_(order,order)]
    fc_error=float(np.max(np.abs(fc-original_fc)))
    missing=meta['nonzero_indegrees_omitted_from_simulation_export']
    missing_expected_synapses=[float(k[i,j]*n[i]) for i,j in missing]
    report=dict(schema='b2-mam-original-synaptic-reconstruction-v1',scientific_equivalence=False,
        original_population_to_area_input_reproduced=all(r['passed'] for r in rows),
        original_population_to_fc_reproduced=bool(fc_error<=1e-12),
        current_numerical_rtol=1e-12,current_numerical_atol=1e-8,fc_numerical_atol=1e-12,
        population_count=254,area_count=32,bins=100000,observation_seconds=100.,
        source_population_catalog_sha256=sha(populations/'catalog.json'),
        source_reference_catalog_sha256=sha(reference/'catalog.json'),
        matrix_metadata_sha256=sha(matrices/'matrices.json'),matrix_sha256=meta['matrix_sha256'],
        source_revision=REV,source_label=LABEL,parameters_sha256=PARAMETERS_SHA,
        area_names=areas,areas=rows,maximum_absolute_fc_error=fc_error,
        omitted_integer_zero_projections=dict(count=len(missing),
            summed_unrounded_synapses=float(sum(missing_expected_synapses)),
            maximum_unrounded_synapses=float(max(missing_expected_synapses,default=0.)),
            all_below_one=all(0<x<1 for x in missing_expected_synapses)),
        compute_seconds=compute_seconds,analysis_seconds=time.perf_counter()-start,settings=SETTINGS,
        implementation_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mam_paper_interarea.py'),Path(__file__).with_name('audit_mam_original_interarea.py'),Path(__file__).with_name('export_mam_interarea_matrices.py')]},
        scope='All original processed 100 s population-rate samples to complete unrounded official recurrent-input approximation and FC. Raw original spikes were not retrieved; their rate-bin construction and historical environment remain separate. No Rust/NEST, BOLD, lag, multi-seed or speed/cost equivalence claim.')
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'reconstructed.npz',area_synaptic_inputs=current,functional_connectivity=fc,reference_fc=original_fc)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'catalog.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir()},indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='areas'},indent=2))
    if not report['original_population_to_area_input_reproduced'] or not report['original_population_to_fc_reproduced']:
        raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['populations','reference','matrices','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.populations,a.reference,a.matrices,a.output)
