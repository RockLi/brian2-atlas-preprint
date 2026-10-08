"""Apply unchanged FC arithmetic only to the pinned confirmation full-observation rates."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from mam_paper_interarea import synaptic_area_inputs, functional_connectivity, SETTINGS
from export_mam_interarea_matrices import PARAMETERS_SHA, GENERATED_SHA
from mam_confirmation_terminal_sync import load_identity

def admit_confirmation(identity,bins,pinned):
    if (bins!=100000 or identity.get('simulator')!='Rust'
            or identity.get('model_sha256')!=pinned['model_sha256']
            or type(identity.get('seed')) is not int
            or identity['seed']!=pinned['random_keys']['runtime_input']):
        raise ValueError('full confirmation model and actual runtime random key required')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def checked(root,cat,name,cap):
    path=root/name;row=cat[name]
    if path.is_symlink() or not 0<path.stat().st_size<=cap or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
        raise ValueError('artifact size/hash mismatch: '+name)
    return path


def compare_fc(left,right):
    left,right=np.asarray(left),np.asarray(right)
    if left.shape!=(32,32) or right.shape!=left.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('finite full 32-area matrices required')
    index=np.triu_indices(32,1);x,y=left[index],right[index];delta=x-y
    return dict(unordered_area_pairs=496,mean_absolute_difference=float(np.abs(delta).mean()),
                maximum_absolute_difference=float(np.abs(delta).max()),rmse=float(np.sqrt(np.mean(delta**2))),
                entry_correlation=float(np.corrcoef(x,y)[0,1]))


def analyze(series,matrices,reference_audit,output,identity_path):
    pinned=load_identity(identity_path)
    start=time.perf_counter()
    m=json.loads((matrices/'matrices.json').read_text());refcat=json.loads((reference_audit/'catalog.json').read_text())
    refreport=json.loads(checked(reference_audit,refcat,'report.json',2**20).read_text())
    if (not refreport['original_population_to_area_input_reproduced'] or not refreport['original_population_to_fc_reproduced']
            or refreport['matrix_metadata_sha256']!=sha(matrices/'matrices.json')
            or m['matrix_sha256']!=sha(matrices/'matrices.npz') or m['parameters_sha256']!=PARAMETERS_SHA
            or m['generated_data_sha256']!=GENERATED_SHA):
        raise ValueError('missing verified original population-to-FC reconstruction')
    cat=json.loads((series/'catalog.json').read_text())
    report=json.loads(checked(series,cat,'time-series.json',2**20).read_text())
    array_path=checked(series,cat,'time-series.npz',256*2**20)
    span=report['helper_histogram_range_ms'];bins=int(span[1]-span[0])
    if (report['schema']!='b2-mam-modern-paper-time-series-v1' or bins!=100000
            or span!=[500.5,bins+500.5] or report['bin_ms']!=1 or report['physical_tick_ms']!=.1
            or report['actual_simulated_neurons']!=4129924 or not report['frozen_histogram_exact']
            or not report['endpoint_count_identity_exact'] or GENERATED_SHA not in report['source_sha256'].values()):
        raise ValueError('unaudited or mismatched full-scale rate convention')
    identity=report['identity']
    admit_confirmation(identity,bins,pinned)
    with np.load(matrices/'matrices.npz',allow_pickle=False) as data:
        w,k,n,indices=[data[key] for key in ['weights','indegrees','neuron_numbers','area_indices']]
    names=['mam_'+name.replace('-','_') for name in m['population_names']]
    if len(report['population_names'])!=254 or set(report['population_names'])!=set(names):
        raise ValueError('population set differs')
    order=[report['population_names'].index(name) for name in names]
    with zipfile.ZipFile(array_path) as archive:
        entries=archive.infolist()
        if len(entries)>12 or sum(e.file_size for e in entries)>512*2**20 or any(e.file_size>256*2**20 for e in entries):
            raise ValueError('rate archive exceeds explicit expansion budget')
    with np.load(array_path,allow_pickle=False) as data:
        counts=data['population_counts'];rates=data['population_rates_hz']
        if counts.shape!=(254,bins) or rates.shape!=counts.shape or counts.dtype!=np.int64 or np.any(counts<0):
            raise ValueError('full population arrays required')
        rates=rates[order];counts=counts[order]
        np.testing.assert_array_equal(rates,counts/(n[:,None]*1./1000.))
    current=synaptic_area_inputs(rates,w,k,n,indices,m['tau_syn_ms'])
    fc,flat=functional_connectivity(current)
    if len(flat):raise ValueError('flat modeled area input')
    refpath=checked(reference_audit,refcat,'reconstructed.npz',64*2**20)
    with np.load(refpath,allow_pickle=False) as data:reference=data['reference_fc']
    if refreport['area_names']!=m['area_names']:raise ValueError('original FC area order differs')
    result=dict(schema='b2-mam-paper-fc-v1',scientific_equivalence=False,validated_rate_input=True,
        identity=identity,observation_seconds=bins/1000,reference_observation_seconds=100.,
        equal_observation_duration=(bins==100000),area_names=m['area_names'],settings=SETTINGS,
        parameters_sha256=PARAMETERS_SHA,normalization_bins_exact=True,
        compared_with_original=compare_fc(fc,reference),
        source_sha256=dict(series_report=sha(series/'time-series.json'),series_arrays=sha(array_path),
            matrices=sha(matrices/'matrices.npz'),matrices_metadata=sha(matrices/'matrices.json'),
            reference_report=sha(reference_audit/'report.json'),reference_arrays=sha(refpath)),
        implementation_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mam_paper_interarea.py')]},
        analysis_seconds=time.perf_counter()-start,
        scope='Descriptive Fig. 8 synaptic-input FC comparison using complete audited population-rate bins. Unequal duration is labeled, no scientific equivalence threshold is applied. Dependent matrix entries are not independent seed samples. No BOLD, lag, causal or speed/cost claim.')
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'fc.npz',area_synaptic_inputs=current,functional_connectivity=fc,reference_fc=reference)
    (output/'fc.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'catalog.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir()},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['series','matrices','reference-audit','output','identity']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();analyze(a.series,a.matrices,a.reference_audit,a.output,a.identity)
