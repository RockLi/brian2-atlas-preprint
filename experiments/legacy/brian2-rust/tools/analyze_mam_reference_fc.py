"""Explicit full native reference admission for the frozen FC calculation.

The original analyzer remains unchanged and seed1729-only. The analyze body
is identical except for the explicit reference seed admission; shared numeric
helpers, array checks, normalization and FC arithmetic are imported unchanged.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from analyze_mam_paper_fc import (sha,checked,compare_fc,RUST_MODELS,
    PARAMETERS_SHA,GENERATED_SHA,synaptic_area_inputs,functional_connectivity,SETTINGS)
from mam_launch_native_full_reference import label_for


def admit_reference(identity,bins,reference_seed):
    label_for(reference_seed)
    expected=dict(simulator='NEST',seed=reference_seed,ranks=48,threads=4,
                  duration_ms=100500,dt_ms=.1,nest_version='3.10.0')
    if bins!=100000 or any(identity.get(k)!=v for k,v in expected.items()):
        raise ValueError('full diagnostic NEST identity differs')


def analyze(series,matrices,reference_audit,output,reference_seed):
    label_for(reference_seed)
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
    if (report['schema']!='b2-mam-modern-paper-time-series-v1' or bins not in RUST_MODELS
            or span!=[500.5,bins+500.5] or report['bin_ms']!=1 or report['physical_tick_ms']!=.1
            or report['actual_simulated_neurons']!=4129924 or not report['frozen_histogram_exact']
            or not report['endpoint_count_identity_exact'] or GENERATED_SHA not in report['source_sha256'].values()):
        raise ValueError('unaudited or mismatched full-scale rate convention')
    identity=report['identity']
    admit_reference(identity,bins,reference_seed)
    if identity['simulator']=='Rust':
        if identity['model_sha256']!=RUST_MODELS[bins]:raise ValueError('unknown Rust run identity')
    elif identity['simulator']=='NEST':
        if identity['parameters_sha256']!=PARAMETERS_SHA or identity['condition']!='metastable' or identity['duration_ms']!=bins+500:
            raise ValueError('NEST condition or duration mismatch')
    else:raise ValueError('unknown simulator')
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
    for name in ['series','matrices','reference-audit','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-seed',type=int,choices=[1730,1731],required=True)
    a=p.parse_args();analyze(a.series,a.matrices,a.reference_audit,a.output,a.reference_seed)
