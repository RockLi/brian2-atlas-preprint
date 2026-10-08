"""Audit a fresh official ground export against the pinned full metastable one."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

BASELINE_SHA='ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e'
MATRIX_SHA='cc970f6089dd1c997d7978c9fe6db44017cb478b7ad48e6963343d87fa9c3c58'


def compare_parameters(baseline, ground):
    if (baseline['state'],ground['state']) != ('metastable','stabilized-ground'):
        raise ValueError('unexpected condition labels')
    bconn=baseline['params']['connection_params'];gconn=ground['params']['connection_params']
    if (bconn['cc_weights_factor'],bconn['cc_weights_I_factor'],gconn['cc_weights_factor'],gconn['cc_weights_I_factor']) != (1.9,2.,1.,1.):
        raise ValueError('condition factors must be explicit and resolved')
    restored=copy.deepcopy(ground)
    restored['state']='metastable'
    rconn=restored['params']['connection_params']
    for key in ['cc_weights_factor','cc_weights_I_factor','K_stable']:rconn[key]=bconn[key]
    # Full scale contains no location-dependent fullscale_rates path. Matrix
    # identity is verified by caller against the pinned prior source audit.
    if restored['params'] != baseline['params']:
        raise ValueError('undeclared resolved parameter change')
    if ground['populations'] != baseline['populations']:
        raise ValueError('neuron/external/DC/population identity changed')
    if len(ground['projections']) != len(baseline['projections']):
        raise ValueError('projection universe changed')
    classes={'local':0,'cc_to_E':0,'cc_to_I':0};exact_weights=0;max_relative=0.;max_absolute=0.
    for i,(b,g) in enumerate(zip(baseline['projections'],ground['projections'],strict=True)):
        source,target=ground['populations'][g['source']],ground['populations'][g['target']]
        category='local' if source['area']==target['area'] else ('cc_to_I' if target['population'].endswith('I') else 'cc_to_E')
        classes[category]+=1
        restored_projection=restored['projections'][i]
        for key in ['weight_mean_pA','weight_sd_pA']:
            value=g[key]
            if category!='local':value*=1.9
            if category=='cc_to_I':value*=2.
            if not math.isfinite(value) or value != b[key]:
                raise ValueError('unexpected weight transformation')
            if category=='local' and g[key]!=b[key]:raise ValueError('local weight changed')
            exact_weights+=value==b[key];max_absolute=max(max_absolute,abs(value-b[key]));max_relative=max(max_relative,abs(value-b[key])/abs(b[key]) if b[key] else 0.)
            restored_projection[key]=b[key]
        if restored_projection!=b:raise ValueError('undeclared projection count/delay/identity change')
    # NumPy version is provenance, never silently discarded; this audit requires
    # the same version as the original export to avoid hidden generator changes.
    if restored!=baseline:raise ValueError('undeclared top-level export change')
    return dict(only_declared_condition_changes=True,populations=len(ground['populations']),
                projections=len(ground['projections']),projection_classes=classes,
                weight_fields_reconstructed_exactly=exact_weights,
                max_reconstruction_relative_error=max_relative,max_reconstruction_absolute_pA=max_absolute,
                roundoff_tolerance='zero: scalar weight reconstruction must be exact',
                populations_external_input_and_dc_exact=True,connection_counts_and_delays_exact=True)


def audit(baseline_path,ground_dir,output,matrix_path=None):
    raw=baseline_path.read_bytes();assert hashlib.sha256(raw).hexdigest()==BASELINE_SHA
    baseline=json.loads(raw);p=ground_dir/'parameters.json';ground=json.loads(p.read_text())
    manifest=json.loads((ground_dir/'condition-manifest.json').read_text())
    assert manifest['condition']=='stabilized-ground' and manifest['parameters_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    assert manifest['source_manifest_sha256']==ground['source_manifest_sha256']==baseline['source_manifest_sha256']
    matrix_path = matrix_path or Path(ground['params']['connection_params']['K_stable'])
    assert hashlib.sha256(matrix_path.read_bytes()).hexdigest()==manifest['stabilized_matrix_sha256']==MATRIX_SHA
    assert ground['N_scaling']==ground['K_scaling']==1 and ground['total_neurons']==4129924 and ground['total_recurrent_synapses']==24126516728
    result=compare_parameters(baseline,ground)
    result.update(passed=True,baseline_parameters_sha256=BASELINE_SHA,ground_parameters_sha256=manifest['parameters_sha256'],
                  stabilized_matrix_sha256=MATRIX_SHA,verified_matrix_path=str(matrix_path),condition_manifest_sha256=hashlib.sha256((ground_dir/'condition-manifest.json').read_bytes()).hexdigest(),
                  neurons=ground['total_neurons'],recurrent_synapses=ground['total_recurrent_synapses'],
                  scope='Fresh official parameter generation; no neuron allocation or simulation. Does not establish scientific reproduction.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','ground-dir','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--matrix',type=Path,help='Relocated matrix file; its pinned content hash is still mandatory')
    a=p.parse_args();audit(a.baseline,a.ground_dir,a.output,a.matrix)
