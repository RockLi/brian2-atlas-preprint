"""Export official MAM population/projection parameters without creating neurons.

Run in a fresh process with pandas, scipy, nested_dict and dicthash installed.
The source bundle must be pinned and verified. No NEST simulation is performed:
a deliberately inert import shim only selects the official NEST 3 parameter path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types

COMMIT = '0a658be40bef3249cbe452f38809edf7d2f524ba'
STABILIZED_MATRIX_SHA256 = 'cc970f6089dd1c997d7978c9fe6db44017cb478b7ad48e6963343d87fa9c3c58'
CONDITIONS = {'metastable': (1.9, 2.0), 'stabilized-ground': (1.0, 1.0)}


def condition_overrides(condition):
    if condition not in CONDITIONS:
        raise ValueError('unknown stabilized condition')
    # Preserve the established default export path. Its resolved factors are
    # independently checked below so a label cannot silently select defaults.
    if condition == 'metastable':
        return {}
    cc, inhibitory = CONDITIONS[condition]
    return {'connection_params': {'cc_weights_factor': cc,
                                  'cc_weights_I_factor': inhibitory}}


def resolved_condition(params, condition):
    if condition not in CONDITIONS:
        raise ValueError('unknown stabilized condition')
    conn = params['connection_params']
    expected = dict(g=-11., fac_nu_ext_TH=1.2, fac_nu_ext_5E=1.125,
                    fac_nu_ext_6E=1.41666667, av_indegree_V1=3950.,
                    cc_weights_factor=CONDITIONS[condition][0],
                    cc_weights_I_factor=CONDITIONS[condition][1])
    if any(conn[k] != v for k, v in expected.items()):
        raise ValueError('resolved parameters do not select declared stabilized condition')
    if (params['input_params']['rate_ext'] != 10.
            or params['neuron_params']['V0_mean'] != -150.
            or params['neuron_params']['V0_sd'] != 50.):
        raise ValueError('unexpected stabilized input/initial conditions')
    return expected


def export(source, output, n_scaling=1.0, k_scaling=1.0, *, condition='metastable'):
    overrides = condition_overrides(condition)
    if condition == 'stabilized-ground' and (n_scaling != 1 or k_scaling != 1):
        raise ValueError('scaled ground export requires its own fullscale rate reference')
    manifest=json.loads((source/'source-manifest.json').read_text())
    if manifest['commit'] != COMMIT:
        raise ValueError('unexpected official source version')
    for item in manifest['files']:
        path=source/item['path']
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError('source path escapes bundle')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('official source changed: '+item['path'])
    if not 0 < n_scaling <= 1 or not 0 < k_scaling <= 1:
        raise ValueError('scales must be in (0,1]')
    output.mkdir(parents=True,exist_ok=False)
    work=output/'official-source'
    shutil.copytree(source,work)
    (work/'config_files').mkdir(exist_ok=True)
    config=types.ModuleType('config')
    config.base_path=str(work);config.data_path=str(output/'nest-output-disabled')
    sys.modules['config']=config
    nest=types.ModuleType('nest')
    def forbidden(name):
        raise RuntimeError('NEST simulation unavailable in parameter-only export: '+name)
    nest.__getattr__=forbidden
    sys.modules['nest']=nest
    sys.path.insert(0,str(work))
    from multiarea_model import MultiAreaModel
    import numpy as np
    spec={'N_scaling':n_scaling,'K_scaling':k_scaling,**overrides}
    if n_scaling!=1 or k_scaling!=1:
        spec['fullscale_rates']=str(work/'tests/fullscale_rates.json')
    model=MultiAreaModel(spec,simulation=False,theory=False,analysis=False)
    resolved = resolved_condition(model.params, condition)
    matrix_path = Path(model.params['connection_params']['K_stable'])
    matrix_sha = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    if matrix_sha != STABILIZED_MATRIX_SHA256:
        raise ValueError('unexpected stabilized connectivity matrix')
    populations=[];projections=[]
    for index, name in enumerate(model.structure_vec):
        area,pop=name.split('-')
        count=int(model.N[area][pop])
        populations.append({'name':name,'area':area,'population':pop,'count':count,
                            'external_indegree':float(model.K_matrix[index,-1]),
                            'external_weight_pA':float(model.W_matrix[index,-1]),
                            'dc_pA':float(model.add_DC_drive[index])})
    for target,t in enumerate(populations):
        for source_index,s in enumerate(populations):
            count=int(model.syn_matrix[target,source_index])
            if count<=0:continue
            local=t['area']==s['area'];excitatory=s['population'].endswith('E')
            delay=(model.params['delay_params']['delay_e' if excitatory else 'delay_i'] if local
                   else model.distances[t['area']][s['area']]/model.params['delay_params']['interarea_speed'])
            projections.append({'source':source_index,'target':target,'count':count,
                'weight_mean_pA':float(model.W_matrix[target,source_index]),
                'indegree':float(model.K_matrix[target,source_index]),
                'weight_sd_pA':float(model.W_sd[t['area']][t['population']][s['area']][s['population']]),
                'excitatory':excitatory if local else True,
                'delay_mean_ms':float(delay),'delay_sd_ms':float(delay*model.params['delay_params']['delay_rel'])})
    report={'schema':'b2-official-mam-parameters-v1','source_commit':COMMIT,
            'source_manifest_sha256':hashlib.sha256((source/'source-manifest.json').read_bytes()).hexdigest(),
            'scientific_scope':'Official parameter generation only; no NEST or Brian2 simulation',
            'nest_parameter_path':3,'state':condition,'N_scaling':n_scaling,'K_scaling':k_scaling,
            'params':model.params,'populations':populations,'projections':projections,
            'total_neurons':sum(p['count'] for p in populations),
            'total_recurrent_synapses':sum(p['count'] for p in projections),
            'max_projection_synapses':max(p['count'] for p in projections),
            'numpy_version':np.__version__}
    (output/'parameters.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    condition_report = dict(schema='b2-mam-stabilized-condition-v1', condition=condition,
        source_commit=COMMIT,source_manifest_sha256=report['source_manifest_sha256'],
        parameters_sha256=hashlib.sha256((output/'parameters.json').read_bytes()).hexdigest(),
        declared_overrides=overrides,resolved_connection_fields=resolved,
        stabilized_matrix_sha256=matrix_sha,neuron_params=model.params['neuron_params'],
        input_params=model.params['input_params'],N_scaling=n_scaling,K_scaling=k_scaling,
        neurons=report['total_neurons'],recurrent_synapses=report['total_recurrent_synapses'],
        simulated=False,scientific_equivalence=False,
        scope='Parameter-only explicit stabilized condition; runtime admission, numerical and activity validation are separate.')
    (output/'condition-manifest.json').write_text(json.dumps(condition_report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['source_commit','total_neurons','total_recurrent_synapses','max_projection_synapses']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n-scaling',type=float,default=1.0);p.add_argument('--k-scaling',type=float,default=1.0)
    p.add_argument('--condition',choices=CONDITIONS,default='metastable')
    a=p.parse_args();export(a.source.resolve(),a.output.resolve(),a.n_scaling,a.k_scaling,condition=a.condition)
