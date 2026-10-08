"""Bind full official W/K/N matrices to the already verified primary parameters.

Parameter-only cached initialization; the NEST import shim forbids all API use.
No neurons, connections, theory integration or simulation are created.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types
import numpy as np

COMMIT = '0a658be40bef3249cbe452f38809edf7d2f524ba'
PARAMETERS_SHA = 'ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e'
GENERATED_SHA = '8c66bb68d55cf2bff222c67716952a2b6260576d6ba0a3650cc150af49f7c2cb'
K_SHA = 'cc970f6089dd1c997d7978c9fe6db44017cb478b7ad48e6963343d87fa9c3c58'
LABEL = 'c2776375839926e8a147b76a82ceab0a'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()


def export(source, parameters, output):
    if sha(parameters) != PARAMETERS_SHA:
        raise ValueError('primary parameter identity mismatch')
    params = json.loads(parameters.read_text())
    manifest = json.loads((source/'source-manifest.json').read_text())
    if manifest['commit'] != COMMIT or len(manifest['files']) != 60:
        raise ValueError('unexpected source manifest')
    for row in manifest['files']:
        path = source/row['path']
        if not path.resolve().is_relative_to(source.resolve()) or sha(path) != row['sha256']:
            raise ValueError('official source changed')
    generated = source/'config_files'/('custom_Data_Model_'+LABEL+'.json')
    config_path = source/'config_files'/(LABEL+'_config')
    if sha(generated) != GENERATED_SHA or json.loads(config_path.read_text()) != {'N_scaling':1.0, 'K_scaling':1.0}:
        raise ValueError('cached parameter model differs from admitted full-size export')
    output.mkdir(exist_ok=False)
    work = output/'official-source'; work.mkdir()
    for row in manifest['files']:
        dst = work/row['path']; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/row['path'],dst)
    (work/'config_files').mkdir(exist_ok=True)
    for path in [generated, config_path]:shutil.copyfile(path,work/'config_files'/path.name)
    config = types.ModuleType('config');config.base_path=str(work);config.data_path=str(output/'disabled')
    sys.modules['config']=config
    nest=types.ModuleType('nest')
    def forbidden(name):raise RuntimeError('NEST API forbidden during parameter-only export: '+name)
    nest.__getattr__=forbidden;sys.modules['nest']=nest
    sys.path.insert(0,str(work))
    from multiarea_model import MultiAreaModel
    model=MultiAreaModel(LABEL,simulation=False,theory=False,analysis=False)
    if sha(Path(model.params['connection_params']['K_stable'])) != K_SHA:
        raise ValueError('stabilized matrix identity mismatch')
    # Recreate the figure's dict access/order, independently cross-check vectors.
    names=[a+'-'+p for a in model.area_list for p in model.structure[a]]
    if names != list(model.structure_vec) or names != [p['name'] for p in params['populations']]:
        raise ValueError('official population ordering differs')
    w=np.array([[model.W[a][p][b][q] for b in model.area_list for q in model.structure[b]] for a in model.area_list for p in model.structure[a]])
    k=np.array([[model.K[a][p][b][q] for b in model.area_list for q in model.structure[b]] for a in model.area_list for p in model.structure[a]])
    neurons=np.array([model.N[a][p] for a in model.area_list for p in model.structure[a]])
    area_indices=np.array([model.area_list.index(a) for a in model.area_list for p in model.structure[a]],dtype=np.int64)
    np.testing.assert_array_equal(w,model.W_matrix[:,:-1]);np.testing.assert_array_equal(k,model.K_matrix[:,:-1])
    if w.shape != (254,254) or not np.isfinite(w).all() or not np.isfinite(k).all() or np.any(k<0):
        raise ValueError('invalid full official matrices')
    expected_neurons=np.array([p['count'] for p in params['populations']])
    np.testing.assert_array_equal(neurons.astype(np.int64),expected_neurons)
    for name in ['neuron_params','input_params']:
        if model.params[name] != params['params'][name]:raise ValueError(name+' differs')
    for key,value in params['params']['connection_params'].items():
        if key != 'K_stable' and model.params['connection_params'][key] != value:
            raise ValueError('connection parameter differs: '+key)
    covered=np.zeros_like(k,dtype=bool)
    for p in params['projections']:
        i,j=p['target'],p['source']
        if covered[i,j] or w[i,j]!=p['weight_mean_pA'] or k[i,j]!=p['indegree']:
            raise ValueError('exported projection differs from full official matrices')
        covered[i,j]=True
    missing_nonzero=np.argwhere((~covered)&(k!=0))
    output_npz=output/'matrices.npz'
    np.savez_compressed(output_npz,weights=w,indegrees=k,neuron_numbers=neurons,area_indices=area_indices)
    report=dict(schema='b2-mam-official-interarea-matrices-v1',source_commit=COMMIT,
        parameters_sha256=PARAMETERS_SHA,generated_data_sha256=GENERATED_SHA,
        stabilized_matrix_sha256=K_SHA,source_manifest_sha256=sha(source/'source-manifest.json'),
        population_names=names,area_names=model.area_list,tau_syn_ms=model.params['neuron_params']['single_neuron_dict']['tau_syn_ex'],
        covered_projection_count=int(covered.sum()),nonzero_mean_indegrees=int(np.count_nonzero(k)),
        nonzero_indegrees_omitted_from_simulation_export=missing_nonzero.tolist(),
        unrounded_neuron_sum=float(neurons.sum()),simulated_neuron_sum=int(expected_neurons.sum()),
        matrix_sha256=sha(output_npz),matrix_bytes=output_npz.stat().st_size,
        implementation_sha256=sha(Path(__file__)),complete_dict_vector_matrices_exact=True,
        scope='Full official recurrent mean W/K and unrounded N, checked against every primary exported projection. No simulation or scientific-equivalence claim.')
    (output/'matrices.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['area_names','population_names']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--parameters',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.source,a.parameters,a.output)
