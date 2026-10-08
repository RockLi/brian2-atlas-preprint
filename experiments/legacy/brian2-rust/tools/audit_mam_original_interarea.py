"""Reproduce published processed-input FC before applying it to our simulations."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import scipy
from mam_paper_interarea import functional_connectivity, SETTINGS

REV = '11fa93a4427a0e4e4de307ca7a5455e80265053a'
LABEL = '99c0024eacc275d13f719afd59357f7d12f02b77'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_checked(root, catalog, name):
    if Path(name).name != name:
        raise ValueError('flat catalog names required')
    path = root / name
    row = catalog['files'][name]
    if (path.is_symlink() or not 0 < path.stat().st_size <= 2**20
            or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']):
        raise ValueError('reference size/hash mismatch: ' + name)
    return path


def audit(source, series, output):
    start = time.perf_counter()
    cat = json.loads((source / 'catalog.json').read_text())
    rates_cat = json.loads((series / 'catalog.json').read_text())
    if (cat['revision'] != REV or cat['label'] != LABEL or cat['failures']
            or len(cat['files']) != 34 or rates_cat['revision'] != REV
            or rates_cat['labels']['metastable100'] != LABEL):
        raise ValueError('wrong or incomplete original reference catalog')
    params = json.loads(read_checked(source, cat, 'synaptic_input_Parameters.json').read_text())
    areas = params['areas']
    if len(areas) != 32 or len(set(areas)) != 32 or (params['t_min'], params['t_max'], params['resolution']) != (500., 100500., 1.):
        raise ValueError('wrong 100 s original synaptic-input metadata')
    expected = {'synaptic_input_' + a + '.npy' for a in areas} | {'synaptic_input_Parameters.json', 'functional_connectivity_synaptic_input.npy'}
    if set(cat['files']) != expected:
        raise ValueError('unexpected reference file set')
    raw_params = json.loads(read_checked(series, rates_cat, 'metastable100--rate_time_series_full_Parameters.json').read_text())
    rates = []
    currents = []
    for area in areas:
        for target, root, catalog, name in [
            (currents, source, cat, 'synaptic_input_' + area + '.npy'),
            (rates, series, rates_cat, 'metastable100--rate_time_series_full_' + area + '.npy')]:
            data = np.load(read_checked(root, catalog, name), allow_pickle=False)
            if data.shape != (100000,) or data.dtype != np.float64 or not np.isfinite(data).all():
                raise ValueError('invalid original area series: ' + name)
            target.append(data)
    reference = np.load(read_checked(source, cat, 'functional_connectivity_synaptic_input.npy'), allow_pickle=False)
    if reference.shape != (32, 32) or not np.isfinite(reference).all():
        raise ValueError('invalid published FC')
    computed, flat = functional_connectivity(currents)
    if len(flat):
        raise ValueError('flat original synaptic inputs')
    raw_fc, raw_flat = functional_connectivity(rates)
    error = np.abs(reference - computed)
    upper = np.triu_indices(32, 1)
    if len(raw_flat):
        raise ValueError('flat original area rates')
    method_delta = (computed - raw_fc)[upper]
    tolerance = 1e-12  # Numerical reconstruction only; not a scientific margin.
    report = dict(schema='b2-mam-original-interarea-audit-v1',
        processed_reference_fc_reproduced=bool(error.max() <= tolerance),
        scientific_equivalence=False, input_reconstruction_verified=False,
        area_names=areas, bins=100000, observation_seconds=100.,
        original_revision=REV, original_label=LABEL,
        synaptic_input_parameters=params, raw_rate_parameters=raw_params,
        original_source_sha256=sha(source / 'catalog.json'),
        original_rates_catalog_sha256=sha(series / 'catalog.json'),
        reference_fc_sha256=cat['files']['functional_connectivity_synaptic_input.npy']['sha256'],
        max_absolute_fc_error=float(error.max()),mean_absolute_fc_error=float(error.mean()),
        numerical_tolerance=tolerance, settings=SETTINGS,
        raw_rate_fc_difference=dict(off_diagonal_pairs=496,
            mean_absolute_difference=float(np.abs(method_delta).mean()),
            maximum_absolute_difference=float(np.abs(method_delta).max()),
            rmse=float(np.sqrt(np.mean(method_delta**2))),
            matrix_entry_correlation=float(np.corrcoef(computed[upper], raw_fc[upper])[0, 1])),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        implementation_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mam_paper_interarea.py')]},
        analysis_seconds=time.perf_counter()-start,
        scope='Reconstruction of published FC from its published synaptic-input series only. Does not establish population-rate-to-input reconstruction, BOLD, clustering, propagation, causal interactions, or Rust/NEST/paper equivalence. Matrix entries are dependent, not independent replicate samples.')
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'matrices.npz', published_fc=reference, recomputed_fc=computed, raw_area_rate_fc=raw_fc)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    catalog={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir()}
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['processed_reference_fc_reproduced']:
        raise SystemExit(2)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--series',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.source,args.series,args.output)
