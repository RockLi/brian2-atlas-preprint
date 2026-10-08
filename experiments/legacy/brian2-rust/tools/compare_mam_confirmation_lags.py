"""Matched 100 s propagation comparison for confirmation replicate1750.

Read only already analyzed lag artifacts; never recompute the ten short
original windows or claim scientific equivalence from one realization.
"""
from mam_confirmation_profile import REPLICATE,selected,worker_environment,IDENTITY_SHA_1751
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from analyze_mam_paper_fc import checked, sha
from compare_mam_primary_lags import plot
from mam_confirmation_analysis import accepted_raw
from mam_launch_confirmation_analysis import publish
from mam_launch_confirmation_run import read
from mam_confirmation_terminal_sync import IDENTITY_SHA
from audit_mam_original_interarea import LABEL, REV
from compare_mam_paper_lags import lag_stats
from export_mam_interarea_matrices import PARAMETERS_SHA
from mam_paper_propagation import fit_hierarchy, SETTINGS
IDENTITY_SHA=selected(IDENTITY_SHA,IDENTITY_SHA_1751)


def require(ok, message):
    if not ok: raise ValueError(message)


def read_artifact(root, case, admitted):
    require((root/'catalog.json').stat().st_size <= 2**20, 'oversized lag catalog')
    catalog=json.loads((root/'catalog.json').read_text())
    report_path=checked(root,catalog,'lags.json',2**20)
    array_path=checked(root,catalog,'lags.npz',2**20)
    report=json.loads(report_path.read_text())
    require(report['schema']=='b2-mam-paper-propagation-v1'
            and report['observation_seconds']==100.
            and report['excluded_from_hierarchy']==['MDP'], 'complete 100 s propagation artifact required')
    require(report['settings']==SETTINGS and report['implementation_sha256']['mam_paper_propagation.py']
            ==sha(Path(__file__).with_name('mam_paper_propagation.py')), 'propagation convention/source differs')
    identity=report['identity']
    if case=='original':
        require(identity==dict(simulator='Published original rates',label=LABEL,revision=REV), 'published reference identity differs')
    elif case=='rust':
        require(type(identity.get('seed')) is int and identity==dict(simulator='Rust',
                model_sha256=admitted['model_sha256'],seed=admitted['random_keys']['runtime_input']),
                'Rust confirmation identity differs; replicate label is not the runtime key')
    elif case=='native':
        require(identity==dict(simulator='NEST',condition='metastable',parameters_sha256=PARAMETERS_SHA,
                seed=1729,ranks=48,threads=4,duration_ms=100500.,dt_ms=.1,nest_version='3.10.0'),
                'NEST primary identity differs')
    else: raise ValueError('unknown propagation comparison case')
    expected={'lag_ms.npy','retained_lag_ms.npy','levels_ms.npy','normalized_levels.npy'}
    with zipfile.ZipFile(array_path) as archive:
        entries=archive.infolist()
        require(len(entries)==len(expected) and {r.filename for r in entries}==expected
                and sum(r.file_size for r in entries)<2**20, 'lag array expansion/coverage exceeds admission')
    with np.load(array_path,allow_pickle=False) as data:
        full,retained,levels,normalized=[data[k] for k in ['lag_ms','retained_lag_ms','levels_ms','normalized_levels']]
    areas=report['area_names'];names=report['hierarchy_area_names']
    require(len(areas)==32 and len(set(areas))==32 and areas.count('MDP')==1
            and names==[a for a in areas if a!='MDP'], 'brain area ordering/coverage differs')
    require(full.shape==(32,32) and retained.shape==(31,31) and levels.shape==normalized.shape==(31,),
            'complete lag/hierarchy arrays required')
    keep=[i for i,a in enumerate(areas) if a!='MDP']
    np.testing.assert_array_equal(retained,full[np.ix_(keep,keep)])
    np.testing.assert_array_equal(levels,np.asarray(report['levels_ms']))
    np.testing.assert_array_equal(normalized,np.asarray(report['normalized_levels']))
    require(np.isfinite(retained).all() and np.isfinite(levels).all() and np.isfinite(normalized).all(),
            'missing lag/hierarchy entries may not be imputed')
    np.testing.assert_array_equal(np.diag(retained),np.zeros(31))
    fitted=fit_hierarchy(retained)
    require(fitted['normalization_defined'], 'undefined hierarchy')
    # Numerical consistency of the existing least-squares summary, not a
    # simulator-equivalence margin. Same tolerance as the retained audit core.
    np.testing.assert_allclose(levels,fitted['levels_ms'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(normalized,fitted['normalized_levels'],rtol=1e-12,atol=1e-12)
    require(report['ordered_early_to_late']==[names[i] for i in np.argsort(levels,kind='stable')],
            'reported temporal order differs from levels')
    return report,retained,levels,normalized,dict(report=sha(report_path),arrays=sha(array_path),catalog=sha(root/'catalog.json'))


def compare(original,rust,native,output,case):
    if not (case/'science/complete.json').is_file():
        return dict(ready=False,comparison_written=False,reason='Full six-metric analysis is pending.')
    admitted,science=accepted_science(case)
    require(sha(rust/'catalog.json')==science['catalogs']['lags'], 'Rust lag artifact is not the accepted analysis')
    require(not output.exists(),'comparison output already exists')
    rows=[read_artifact(root,kind,admitted) for root,kind in [(original,'original'),(rust,'rust'),(native,'native')]]
    reports,curves,levels,normalized,hashes=map(list,zip(*rows))
    for report in reports[1:]:
        for key in ['area_names','hierarchy_area_names','settings']:
            require(report[key]==reports[0][key],'incompatible '+key)
        for key in ['reference_audit','matrices_metadata']:
            require(report['source_sha256'][key]==reports[0]['source_sha256'][key], 'different reference evidence: '+key)
    pairs={'rust_vs_original':(1,0),'native_vs_original':(2,0),'rust_vs_native':(1,2)}
    metrics={name:lag_stats(curves[i],levels[i],curves[j],levels[j]) for name,(i,j) in pairs.items()}
    result=dict(schema='b2-mam-confirmation-propagation-comparison-v1',scientific_equivalence=False,
        equal_observation_duration=True,observation_seconds=100.,rust_replicate=REPLICATE,native_seed=1729,
        identity_sha256=IDENTITY_SHA,science_report_sha256=sha(case/'science/report.json'),
        formal_equivalence_acceptance=False,performance_cost_acceptance=False,
        hierarchy_area_names=reports[0]['hierarchy_area_names'],metrics=metrics,
        identities={case:report['identity'] for case,report in zip(['original','rust','native'],reports,strict=True)},
        source_sha256=dict(zip(['original','rust','native'],hashes,strict=True)),
        original_short_windows_recomputed=False,acceptance_thresholds_fitted=False,
        implementation_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('compare_mam_paper_lags.py'),
            Path(__file__).with_name('mam_paper_propagation.py'),Path(__file__).with_name('compare_mam_primary_lags.py')]},
        scope='Descriptive matched-duration comparison: confirmation Rust1750 and historical NEST1729 and the published '
              'original rate reanalysis, all 100 s after the transient. The 930 directed matrix entries are dependent. '
              'Original full hierarchy is an available-source reanalysis; only four stored covariance pairs were directly '
              'audited. No recomputed short-window ranges, independent-seed confidence intervals, causal conclusion, '
              'scientific equivalence or performance/cost acceptance.')
    # Check serializability/non-finite metrics before creating any output.
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    output.mkdir()
    (output/'comparison.json').write_text(text)
    np.savez_compressed(output/'comparison-arrays.npz',
        **{case+'_'+kind:values[i] for i,case in enumerate(['original','rust','native'])
           for kind,values in [('lags',curves),('levels_ms',levels),('normalized_levels',normalized)]})
    plot(reports,curves,normalized,metrics,output)
    (output/'catalog.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in output.iterdir()},indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result



def accepted_science(case):
    """Replay guarded engineering/science decisions using compact evidence only."""
    completion_sha=sha(case/'completion.json')
    identity,_,_=accepted_raw(case,completion_sha)
    root=case/'science';complete=read(root/'complete.json')
    require(complete['replicate']==REPLICATE and complete['full_descriptive_analysis_complete'] is True
        and complete['scientific_acceptance'] is False and complete['performance_cost_acceptance'] is False
        and 0<complete['total_seconds']<=10800,'bounded descriptive science completion required')
    report=read(root/'report.json',complete['report_sha256'])
    required={'intent.json','admission.json','controller.json','pending.json','guard.json'}
    required.update(stage+'/catalog.json' for stage in ['activity','cell','correlation','series','fc','lags'])
    require(required<=set(report['evidence_sha256']),'science evidence coverage differs')
    for name,digest in report['evidence_sha256'].items():
        path=root/name
        require(path.resolve().is_relative_to(root.resolve()) and path.is_file() and not path.is_symlink()
            and path.stat().st_size<2*2**20 and sha(path)==digest,'science evidence changed')
    a=read(root/'admission.json')
    reproduced=publish(read(root/'pending.json'),read(root/'guard.json'),read(root/'controller.json'),
        completion_sha=completion_sha,identity=identity,raw_report_sha=sha(case/'raw/report.json'),
        previous=a['previous_seconds'],limit=a['limit_seconds'],catalog=a['source_catalog'],
        elapsed=report['elapsed_through_collection_seconds'])
    require(all(report[k]==v for k,v in reproduced.items()),'science decision not reproducible')
    require(complete['total_seconds']>=report['elapsed_through_collection_seconds'],'science elapsed accounting differs')
    for stage,digest in report['catalogs'].items():
        require(sha(root/stage/'catalog.json')==digest,'science catalog differs')
    return identity,report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['original','rust','native','output','case']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.original,args.rust,args.native,args.output,args.case)
    if result.get('ready') is False:
        print(json.dumps(result));raise SystemExit(2)
