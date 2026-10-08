"""Six fixed descriptive estimators after accepted full confirmation raw output.

One sequential attempt under the separate three-hour science guard. Numerical
estimators remain unchanged; the FC admission alone recognizes the new identity.
"""
from mam_confirmation_profile import REPLICATE,selected,worker_environment,IDENTITY_SHA_1751
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from mam_confirmation_terminal_sync import IDENTITY_SHA,parse,require
from mam_confirmation_raw import terminal_gate,terminal_names
from mam_launch_confirmation_raw import publish as publish_raw
from mam_launch_confirmation_run import CASE,read,sha,write
from mam_primary_analysis_pipeline import execute_sequence,TOTAL_SECONDS
IDENTITY_SHA=selected(IDENTITY_SHA,IDENTITY_SHA_1751)

BASE=Path('/data/brick2/brian2-mpi-region-20260907')
NORMALIZATION='normalization/mam-official-analysis-neuron-sizes-v1.json'
REQUIRED={
    'activity':{'activity.json','activity-arrays.npz','activity-overview.png','spike-rasters.png'},
    'cell':{'paper-cell-metrics.json','cell-metrics.npz'},
    'correlation':{'correlation.json','selection.npz'},
    'series':{'time-series.json','time-series.npz','spectrum.png'},
    'fc':{'fc.json','fc.npz'},'lags':{'lags.json','lags.npz'},
}


def accepted_raw(case,completion_sha):
    """Reproduce the guarded raw decision; never opens simulation payloads."""
    c=read(case/'completion.json',completion_sha)
    require(c['schema']=='b2-mam-confirmation-engineering-completion-v1' and c['case_id']==CASE
            and c['replicate']==REPLICATE and c['identity_sha256']==IDENTITY_SHA
            and c['terminal_resource_audit_passed'] is True and c['raw_output_audit_passed'] is True
            and c['scientific_acceptance'] is False and c['performance_cost_acceptance'] is False,
            'full confirmation engineering completion required')
    expected=set(terminal_names(read(case/'terminal/report.json'))[:5])|{'raw/report.json','raw/intent.json'}
    require(set(c['input_sha256'])==expected,'engineering completion input coverage')
    for name,digest in c['input_sha256'].items():
        require(sha(case/name)==digest,'engineering completion input changed')
    raw=case/'raw';binding_sha=sha(raw/'binding.json')
    binding,identity,hosts,terminal=terminal_gate(case,raw/'binding.json',binding_sha)
    require(binding['protocol_sha256']==c['protocol_sha256'],'raw completion protocol mismatch')
    report=read(raw/'report.json')
    expected={'binding.json','intent.json','admission.json','controller.json','pending.json','guard.json','attempt.json'}
    require(set(report['evidence_sha256'])==expected,'raw stage evidence coverage')
    for name,digest in report['evidence_sha256'].items():
        require(sha(raw/name)==digest,'raw stage evidence changed')
    admission=read(raw/'admission.json');catalog=admission['source_catalog']
    worker=Path(__file__).with_name('mam_confirmation_raw.py')
    require(catalog['tools/mam_confirmation_raw.py']['sha256']==sha(worker),'raw worker source differs')
    attempt=read(raw/'attempt.json');intent=read(raw/'intent.json')
    require(attempt['binding_sha256']==intent['binding_sha256']==binding_sha
            and attempt['automatic_retry'] is False and intent['automatic_retry'] is False
            and intent['attempts']==1 and intent['total_stage_seconds']==7200,'finite raw attempt differs')
    reproduced=publish_raw(read(raw/'pending.json'),read(raw/'guard.json'),read(raw/'controller.json'),
        binding_sha=binding_sha,terminal=terminal,identity=identity,protocol_sha=c['protocol_sha256'],
        raw_source_sha=sha(worker),elapsed=report['elapsed_through_guard_collection_seconds'])
    require(all(report[k]==v for k,v in reproduced.items()),'raw publication not reproducible')
    return identity,report,terminal


def stages(source,identity):
    output=Path(identity['analysis'])
    common=['--model',identity['source_artifact']+'/model.json','--results',identity['output'],'--bounded-memory']
    baseline=['--baseline',str(output/'activity')]
    shared=['--series',str(output/'series'),'--matrices',str(source/'interarea/matrices')]
    plan=[('activity',2400,'analyze_multi_area_activity.py',common+['--spike-tick-offset','1','--end-tick','1005000']),
        ('cell',1800,'analyze_mam_paper_cell_metrics.py',common+baseline),
        ('correlation',2400,'analyze_mam_paper_correlation.py',common+baseline),
        ('series',1800,'analyze_mam_paper_time_series.py',common+baseline+['--normalization',str(source/NORMALIZATION)]),
        ('fc',180,'analyze_mam_confirmation_fc.py',shared+['--reference-audit',str(source/'interarea/fc-reference'),
            '--identity',str(source/'case/identity.json')]),
        ('lags',180,'analyze_mam_paper_lags.py',shared+['--reference-audit',str(source/'interarea/lag-reference'),
            '--fc-audit',str(output/'fc')])]
    return [(name,cap,[sys.executable,str(source/'tools'/tool),*args,'--output',str(output/name)]) for name,cap,tool,args in plan]


def validate(name,source,identity,raw):
    output=Path(identity['analysis']);directory=output/name
    catalog=read(directory/'catalog.json');require(REQUIRED[name]<=set(catalog),'required estimator outputs missing')
    total=0
    for filename,row in catalog.items():
        require(Path(filename).name==filename and not filename.startswith('._'),'unsafe estimator artifact')
        p=directory/filename
        require(p.is_file() and not p.is_symlink() and 0<p.stat().st_size==row['bytes']<=512*2**20
                and sha(p)==row['sha256'],'estimator artifact size/hash differs')
        total+=row['bytes']
    require(total<=2*2**30,'derived output budget exceeded')
    baseline=read(output/'activity/activity.json')
    require(baseline['model_sha256']==identity['model_sha256'] and baseline['result_sha256']==raw['dump_sha256']
            and baseline['bounded_memory'] is True and baseline['population_count']==254
            and baseline['neurons']==4129924,'activity not bound to full raw output')
    w=baseline['window']
    require((w['start_tick'],w['end_tick'],w['seconds'],w['spike_tick_offset'],w['endpoint'])
            ==(5000,1005000,100.,1,'[start,end)'),'Rust physical observation differs')
    if name=='activity':return
    filename=dict(cell='paper-cell-metrics.json',correlation='correlation.json',series='time-series.json',fc='fc.json',lags='lags.json')[name]
    result=read(directory/filename);observed=result['identity']
    require(observed['simulator']=='Rust' and observed['model_sha256']==identity['model_sha256']
            and result['scientific_equivalence'] is False,'estimator run or claim differs')
    if name!='cell':require(type(observed['seed']) is int and observed['seed']==identity['random_keys']['runtime_input'],
            'actual random key differs; replicate label is not runtime seed')
    if name in ['cell','correlation','series']:
        require(result['bounded_memory'] is True and result['source_sha256'][str(output/'activity/activity.json')]
                ==sha(output/'activity/activity.json'),'baseline source/bounded analysis differs')
        if name=='cell':require(result['window']['end_tick']==1005000 and result['scratch_removed'] is True,'cell scope differs')
        elif name=='correlation':require(result['observation_ms']==[500,100500] and result['raw_events']==raw['spikes']
                and result['scratch_removed'] is True,'correlation observation differs')
        else:require(result['helper_histogram_range_ms']==[500.5,100500.5] and result['raw_events']==raw['spikes']
                and result['frozen_histogram_exact'] is True and result['endpoint_count_identity_exact'] is True,'series observation differs')
    else:
        inputs=result['source_sha256']
        require(result['observation_seconds']==100. and len(result['area_names'])==len(set(result['area_names']))==32
            and inputs['series_report']==sha(output/'series/time-series.json')
            and inputs['series_arrays']==sha(output/'series/time-series.npz')
            and inputs['matrices_metadata']==sha(source/'interarea/matrices/matrices.json'),'interarea source/observation differs')
        if name=='fc':require(result['schema']=='b2-mam-paper-fc-v1' and result['validated_rate_input'] is True
            and result['normalization_bins_exact'] is True and result['equal_observation_duration'] is True
            and inputs['reference_report']==sha(source/'interarea/fc-reference/report.json'),'FC reference validation differs')
        else:require(result['schema']=='b2-mam-paper-propagation-v1' and result['excluded_from_hierarchy']==['MDP']
            and len(result['hierarchy_area_names'])==31 and inputs['fc_report']==sha(output/'fc/fc.json')
            and inputs['reference_audit']==sha(source/'interarea/lag-reference/report.json'),'lag reference/ordering differs')


def run(source,completion_sha,previous_seconds):
    begin=time.monotonic();require(0<=previous_seconds<600,'bounded preparation charge required')
    started=begin-previous_seconds;source=source.resolve()
    require(source.is_relative_to(BASE.resolve()),'analysis source must be on brick2')
    manifest=read(source/'catalog.json')
    for name,row in manifest.items():
        p=source/name
        require(p.resolve().is_relative_to(source) and not p.is_symlink() and p.stat().st_size==row['bytes']
                and sha(p)==row['sha256'],'analysis source changed')
    identity,raw,terminal=accepted_raw(source/'case',completion_sha)
    output=Path(identity['analysis'])
    require(output.is_relative_to(BASE) and output.parent.resolve()==output.parent
            and output.parent.is_dir() and not output.exists(),'new resolved confirmation analysis output required')
    output.mkdir()
    write(output/'intent.json',dict(schema='b2-mam-confirmation-analysis-attempt-v1',replicate=REPLICATE,
        identity_sha256=IDENTITY_SHA,completion_sha256=completion_sha,maximum_analysis_attempts=1,
        previous_seconds=previous_seconds,total_seconds=TOTAL_SECONDS,automatic_retry=False,estimator_changes=False))
    try:
        rows=execute_sequence(stages(source,identity),output,started=started,
            validate=lambda name:validate(name,source,identity,raw))
        result=dict(schema='b2-mam-confirmation-six-metrics-pending-v1',replicate=REPLICATE,identity_sha256=IDENTITY_SHA,
            model_sha256=identity['model_sha256'],runtime_random_key=identity['random_keys']['runtime_input'],
            completion_sha256=completion_sha,raw_report_sha256=sha(source/'case/raw/report.json'),
            full_descriptive_analysis_complete=True,analysis_guard_passed=False,pending_guard_acceptance=True,
            scientific_acceptance=False,performance_cost_acceptance=False,formal_equivalence_acceptance=False,
            source_catalog_sha256=sha(source/'catalog.json'),stages=rows,
            catalogs={n:sha(output/n/'catalog.json') for n in REQUIRED},
            previous_seconds=previous_seconds,science_seconds=time.monotonic()-begin,
            total_accounted_seconds=time.monotonic()-started,shared_budget_seconds=TOTAL_SECONDS,
            scope='One new complete 100 s observation with fixed estimators. Numerical comparability and scientific acceptance require the stated remaining checks; no marginal tolerances or additional seeds are inferred.')
        require(result['total_accounted_seconds']<TOTAL_SECONDS,'shared science deadline exceeded')
        write(output/'pending.json',result);return result
    except BaseException as error:
        write(output/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False,
            elapsed_seconds=time.monotonic()-started));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--completion-sha256',required=True)
    p.add_argument('--previous-seconds',type=float,required=True)
    a=p.parse_args();result=run(a.source,a.completion_sha256,a.previous_seconds)
    print(json.dumps({k:result[k] for k in ['replicate','full_descriptive_analysis_complete','analysis_guard_passed','total_accounted_seconds']}))
