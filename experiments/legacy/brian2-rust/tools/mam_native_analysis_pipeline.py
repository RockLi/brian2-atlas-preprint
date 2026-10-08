"""Four native scientific summaries after guarded raw-audit publication.

Uses only the remaining portion of the original 10800-second analysis budget.
All outputs are descriptive; no scientific equivalence or cost advantage is
conferred by successful execution.
"""
import argparse
import json
from pathlib import Path
import sys
import time

from mam_primary_analysis_pipeline import execute_sequence, check, read, sha, TOTAL_SECONDS
from mam_native_primary_resources import NEURONS, PARAMETERS, NODES
from mam_collect_native_primary_raw import BUILD, DESTINATION
from mam_native_primary_output_audit import LABEL, PRIOR_SPIKES, audit_spec

BASE=Path(BUILD)
OUTPUT=BASE/'native-primary-postrun-v1'
RAW=Path(DESTINATION)
MAX_RAW=96*2**30
IDENTITY=dict(seed=1729,ranks=48,threads=4,duration_ms=100500,dt_ms=.1,nest_version='3.10.0')


def analysis_spec(reference_seed):
    spec=audit_spec(reference_seed)
    return dict(**spec,raw=RAW if reference_seed is None else Path(spec['destination']),
                output=OUTPUT if reference_seed is None else BASE/f'native-reference{reference_seed}-postrun-v1',
                identity=IDENTITY if reference_seed is None else {**IDENTITY,'seed':reference_seed})


def stages(source,normalization,*,reference_seed=None):
    spec=analysis_spec(reference_seed);raw=spec['raw'];output=spec['output']
    common=['--native-audit',str(raw),'--bounded-memory']
    baseline=['--baseline',str(output/'activity')]
    plan=[('activity',2400,'analyze_native_mam_activity.py',
        ['--audit-dir',str(raw),'--end-tick','1005000','--max-bytes',str(MAX_RAW),'--bounded-memory']),
        ('cell',1800,'analyze_mam_paper_cell_metrics.py',common+baseline),
        ('correlation',2400,'analyze_mam_paper_correlation.py',common+baseline),
        ('series',1800,'analyze_mam_paper_time_series.py',common+baseline+['--normalization',str(normalization)])]
    if reference_seed is not None:
        shared=['--series',str(output/'series'),'--matrices',str(source/'interarea/matrices')]
        plan.extend([
            ('fc',180,'analyze_mam_reference_fc.py',shared+[
                '--reference-audit',str(source/'interarea/fc-reference'),'--reference-seed',str(reference_seed)]),
            ('lags',180,'analyze_mam_paper_lags.py',shared+[
                '--reference-audit',str(source/'interarea/lag-reference'),'--fc-audit',str(output/'fc')]),
        ])
    return [(name,cap,[sys.executable,str(source/'tools'/tool),*args,'--output',str(output/name)])
            for name,cap,tool,args in plan]


def raw_gate(budget,*,reference_seed=None):
    spec=analysis_spec(reference_seed);raw=spec['raw']
    report=read(raw/'summary.json')
    check(sha(raw/'summary.json')==budget['summary_sha256'] and report['schema']=='b2-mam-native-primary-output-audit-v1'
        and report['passed'] is True and report['audit_guard_passed'] is True
        and report['raw_output_audit_passed'] is True and report['terminal_resource_audit_passed'] is True
        and report['label']==spec['label'] and report['parameters_sha256']==PARAMETERS
        and report['retained_prefix_spikes']==spec['prior_spikes'] and report[f"all_first_{spec['prior_duration_ms']}ms_event_prefixes_exact"] is True
        and report['duration_ms']==100500 and report['event_bytes']<=MAX_RAW,
        'native primary published raw prerequisite differs')
    if reference_seed is not None:
        from mam_launch_native_full_reference import PROTOCOL_SHA
        check(report['seed']==reference_seed and budget['seed']==reference_seed
              and budget['label']==spec['label'] and budget['protocol_sha256']==PROTOCOL_SHA,
              'reference raw/budget identity differs')
    check(report['audit_guard_sha256']==budget['raw_guard_sha256']
        and report['shared_analysis_budget_seconds']==budget['shared_analysis_budget_seconds']==TOTAL_SECONDS
        and report['analysis_wall_seconds_consumed']<=budget['previous_analysis_seconds']<TOTAL_SECONDS,
        'native remaining analysis budget differs')
    return report


def validate_output(name,*,reference_seed=None,source=None):
    spec=analysis_spec(reference_seed);output=spec['output'];raw_root=spec['raw']
    directory=output/name;catalog=read(directory/'catalog.json')
    required={'activity':{'activity.json','activity-arrays.npz','activity-overview.png','spike-rasters.png'},
        'cell':{'paper-cell-metrics.json','cell-metrics.npz'},
        'correlation':{'correlation.json','selection.npz'},
        'series':{'time-series.json','time-series.npz','spectrum.png'},
        'fc':{'fc.json','fc.npz'},'lags':{'lags.json','lags.npz'}}[name]
    check(required<=set(catalog),'native analysis output catalog incomplete')
    for filename,row in catalog.items():
        check(Path(filename).name==filename and not filename.startswith('._'),'unsafe analysis catalog entry')
        p=directory/filename
        check(p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes']<=512*2**20
              and sha(p)==row['sha256'],'native analysis output differs')
    baseline=read(output/'activity/activity.json')
    raw=read(raw_root/'summary.json')
    check(baseline['schema']=='b2-native-mam-activity-v1' and baseline['simulation']==spec['identity']
          and baseline['parameters_sha256']==PARAMETERS and baseline['neurons']==NEURONS
          and baseline['population_count']==254 and baseline['bounded_memory'] is True,
          'native activity identity differs')
    w=baseline['window']
    check((w['start_tick'],w['end_tick'],w['seconds'],w['spike_tick_offset'],w['endpoint'])
          ==(5000,1005000,100.,0,'[start,end)') and w['raster_end_tick']==105000,
          'native physical observation/raster window differs')
    check(baseline['observed_spikes']==sum(raw['physical_50ms_bin_counts'][10:]),'native activity physical count differs')
    expected={}
    for rank in range(48):
        root=raw_root/('node'+NODES[rank//8].rsplit('-',1)[-1])/'runs'/spec['label']
        r=read(root/('rank'+str(rank)+'.json'))
        expected[str(root/('rank'+str(rank)+'.events.bin'))]=dict(bytes=r['event_bytes'],sha256=r['event_sha256'])
    check(baseline['result_files']==expected,'native activity raw source identities differ')
    if name in ['fc','lags']:
        check(reference_seed is not None and source is not None,'inter-area reference admission required')
        r=read(directory/(name+'.json'));identity=r['identity'];inputs=r['source_sha256']
        check(identity['simulator']=='NEST' and all(identity[k]==v for k,v in spec['identity'].items())
              and r['scientific_equivalence'] is False and r['observation_seconds']==100.
              and len(r['area_names'])==len(set(r['area_names']))==32,'reference inter-area identity differs')
        check(inputs['series_report']==sha(output/'series/time-series.json')
              and inputs['series_arrays']==sha(output/'series/time-series.npz')
              and inputs['matrices_metadata']==sha(source/'interarea/matrices/matrices.json'),
              'reference inter-area rate/matrix inputs differ')
        if name=='fc':
            check(r['schema']=='b2-mam-paper-fc-v1' and r['validated_rate_input'] is True
                  and r['normalization_bins_exact'] is True and r['equal_observation_duration'] is True
                  and inputs['reference_report']==sha(source/'interarea/fc-reference/report.json'),
                  'reference FC validation differs')
        else:
            check(r['schema']=='b2-mam-paper-propagation-v1' and r['excluded_from_hierarchy']==['MDP']
                  and len(r['hierarchy_area_names'])==31
                  and inputs['fc_report']==sha(output/'fc/fc.json')
                  and inputs['reference_audit']==sha(source/'interarea/lag-reference/report.json'),
                  'reference lag validation differs')
        return
    if name!='activity':
        filename={'cell':'paper-cell-metrics.json','correlation':'correlation.json','series':'time-series.json'}[name]
        r=read(directory/filename);identity=r['identity']
        check(identity['simulator']=='NEST' and (name=='cell' or identity['condition']=='metastable')
              and all(identity[k]==v for k,v in spec['identity'].items()),'native paper summary identity differs')
        # Each native analysis verifies baseline hashes and the complete raw
        # source hashes itself; the catalog above preserves its exact report.
        check(r['scientific_equivalence'] is False and r['bounded_memory'] is True,'descriptive/bounded output differs')
        check(r['source_sha256'][str(output/'activity/activity.json')]==sha(output/'activity/activity.json'),
              'native paper summary baseline hash differs')
        if name=='cell':check(r['window']['end_tick']==1005000 and r['scratch_removed'] is True,'native cell window/scratch differs')
        elif name=='correlation':check(r['observation_ms']==[500,100500] and r['raw_events']==raw['spikes']
                                      and r['scratch_removed'] is True,'native correlation window/count differs')
        else:check(r['helper_histogram_range_ms']==[500.5,100500.5] and r['raw_events']==raw['spikes']
                   and r['included_terminal_events']==raw['terminal_tick_events'],'native time-series window/count differs')


def run(source,normalization,budget_path,*,reference_seed=None):
    began=time.monotonic();budget=read(budget_path)
    check(budget['schema']=='b2-mam-native-analysis-budget-v1' and budget['automatic_retry'] is False,
          'native analysis budget record missing')
    spec=analysis_spec(reference_seed);output=spec['output']
    raw=raw_gate(budget,reference_seed=reference_seed)
    started=began-budget['previous_analysis_seconds']
    source=source.resolve()
    check(source.is_relative_to(BASE.resolve()) and normalization.resolve().is_relative_to(source),
          'native analysis source/normalization must be staged on data volume')
    manifest=read(source/'catalog.json')
    for name,row in manifest.items():
        p=source/name
        check(p.resolve().is_relative_to(source) and not p.is_symlink() and p.stat().st_size==row['bytes']
              and sha(p)==row['sha256'],'staged native analysis input differs')
    check(not output.exists(),'native science analysis already attempted')
    output.mkdir()
    plan=stages(source,normalization,reference_seed=reference_seed)
    rows=execute_sequence(plan,output,started=started,
        validate=lambda name:validate_output(name,reference_seed=reference_seed,source=source))
    elapsed=time.monotonic()-began
    report=dict(schema='b2-mam-native-analysis-pipeline-v1',analysis_complete=True,scientific_acceptance=False,
        performance_cost_acceptance=False,shared_budget_seconds=TOTAL_SECONDS,
        previous_analysis_seconds=budget['previous_analysis_seconds'],science_seconds=elapsed,
        total_accounted_seconds=budget['previous_analysis_seconds']+elapsed,
        raw_summary_sha256=budget['summary_sha256'],raw_guard_sha256=budget['raw_guard_sha256'],
        source_catalog_sha256=sha(source/'catalog.json'),stages=rows,
        catalogs={name:sha(output/name/'catalog.json') for name,_,_ in plan},
        scope='One full native 100-second observation summarized. Inter-area comparisons, independent seeds, '
              'scientific equivalence and tuned speed/cost acceptance remain required.')
    if reference_seed is not None:
        from mam_launch_native_full_reference import PROTOCOL_SHA
        report.update(label=spec['label'],seed=reference_seed,protocol_sha256=PROTOCOL_SHA,
                      interarea_analysis_complete=True,required_stages=[name for name,_,_ in plan])
    check(report['total_accounted_seconds']<TOTAL_SECONDS,'shared native analysis deadline exceeded')
    with (output/'report.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','normalization','budget']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-seed',type=int,choices=[1730,1731])
    args=p.parse_args();print(json.dumps(run(args.source,args.normalization,args.budget,reference_seed=args.reference_seed)))
