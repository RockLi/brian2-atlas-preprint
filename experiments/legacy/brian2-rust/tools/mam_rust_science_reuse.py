"""Reuse six descriptive metrics only after full raw identity and provenance.

No simulator data is opened before accepted terminal/full-raw controls exist.
Different raw outputs require a fresh six-metric analysis, not relaxed matching.
"""
import argparse,json,shlex,shutil,subprocess,tempfile,time
from pathlib import Path
from mam_nest_science_reuse import need,sha,read,add_pin,remote_code,MAX_REHASH_BYTES,BASE,NODE
from mam_launch_rust_recovery import CASE,PROTOCOL_SHA
from mam_launch_rust_performance import MODEL,PLAN,EXE_SHA,OLD,BRICK
from mam_benchmark_terminal import guard

OLD_PIPELINE='49617ef8ac6f8b5dde86869594f1ac8f9c879c69e9ca132bf1181c5f4138f0f5'
OLD_RAW='9f7a7f7e2b8de43d985272d3f1a4567a5dff54920103bf6e97213446b46e8db9'
OLD_GUARD='f83d07c04e3ffff9179e8390da0a63fc59b48b6c83fae101ced7623657d7652a'
INTERAREA='87529af8df49ab9c65ea2710c2285a8f777197184052b52ad34224cbf3d15546'
SOURCE=BASE/'primary-postrun-v1-source'

def validate_raw_source_catalog(evidence,catalog):
    from mam_launch_rust_benchmark_raw import RECOVERY_VALIDATION_SHA,RECOVERY_VALIDATION_PATH
    path=evidence/RECOVERY_VALIDATION_PATH
    need(sha(path)==RECOVERY_VALIDATION_SHA,'approved raw verifier validation changed')
    proof=read(path)
    expected={name:item['sha256'] for name,item in proof['files'].items() if name.startswith('tools/')}
    expected.update(proof['unchanged_reader_and_control_dependencies'])
    for name,digest in expected.items():
        need(name in catalog and catalog[name]['sha256']==digest,'unapproved raw verifier or reader source')

def match_raw(new,previous):
    need(new['raw_output_audit_passed'] is True and new['audit_guard_passed'] is True
         and new['complete_binary_scan_passed'] is True,'new independent full raw acceptance required')
    need(previous['internal_output_audit_passed'] is True and previous['full_independent_readers_passed'] is True,
         'old independent full raw acceptance required')
    for key,value in [('model_sha256',MODEL),('plan_sha256',PLAN),('executable_sha256',EXE_SHA)]:
        need(new[key]==previous[key]==value,'raw model, plan or executable differs')
    need(set(new['dump_sha256'])==set(previous['dump_sha256'])=={'results.bin','events.bin'},'both full files required')
    sizes={'results.bin':previous['dump_bytes']['results_bytes'],'events.bin':previous['dump_bytes']['events_bytes']}
    return (new['dump_sha256']==previous['dump_sha256'] and new['dump_bytes']==sizes
            and new['spikes']==previous['spikes'] and new['delivered_edges']==previous['delivered_edges'])

def accepted_new_raw(evidence,completion_sha256):
    from mam_rust_benchmark_raw import terminal_gate
    from mam_launch_rust_benchmark_raw import publish
    case=evidence/'performance-runs-v1'/CASE
    if not (case/'completion.json').is_file():return None
    need(completion_sha256 is not None and sha(case/'completion.json')==completion_sha256,'exact new completion pin required')
    complete=read(case/'completion.json')
    need(complete['case_id']==CASE and complete['protocol_sha256']==PROTOCOL_SHA
         and complete['terminal_resource_audit_passed'] is True and complete['raw_output_audit_passed'] is True,
         'full recovery case not accepted')
    need(set(complete['input_sha256'])=={'admission.json','launch.json','terminal/report.json','raw/report.json','raw/intent.json'},
         'completion input coverage')
    for name,digest in complete['input_sha256'].items():need(sha(case/name)==digest,'completion input changed')
    raw=read(case/'raw/report.json');binding=read(case/'raw/binding.json');intent=read(case/'raw/intent.json')
    need(raw['case_id']==CASE and raw['protocol_sha256']==PROTOCOL_SHA,'new raw identity')
    need(set(raw['evidence_sha256'])=={'binding.json','intent.json','admission.json','controller.json','pending.json','guard.json','attempt.json'},
         'full raw control coverage')
    for name,digest in raw['evidence_sha256'].items():need(sha(case/'raw'/name)==digest,'raw control changed')
    with tempfile.TemporaryDirectory(prefix='mam-rust-reuse-controls-',dir='/private/tmp') as temporary:
        target=Path(temporary);total=0
        for name in binding['files']:
            need(not Path(name).is_absolute() and '..' not in Path(name).parts,'bounded terminal path')
            source=evidence/'primary-run/package.json' if name=='package.json' else case/name
            total+=source.stat().st_size;need(total<=48*2**20,'bounded terminal controls')
            dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        _,_,_,terminal=terminal_gate(target,case/'raw/binding.json',intent['binding_sha256'])
    admission=read(case/'raw/admission.json')
    validate_raw_source_catalog(evidence,admission['source_catalog'])
    reproduced=publish(read(case/'raw/pending.json'),read(case/'raw/guard.json'),read(case/'raw/controller.json'),
        binding_sha=intent['binding_sha256'],terminal=terminal,
        raw_source_sha=admission['source_catalog']['tools/mam_rust_benchmark_raw.py']['sha256'],
        elapsed=raw['elapsed_through_guard_collection_seconds'])
    need(all(raw[k]==v for k,v in reproduced.items()),'new raw decision not reproducible')
    return raw

def old_artifacts(evidence,artifacts):
    from mam_primary_analysis_pipeline import resource_gate
    local=evidence/'primary-postrun';old=artifacts/'primary-postrun-v1';inter=artifacts/'mam-primary-interarea-v1'
    for path,digest in [(local/'report.json',OLD_PIPELINE),(local/'full-report.json',OLD_RAW),
                        (local/'full-guard.json',OLD_GUARD),(inter/'complete.json',INTERAREA)]:
        need(sha(path)==digest,'old scientific provenance pin changed')
    pipeline=read(local/'report.json');raw=read(local/'full-report.json')
    need(pipeline['analysis_complete'] is True and pipeline['shared_budget_seconds']==10800
         and 0<pipeline['elapsed_seconds']<=10800 and pipeline['output_audit_sha256']==OLD_RAW,'old pipeline incomplete')
    resources=resource_gate(artifacts/'primary-terminal-resources-v1')
    need(resources==pipeline['resource_input_sha256']==read(local/'admission.json')['resource_inputs'],
         'old terminal-to-science binding')
    pg=read(local/'full-guard.json');pc=read(old/'controller.json')
    need(pc['returncode']==0 and pc['error'] is None and pg['command']==pc['command'][pc['command'].index('--')+1:],
         'old analysis controller/guard')
    def guarded(g,prefix,memory,file_mib,seconds):
        return guard(g,dict(host=NODE,role='v1',memory_bytes=memory*2**30,pids_max=64,cpu_ids=[8,9],
            cpu_quota_cores=2,volume='/data/brick2',allow_root_volume=False,file_limit_bytes=file_mib*2**20,
            minimum_free_bytes=1280*2**30,reserved_host_memory_bytes=64*2**30),prefix,seconds)
    guarded(pg,'b2mpi-analysis-primary-postrun',16,512,10800)
    need(pipeline['elapsed_seconds']<=pg['wall_seconds']+.1,'old science outside guard')
    ig=read(inter/'guard.json');ic=read(inter/'controller.json');icomp=read(inter/'complete.json')
    need(icomp['completed'] is True and sha(inter/'guard.json')==icomp['guard_sha256']
         and ic['returncode']==0 and ic['error'] is None and ig['command']==ic['command'][ic['command'].index('--')+1:],
         'old interarea terminal binding')
    guarded(ig,'b2mpi-primary-interarea',4,64,360)
    manifest={};reports={};catalogs={};output_pins={}
    phases=[('activity',old/'activity','activity.json',BASE/'primary-postrun-v1/activity'),
      ('cell',old/'cell','paper-cell-metrics.json',BASE/'primary-postrun-v1/cell'),
      ('correlation',old/'correlation','correlation.json',BASE/'primary-postrun-v1/correlation'),
      ('series',old/'series','time-series.json',BASE/'primary-postrun-v1/series'),
      ('fc',inter/'rust-fc','fc.json',BASE/'mam-primary-interarea-v1/rust-fc'),
      ('lags',inter/'rust-lags','lags.json',BASE/'mam-primary-interarea-v1/rust-lags')]
    for phase,folder,name,remote in phases:
        catalog=read(folder/'catalog.json');catalogs[phase]=sha(folder/'catalog.json')
        if phase in pipeline['catalogs']:need(catalogs[phase]==pipeline['catalogs'][phase],'old metric catalog binding')
        add_pin(manifest,remote/'catalog.json',catalogs[phase],(folder/'catalog.json').stat().st_size)
        for filename,item in catalog.items():
            need(Path(filename).name==filename,'metric catalog basename')
            add_pin(manifest,remote/filename,item['sha256'],item['bytes']);output_pins[str(remote/filename)]=item['sha256']
            if (folder/filename).exists():need(sha(folder/filename)==item['sha256'],'local metric copy changed')
        need(name in catalog,'metric report absent');r=read(folder/name);reports[phase]=r
        need(r['scientific_equivalence'] is False,'descriptive metric scope')
        if phase!='activity':
            need(r['identity']['simulator']=='Rust' and r['identity']['model_sha256']==MODEL,'metric model identity')
            if 'seed' in r['identity']:need(r['identity']['seed']==1729,'metric seed identity')
        for name,digest in r['implementation_sha256'].items():
            need(Path(name).name==name,'implementation basename')
            parent=(BASE/'native-primary-postrun-v1-source/tools') if phase in ['fc','lags'] else \
                   SOURCE/('python/brian2_rust' if name in ['results.py','multi_area_analysis.py'] else 'tools')
            add_pin(manifest,parent/name,digest)
    activity=reports['activity'];need(activity['model_sha256']==MODEL and activity['result_sha256']==raw['dump_sha256'],
                                     'old activity raw identity')
    need(activity['population_count']==254 and activity['neurons']==4129924,'full activity geometry')
    for phase in ['cell','correlation','series']:
        sources=reports[phase]['source_sha256']
        for filename in ['results.bin','events.bin']:
            need(sources[str(Path(BRICK)/'runs'/OLD/filename)]==raw['dump_sha256'][filename],'complete scientific raw identity')
        for name,digest in sources.items():
            if name.endswith('/results.bin') or name.endswith('/events.bin'):continue
            if name==str(BASE/OLD/'model.json'):
                need(digest==MODEL,'model source differs');continue  # Rehashed by the new full raw audit.
            if name in output_pins:need(digest==output_pins[name],'metric dependency differs')
            add_pin(manifest,name,digest)
    need(activity['window']['start_tick']==5000 and activity['window']['end_tick']==1005000
         and activity['window']['spike_tick_offset']==1 and activity['window']['endpoint']=='[start,end)',
         'Rust physical rate convention changed')
    cell=reports['cell'];need(cell['window']==dict(start_tick=5000,end_tick=1005000,dt_ms=.1,
        rate_endpoint='(start,end)',lvr_endpoint='[start,end)',refractory_ms=2)
        and cell['lvr_aggregation']=='All recorded neurons, including zero for <3 spikes. Eligible mean is supplemental.',
        'cell endpoint or denominator changed')
    corr=reports['correlation'];need(corr['observation_ms']==[500,100500] and corr['bin_ms']==1,'correlation window')
    series=reports['series'];need(series['helper_histogram_range_ms']==[500.5,100500.5] and series['bin_ms']==1
         and series['total_population_bin_entries']==25400000 and series['welch_segments']==4125,'full spectrum window')
    for phase in ['fc','lags']:
        r=reports[phase];need(r['source_sha256']['series_report']==sha(old/'series/time-series.json')
            and r['source_sha256']['series_arrays']==output_pins[str(BASE/'primary-postrun-v1/series/time-series.npz')]
            and r['observation_seconds']==100,'interarea full series binding')
    need(reports['lags']['source_sha256']['fc_report']==sha(inter/'rust-fc/fc.json'),'lag FC binding')
    for name,item in read(inter/'pins.json').items():
        if '/native-primary-postrun-v1/' in name:continue
        add_pin(manifest,name,item['sha256'],item['bytes'])
    ref=BASE/'mam-primary-interarea-v1-source'
    bindings=[('fc','matrices',ref/'matrices/matrices.npz'),('fc','matrices_metadata',ref/'matrices/matrices.json'),
      ('fc','reference_report',ref/'fc-reference/report.json'),('fc','reference_arrays',ref/'fc-reference/reconstructed.npz'),
      ('lags','reference_audit',ref/'lag-reference/report.json'),('lags','matrices_metadata',ref/'matrices/matrices.json')]
    for phase,key,path in bindings:need(manifest[str(path)]['sha256']==reports[phase]['source_sha256'][key],'reference binding changed')
    add_pin(manifest,SOURCE/'catalog.json',pipeline['source_catalog_sha256'])
    for name,digest in resources.items():add_pin(manifest,SOURCE/'resources'/name,digest)
    add_pin(manifest,BASE/'primary-postrun-v1/report.json',OLD_PIPELINE)
    add_pin(manifest,BASE/'guards/primary-postrun-v1.json',OLD_GUARD)
    add_pin(manifest,BASE/'guards/mam-primary-interarea-v1.json',icomp['guard_sha256'])
    need(len(manifest)<=128 and sum(x['bytes'] or 0 for x in manifest.values())<MAX_REHASH_BYTES,'bounded artifact manifest')
    return dict(manifest=manifest,metric_catalogs=catalogs,old_raw=raw,mean_rate_hz=activity['mean_rate_hz'],
        input_sha256=dict(old_pipeline=OLD_PIPELINE,old_raw=OLD_RAW,old_guard=OLD_GUARD,old_interarea=INTERAREA))

def prepare(evidence,artifacts,completion_sha256=None):
    new=accepted_new_raw(evidence,completion_sha256)
    if new is None:return dict(ready=False,reason='successful full target and raw audit are pending')
    plan=old_artifacts(evidence,artifacts)
    if not match_raw(new,plan['old_raw']):
        return dict(ready=False,fresh_analysis_required=True,reason='full raw identity differs; recompute all six metrics')
    need(new['sources'][str(BASE/OLD/'model.json')]==MODEL,'new raw model rehash required')
    plan.pop('old_raw');plan['input_sha256'].update(new_completion=completion_sha256,new_raw=sha(evidence/'performance-runs-v1'/CASE/'raw/report.json'))
    return dict(ready=True,**plan)

def run(evidence,t7,completion_sha256=None):
    start=time.monotonic();plan=prepare(evidence,t7/'artifacts',completion_sha256)
    if not plan['ready']:return dict(**plan,descriptive_metrics_reuse_accepted=False,remote_rehash_started=False)
    out=evidence/'performance-runs-v1'/CASE/'science-reuse-v2';out.mkdir(exist_ok=False)
    def write(name,row):
        with (out/name).open('x') as f:json.dump(row,f,indent=2);f.write('\n')
    write('intent.json',dict(**plan,maximum_rehash_bytes=MAX_REHASH_BYTES,maximum_stage_seconds=180,
                            maximum_remote_checks=1,automatic_retry=False,implementation_sha256=sha(Path(__file__))))
    try:
        command=['tsh','ssh','rock@'+NODE,shlex.join(['taskset','-c','8,9','python3','-c',remote_code(plan['manifest'])])]
        with (out/'remote.json').open('xb') as stdout,(out/'remote.stderr').open('xb') as stderr:
            result=subprocess.run(command,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,timeout=60)
        need(result.returncode==0,'remote provenance rehash failed; no retry')
        remote=read(out/'remote.json');need(set(remote['files'])==set(plan['manifest']),'full artifact coverage')
        for name,item in plan['manifest'].items():
            r=remote['files'][name];need(r['sha256']==item['sha256'] and (item['bytes'] is None or r['bytes']==item['bytes']),
                                      'remote provenance identity')
        need(remote['host']==NODE and remote['total_bytes']<=MAX_REHASH_BYTES and remote['wall_seconds']<=55
             and remote['cpu_seconds']<=30 and remote['peak_rss_kib']*1024<=256*2**20,'provenance rehash budget')
        need(time.monotonic()-start<180,'shared reuse stage deadline')
        report=dict(schema='b2-mam-rust-science-reuse-v2',case_id=CASE,descriptive_metrics_reuse_accepted=True,
            six_metrics=['firing_rates','LvR','pairwise_correlation','PSD','FC','interarea_lags'],
            full_binary_identity=True,metric_catalogs=plan['metric_catalogs'],mean_rate_hz=plan['mean_rate_hz'],
            input_sha256=plan['input_sha256'],intent_sha256=sha(out/'intent.json'),remote_sha256=sha(out/'remote.json'),
            additional_independent_seed_samples=0,scientific_equivalence=False,paper_reproduction_accepted=False,
            performance_cost_acceptance=False,independent_full_raw_backup=False,elapsed_seconds=time.monotonic()-start,
            scope='Descriptive metrics reused from identical complete raw inputs and verified analysis provenance; no scientific equivalence or new seed claim.')
        write('report.json',report)
        dest=t7/'artifacts/performance-runs-v1'/CASE/'science-reuse-v2';dest.mkdir(parents=True,exist_ok=False)
        for p in out.iterdir():shutil.copyfile(p,dest/p.name);need(sha(p)==sha(dest/p.name),'T7 provenance archive')
        return report
    except BaseException as error:
        write('failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--t7',type=Path,required=True)
    p.add_argument('--completion-sha256')
    a=p.parse_args();result=run(a.evidence,a.t7,a.completion_sha256);print(json.dumps(result));raise SystemExit(0 if result.get('descriptive_metrics_reuse_accepted') else 2)
