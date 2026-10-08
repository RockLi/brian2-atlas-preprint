"""Sequential, resumable real-network correctness gates and frozen 5x5 CPU study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiment import CONDITIONS,SEEDS


def compare(left,right,exact=False):
    errors={}
    with np.load(left/'snapshot.npz') as a,np.load(right/'snapshot.npz') as b:
        if set(a.files)!=set(b.files):raise AssertionError('snapshot fields differ')
        for key in a.files:
            assert a[key].shape==b[key].shape and a[key].dtype==b[key].dtype,key
            if exact or key in ('spike_i','spike_t','counts','not_refractory'):
                np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            else:np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-14,err_msg=key)
            errors[key]=float(np.max(np.abs(a[key].astype(float)-b[key].astype(float)),initial=0))
    return errors


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('experiment.py')
    identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [source,source.with_name('spiking.py'),source.with_name('PROTOCOL.md'),args.prepared/'manifest.json']}
    identity['amplitude_mv']=args.amplitude
    manifest=out/'study_identity.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=identity:raise ValueError('study identity changed; use a new output directory')
    else:manifest.write_text(json.dumps(identity,indent=2)+'\n')
    registry=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
    def save(): (out/'progress.json').write_text(json.dumps(registry,indent=2)+'\n')
    def job(name,options,partial=False):
        folder=out/name
        if registry.get(name,{}).get('status')=='complete':
            expected='partial.json' if partial else 'report.json'
            if not (folder/expected).exists():raise ValueError(f'missing completed artifact {name}')
            return folder
        if folder.exists():
            previous=registry.get(name,{})
            if not (args.retry_failed and previous.get('status')=='failed' and
                    previous.get('exit_code') not in (None,0)):
                raise RuntimeError(f'incomplete directory {folder}; inspect prior process before explicitly choosing a retry path')
            # The supervisor has wait()ed this exact attempt to a terminal
            # exit. Archive its entire evidence before an explicit retry;
            # a merely running/stale PID entry is never sufficient.
            archive=out/'failed-attempts'/f'{name}-{time.time_ns()}'
            archive.parent.mkdir(exist_ok=True)
            folder.rename(archive)
            old_log=out/f'{name}.log'
            if old_log.exists():old_log.rename(archive/'process.log')
            (archive/'attempt.json').write_text(json.dumps(previous,indent=2)+'\n')
            print(f'ARCHIVED FAILED {name} {archive}',flush=True)
        command=[sys.executable,str(source),'--prepared',str(args.prepared.resolve()),'--output',str(folder),
                 '--amplitude',str(args.amplitude),*options]
        print(f'START {name}',flush=True)
        started=time.time()
        with (out/f'{name}.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            registry[name]={'status':'running','pid':process.pid,'command':command,'started':started};save()
            code=process.wait()
        registry[name].update(status='complete' if code==0 else 'failed',exit_code=code,wall_seconds=time.time()-started)
        save()
        if code:raise RuntimeError(f'{name} failed; see {out/name}.log')
        print(f'DONE {name} {time.time()-started:.1f}s',flush=True)
        return folder
    validations={}
    # Each full-brain check runs the actual 139,255-node mixed model.
    common=['--scope','full','--protocol','validation']
    full4=job('full-learning-aot4',[*common,'--threads','4'])
    full1=job('full-learning-aot1',[*common,'--threads','1'])
    fullcpp=job('full-learning-cpp',[*common,'--backend','cpp','--threads','1'])
    validations['full_aot_1_4']=compare(full4,full1,exact=True)
    validations['full_cpp']=compare(full4,fullcpp)
    frozen=job('full-frozen-split',[*common,'--condition','frozen','--threads','4'])
    unsplit=job('full-frozen-unsplit',[*common,'--condition','frozen','--threads','4','--unsplit'])
    validations['full_partition_frozen']=compare(frozen,unsplit)
    (out/'correctness.json').write_text(json.dumps(validations,indent=2)+'\n')
    if args.validation_only:return
    # Thread count is an execution parameter; scientific protocol is unchanged.
    # Choose faster observed native loop from the mandatory 1/4 validation pair.
    timings={str(t):json.loads((p/'report.json').read_text())['stages'][0]['timings']['simulation_and_recording_seconds'] for t,p in [(1,full1),(4,full4)]}
    threads=int(min(timings,key=timings.get))
    (out/'execution_choice.json').write_text(json.dumps({'threads':threads,'single_observation_native_seconds':timings,
        'scope':'execution choice only; not a statistically qualified performance benchmark'},indent=2)+'\n')
    for seed in SEEDS:
        for condition in CONDITIONS:
            job(f'seed-{seed}-{condition}',['--scope','full','--protocol','formal','--seed',str(seed),
                                          '--condition',condition,'--threads',str(threads)])
    base=out/'seed-11-paired'
    continuous=job('full-continuous-seed11',['--scope','full','--protocol','formal','--seed','11',
                                          '--threads',str(threads),'--single-run'])
    validations['full_continuous_segmented']=compare(base,continuous,exact=True)
    saved=job('full-save-seed11',['--scope','full','--protocol','formal','--seed','11',
                               '--threads',str(threads),'--save-after-acquisition'],partial=True)
    resumed=job('full-restore-seed11',['--scope','full','--protocol','formal','--seed','11',
                                    '--threads',str(threads),'--resume',str(saved/'acquisition.checkpoint')])
    validations['full_fresh_process_restore']=compare(base,resumed,exact=True)
    # Repeat the last native segment against its saved state and pending events.
    stages=json.loads((base/'report.json').read_text())['stages']
    artifact=Path(stages[-1]['artifact_directory'])
    replay=out/'native-replay'
    if not replay.exists():
        subprocess.run([sys.executable,str(source.with_name('native_measure.py')),str(out/'native-replay-metrics.json'),
                        str(artifact/'native/b2-native'),str(artifact/'native/instance.bin'),str(replay)],
                       env={**os.environ,'B2_NUM_THREADS':str(threads)},check=True,capture_output=True)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(replay/'results.bin')==digest(artifact/'rust/results.bin')
    validations['native_replay_exact']=True
    rows=[]
    for seed in SEEDS:
        baseline=json.loads((out/f'seed-{seed}-frozen/report.json').read_text())
        common_input=None
        for condition in CONDITIONS:
            folder=out/f'seed-{seed}-{condition}';report=json.loads((folder/'report.json').read_text())
            configuration=json.loads((folder/'configuration.json').read_text())
            with np.load(folder/'input.npz') as inputs:
                signature={k:digest_bytes(inputs[k]) for k in inputs.files if k not in ('learning','teaching')}
                if common_input is None:common_input=signature
                assert signature==common_input,'condition input mismatch'
                for trial in configuration['trials']:
                    if trial['training']:continue
                    start=int(round(trial['start_ms']/.1));end=start+5000
                    assert not np.any(inputs['learning'][start:end]) and not np.any(inputs['teaching'][start:end])
            for block in ('pre','post','final'):
                assert next(s for s in report['stages'] if s['name']==block)['weights_frozen_exact']
            if condition=='teaching_off':compare(folder,out/f'seed-{seed}-frozen',exact=True)
            rows.append({'seed':seed,'condition':condition,
                         'post_effect_hz':report['contrasts']['post']['B_minus_A_hz']-baseline['contrasts']['post']['B_minus_A_hz'],
                         'final_effect_hz':report['contrasts']['final']['B_minus_A_hz']-baseline['contrasts']['final']['B_minus_A_hz'],
                         'contrasts':report['contrasts'],'changed_edges':report['changed_edges'],
                         'final_gain_mean':report['final_gain_mean'],'spikes':report['spikes'],
                         'native_seconds':sum(s['timings']['simulation_and_recording_seconds'] for s in report['stages']),
                         'native_peak_rss_bytes':max(s['native']['native_peak_rss_bytes'] for s in report['stages']),
                         'frontend_peak_rss_bytes':report['frontend_peak_rss_bytes'],
                         'end_to_end_seconds':report['end_to_end_seconds']})
    summary={}
    for condition in CONDITIONS:
        group=[r for r in rows if r['condition']==condition]
        summary[condition]={metric:{'mean':float(np.mean([r[metric] for r in group])),
                                   'min':float(min(r[metric] for r in group)),'max':float(max(r[metric] for r in group))}
                            for metric in ('post_effect_hz','final_effect_hz','final_gain_mean','native_seconds','native_peak_rss_bytes')}
    (out/'correctness.json').write_text(json.dumps(validations,indent=2)+'\n')
    (out/'final_report.json').write_text(json.dumps({'rows':rows,'summary':summary,'seeds':SEEDS,
                                                  'conditions':CONDITIONS,'completed_cases':len(rows),'threads':threads,
                                                  'all_correctness_gates_passed':True},indent=2)+'\n')
    print('STUDY COMPLETE',flush=True)


def digest_bytes(array):return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--amplitude',type=float,required=True);p.add_argument('--validation-only',action='store_true')
    p.add_argument('--retry-failed',action='store_true',help='archive and retry only supervisor-confirmed terminal failures')
    main(p.parse_args())
