"""Full-graph EI/sensory correctness, controls, and fair native performance."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile

import numpy as np

from flywire_device import DEFAULTS, read_snapshot
from flywire_ei_analysis import analyze
from flywire_benchmark import compare, replay
from performance_suite import inspect_rustc,validate_rustc,benchmark_environment
from threaded_benchmark import prepare_cxx,default_cxx,parse_levels

CONDITIONS=('rest','odor','cut_rest','cut')


def snapshot(folder):
    with np.load(folder/'snapshot.npz') as d:return {k:d[k] for k in d.files}


def markdown(report):
    bio=report['biology'];summary=report['performance']
    lines=['# FlyWire transmitter-informed conductance LIF benchmark','',
           'Reduced model with empirical topology, curated transmitter signs and DM1 sensory input; not experimentally validated whole-brain physiology.','',
           f"139,255 neurons; 15,091,983 weighted edges; 54,492,922 biological contacts. {report['repeats']} deterministic measured replays per configuration; 1,000 ms simulated time.",
           '',f"Resting-condition rate after the first 100 ms: **{bio['conditions']['rest']['mean_rate_hz']:.3f} Hz**.",
           '', '| Threads | Rust s | C++ s | C++ / Rust | Rust MiB | C++ MiB | Rust 1/N speedup |',
           '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for t,row in summary.items():
        a,c=row['aot'],row['cpp']
        lines.append(f"| {t} | {a['median_seconds']:.4f} | {c['median_seconds']:.4f} | {c['median_seconds']/a['median_seconds']:.2f}x | {a['peak_rss_bytes']/2**20:.1f} | {c['peak_rss_bytes']/2**20:.1f} | {row['rust_self_speedup']:.2f}x |")
    lines += ['', 'Times include simulation and recording; import/build/startup/output are excluded. RSS is measured by a fresh small supervisor, excluding Python and compilers. Raw timing spreads are preserved in JSON.','',
              '| Cell set | Rest Hz during stimulus window | Stimulated Hz | Paired increase Hz | Increase after ORN output cut Hz |',
              '| --- | ---: | ---: | ---: | ---: |']
    for group,row in bio['response_300_to_700ms'].items():
        lines.append(f"| {group} | {row['rest']:.3f} | {row['odor']:.3f} | {row['stimulus_delta_hz']:.3f} | {row['cut_stimulus_delta_hz']:.3f} |")
    lines += ['', 'Acceptance flags:', '', *[f'- {k}: **{v}**' for k,v in report['acceptance'].items()], '',
              'Full snapshots compare final voltage, both conductances, transmission masks, refractory state, all spike IDs/times/counts and recorded trajectories. Every backend is also replayed against its own initial output to test exact determinism.','',
              'Background: 512 independent frozen Poisson-like channels, randomly mapped to all neurons. This low-dimensional background and the 1–5 Hz calibration target are engineering assumptions. DM1 stimulation is a synthetic 80 Hz input during 300–700 ms, not a fitted chemical odor stimulus. Both cut and uncut networks have paired stimulus-off controls.','',
              'Synaptic conductance is positive in both E/I channels; inhibitory current uses a -70 mV reversal, excitatory current 0 mV. The negative signed contact count selects the inhibitory channel with a 4x conductance multiplier; this is not a universal experimental fly-brain ratio. Monoamine-only/unknown fast actions are zeroed, and ambiguous curated E/I evidence is reported.','',
              'Sources: [official v783 topology](https://zenodo.org/records/10676866), [versioned annotations](https://github.com/flyconnectome/flywire_annotations/tree/ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0), [Shiu et al. reduced whole-brain modelling](https://www.nature.com/articles/s41586-024-07763-9). The present conductance model and olfactory experiment differ from that paper.','']
    return '\n'.join(lines)


def measured_replay(*args, **kwargs):
    result = replay(*args, **kwargs)
    # C++ dumps immutable connection arrays too; retaining every replay can fill a disk.
    shutil.rmtree(args[2])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--levels',type=parse_levels,default=[1,4,8]);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--rustc',default='rustc');p.add_argument('--cxx',default=default_cxx());p.add_argument('--config',type=Path)
    p.add_argument('--reuse-builds',action='store_true',help='Validate and replay existing complete builds without recompiling')
    a=p.parse_args();a.graph=a.graph.resolve();a.output=a.output.resolve()
    if not 3<=a.repeats<=10:p.error('3..10 repeats required')
    toolchain=inspect_rustc(a.rustc);validate_rustc(toolchain);os.environ.update(benchmark_environment(toolchain))
    a.output.mkdir(parents=True,exist_ok=a.reuse_builds)
    cxx,cxx_version=prepare_cxx(a.output,a.cxx);env={**os.environ,'CXX':cxx,'OMP_DYNAMIC':'FALSE'}
    cfg={**DEFAULTS,**(json.loads(a.config.read_text()) if a.config else {})}
    config_path=a.output/'config.json'
    if a.reuse_builds:
        if json.loads(config_path.read_text()) != cfg:raise ValueError('reuse configuration differs from existing builds')
    else:config_path.write_text(json.dumps(cfg,indent=2)+'\n')
    builds={};snapshots={};cross_errors={}
    cases=[(backend,condition,1) for condition in CONDITIONS for backend in ('aot','cpp')]
    cases += [('cpp','odor',t) for t in a.levels if t!=1]
    for backend,condition,t in cases:
        label=f'{backend}-{condition}-t{t}';folder=a.output/label
        print(('[reuse] ' if a.reuse_builds else '[build] ')+label,flush=True)
        command=[sys.executable,str(Path(__file__).with_name('flywire_device.py')),'--graph',str(a.graph),
                 '--output',str(folder),'--backend',backend,'--condition',condition,'--threads',str(t),'--config',str(config_path)]
        if not a.reuse_builds:
            with (a.output/(label+'.log')).open('w') as log:
                subprocess.run(command,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
        builds[label]=json.loads((folder/'build.json').read_text());snapshots[label]=snapshot(folder)
        b=builds[label]
        manifest=json.loads((a.graph/'manifest.json').read_text())
        if b['backend']!=backend or b['condition']!=condition or b['threads']!=t or b['config']!=cfg:
            raise ValueError('existing build metadata mismatch: '+label)
        if b['graph_sha256']!=manifest['csr_sha256']:raise ValueError('existing graph hash mismatch')
        if backend=='cpp':cross_errors[label]=compare(snapshots[label],snapshots[f'aot-{condition}-t1'])
    if len({b['input_sha256'] for b in builds.values()})!=1:raise AssertionError('frozen input mismatch')
    groups_file=np.load(a.graph/'annotations.npz')
    groups={k:groups_file[k] for k in ('sensory','pn','all_pn','kc','mbon','cx')}
    biology=analyze({k:snapshots[f'aot-{k}-t1'] for k in CONDITIONS},groups,cfg)
    controls={};samples={str(t):{'aot':[],'cpp':[]} for t in a.levels}
    with tempfile.TemporaryDirectory(dir=a.output,prefix='replays-') as temp:
        temp=Path(temp)
        for condition in CONDITIONS:
            if condition=='odor':continue
            for backend in ('aot','cpp'):
                label=f'{backend}-{condition}-t1';controls[label]=[]
                for i in range(a.repeats):
                    result=measured_replay(backend,a.output/label,temp/f'control-{label}-{i}',1,snapshots[label],reader=read_snapshot)
                    if max(result['max_absolute_error'].values())!=0:raise AssertionError('non-deterministic control')
                    controls[label].append(result)
        for t in a.levels:
            for backend in ('aot','cpp'):
                label=f'{backend}-odor-t{1 if backend=="aot" else t}'
                measured_replay(backend,a.output/label,temp/f'warm-{backend}-{t}',t,snapshots[label],reader=read_snapshot)
        for i in range(a.repeats):
            levels=a.levels[i%len(a.levels):]+a.levels[:i%len(a.levels)]
            for t in levels:
                for backend in (('aot','cpp') if i%2==0 else ('cpp','aot')):
                    print(f'[replay] {i+1}/{a.repeats} {backend} threads={t}',flush=True)
                    label=f'{backend}-odor-t{1 if backend=="aot" else t}'
                    result=measured_replay(backend,a.output/label,temp/f'measured-{i}-{backend}-{t}',t,snapshots[label],reader=read_snapshot)
                    if max(result['max_absolute_error'].values())!=0:raise AssertionError('non-deterministic native replay')
                    samples[str(t)][backend].append(result)
    performance={}
    for t,row in samples.items():
        performance[t]={}
        for backend,values in row.items():
            times=[v['loop_seconds'] for v in values];median=statistics.median(times)
            performance[t][backend]={'median_seconds':median,'spread':(max(times)-min(times))/median,
                                      'peak_rss_bytes':max(v['peak_rss_bytes'] for v in values)}
    for t,row in performance.items():row['rust_self_speedup']=performance['1']['aot']['median_seconds']/row['aot']['median_seconds']
    acceptance={**biology['gates'],'all_backend_numeric_comparisons':True,'three_or_more_exact_replays':True,
                'rust_rss_below_500_MiB':all(v['aot']['peak_rss_bytes']<500*2**20 for v in performance.values()),
                'four_threads_faster_than_one':performance.get('4',{}).get('rust_self_speedup',0)>1,
                'four_thread_speedup_at_least_1_5':performance.get('4',{}).get('rust_self_speedup',0)>=1.5,
                'eight_threads_faster_than_four':('8' in performance and '4' in performance and performance['8']['aot']['median_seconds']<performance['4']['aot']['median_seconds']),
                'timing_spread_below_15_percent':all(v[b]['spread']<=.15 for v in performance.values() for b in ('aot','cpp'))}
    report={'schema':'b2-flywire-ei-benchmark-v1','reused_builds':a.reuse_builds,'host':platform.platform(),'rustc':toolchain,'cxx':cxx_version,
            'config':cfg,'repeats':a.repeats,'levels':a.levels,'builds':builds,'cross_backend_errors':cross_errors,
            'biology':biology,'performance':performance,'samples':samples,'control_replays':controls,'acceptance':acceptance,
            'graph_manifest':json.loads((a.graph/'manifest.json').read_text())}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'report.md').write_text(markdown(report))
    print(markdown(report),flush=True)


if __name__=='__main__':main()
