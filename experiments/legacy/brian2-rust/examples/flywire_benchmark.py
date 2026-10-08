"""Same-graph FlyWire LIF comparison: Rust AOT vs Brian2 C++ standalone.

Homogeneous excitatory LIF dynamics are an engineering workload, not a fitted
fruit-fly model. Every empirical contact contributes the same positive weight.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from brian2_rust.binary_topology import inspect_csr, csr_arrays, file_hash
from performance_suite import inspect_rustc, validate_rustc, benchmark_environment, source_provenance
from threaded_benchmark import prepare_cxx, default_cxx, parse_levels


def rss(value):
    return int(value)*(1 if sys.platform=='darwin' else 1024)


def build(args):
    import brian2 as b
    import brian2_rust as rust
    started=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=False)
    info=inspect_csr(args.graph/'connectome.b2csr')
    dataset=json.loads((args.graph/'manifest.json').read_text())
    if file_hash(info['path'])!=dataset['csr_sha256']:
        raise ValueError('dataset CSR checksum mismatch')
    n=info['source_count']
    project=args.output/'project'
    if args.backend=='aot':
        b.set_device('rust_standalone',engine='aot',threads=args.threads,
                     directory=project,runner=ROOT/'target/release/b2-runner')
    else:
        b.prefs.codegen.cpp.extra_compile_args=['-O3','-std=c++17','-fno-fast-math','-ffp-contract=off']
        b.prefs.devices.cpp_standalone.extra_make_args_unix=['-j2']
        b.prefs.devices.cpp_standalone.openmp_threads=0 if args.threads==1 else args.threads
        b.set_device('cpp_standalone',build_on_run=False)
    # All constants, initial values, ordering and output sampling are shared.
    g=b.NeuronGroup(n,'dv/dt=(drive-v)/(20*ms) : 1 (unless refractory)\ndrive : 1 (constant)',
                   threshold='v>1',reset='v=0',refractory=2*b.ms,method='euler',
                   dt=.1*b.ms,name='flywire_neurons')
    rng=np.random.default_rng(args.seed)
    g.v=rng.uniform(0,.95,n)
    g.drive=rng.uniform(1.1,1.3,n)
    s=b.Synapses(g,g,'multiplicity : 1 (constant)',on_pre='v_post += .0001*multiplicity',
                 delay=.5*b.ms,clock=g.clock,name='flywire_connections')
    if args.backend=='aot':
        rust.connect_binary_csr(s,info['path'],parameters={'multiplicity':0})
    else:
        offsets,targets,values=csr_arrays(info)
        sources=np.repeat(np.arange(n,dtype=np.int32),np.diff(offsets).astype(np.int64))
        s.connect(i=sources,j=targets.astype(np.int32),namespace={})
        s.multiplicity=values[0]
        del sources,targets,values,offsets
    monitor=b.StateMonitor(g,'v',record=np.unique(np.linspace(0,n-1,min(16,n),dtype=int)),name='flywire_trace')
    spikes=b.SpikeMonitor(g,name='flywire_spikes')
    net=b.Network(g,s,monitor,spikes)
    net.run(args.duration_ms*b.ms,namespace={})
    if args.backend=='cpp':
        b.device.build(directory=str(project),compile=True,run=True,with_output=False)
    elapsed=time.perf_counter()-started
    arrays={'v':g.variables['v'],'lastspike':g.variables['lastspike'],
            'not_refractory':g.variables['not_refractory'],
            'trace':monitor.variables['v'],'trace_t':monitor.variables['t'],
            'spike_i':spikes.variables['i'],'spike_t':spikes.variables['t'],
            'spike_count':spikes.variables['count']}
    snapshot={name:np.asarray(var.get_value()).copy() for name,var in arrays.items()}
    np.savez(args.output/'snapshot.npz',**snapshot)
    if args.backend=='cpp':
        layout={name:{'file':b.get_device().get_array_filename(var),
                      'dtype':np.dtype(var.dtype).str,'shape':list(snapshot[name].shape)}
                for name,var in arrays.items()}
        (args.output/'array_layout.json').write_text(json.dumps(layout,indent=2)+'\n')
        loop=float((project/'results/last_run_info.txt').read_text().split()[0])
    else:
        summary=json.loads((project/'rust/summary.json').read_text())
        loop=summary['timings']['simulation_and_recording_seconds']
    report={'backend':args.backend,'threads':args.threads,'end_to_end_seconds':elapsed,
            'first_simulation_seconds':loop,
            'frontend_peak_rss_bytes':rss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            'spikes':len(snapshot['spike_i']),'graph_sha256':dataset['csr_sha256'],
            'brian2':b.__version__,'python':platform.python_version(),
            **source_provenance(ROOT)}
    (args.output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


def compare(actual,expected):
    errors={}
    for name in expected:
        a,z=actual[name],expected[name]
        if name in ('spike_i','spike_count','not_refractory'):
            np.testing.assert_array_equal(a,z,err_msg=name)
        else:
            np.testing.assert_allclose(a,z,rtol=1e-12,atol=1e-14,err_msg=name)
        errors[name]=float(np.max(np.abs(a.astype(float)-z.astype(float)))) if a.size else 0.
    return errors


def read_snapshot(backend,build_dir,result):
    if backend=='cpp':
        layout=json.loads((build_dir/'array_layout.json').read_text())
        return {key:np.fromfile(result/item['file'],dtype=item['dtype']).reshape(item['shape'])
                for key,item in layout.items()}
    from brian2_rust.results import load_results
    model=json.loads((build_dir/'project/model.json').read_text())
    p=load_results(model,result)['populations'][0]
    return {'v':p['states']['v'],'lastspike':p['refractory']['lastspike'],
            'not_refractory':p['refractory']['not_refractory'],
            'trace':p['trace']['v'],'trace_t':p['times'],
            'spike_i':p['indices'],'spike_t':p['spike_times'],'spike_count':p['counts']}


def replay(backend,build_dir,output,threads,expected,reader=read_snapshot,cleanup=False):
    project=build_dir/'project'
    env={**os.environ,'B2_NUM_THREADS':str(threads),'OMP_NUM_THREADS':str(threads),'OMP_DYNAMIC':'FALSE'}
    if backend=='aot':
        command=[project/'native/b2-native',project/'native/instance.bin',output];cwd=None
    else:
        output.mkdir();command=[project/'main','--results_dir',str(output)+os.sep];cwd=project
    # A fresh small parent prevents Linux fork/exec rusage from including this
    # Python process's inherited, potentially large RSS before the native exec.
    usage_path=output.parent/(output.name+'.usage.json')
    measure=[sys.executable, str(Path(__file__).with_name('measure_native.py')),
             '--usage',str(usage_path),'--log',str(output.parent/(output.name+'.log'))]
    if cwd is not None:measure.extend(['--cwd',str(cwd)])
    subprocess.run([*measure,'--',*map(str,command)],env=env,check=True)
    usage=json.loads(usage_path.read_text())
    wall=usage['wall_seconds']
    errors=compare(reader(backend,build_dir,output),expected)
    if backend=='aot':
        summary=json.loads((output/'summary.json').read_text())
        loop=summary['timings']['simulation_and_recording_seconds']
        if summary['threads'] != threads:raise AssertionError('requested AOT thread count not used')
        delivered=summary['synaptic_events']
    else:
        loop=float((output/'last_run_info.txt').read_text().split()[0]);delivered=None
    result={'loop_seconds':loop,'wall_seconds':wall,'peak_rss_bytes':usage['peak_rss_bytes'],
            'delivered_weighted_edge_events':delivered,'max_absolute_error':errors,
            'parallel_on_pre':summary.get('parallel_on_pre') if backend=='aot' else None,
            'parallel_state_update':summary.get('parallel_state_update') if backend=='aot' else None}
    if cleanup:
        # C++ writes immutable connection arrays into every result directory.
        # The caller has already loaded and checked all outputs, so retaining
        # every replay would make disk usage grow with repeats and thread levels.
        shutil.rmtree(output)
    return result


def suite(args):
    toolchain=inspect_rustc(args.rustc);validate_rustc(toolchain)
    os.environ.update(benchmark_environment(toolchain))
    args.output.mkdir(parents=True,exist_ok=False)
    cxx,cxx_version=prepare_cxx(args.output,args.cxx)
    env={**os.environ,'CXX':cxx,'OMP_DYNAMIC':'FALSE'}
    builds={}
    for backend,threads in [('aot',1),*(('cpp',t) for t in args.levels)]:
        label=f'{backend}-t{threads}';folder=args.output/label
        command=[sys.executable,__file__,'--backend',backend,'--graph',str(args.graph),
                 '--output',str(folder),'--threads',str(threads),'--duration-ms',str(args.duration_ms),
                 '--seed',str(args.seed)]
        print(f'[build] {label}',flush=True)
        with (args.output/(label+'.log')).open('w') as log:
            subprocess.run(command,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
        builds[label]=json.loads((folder/'build.json').read_text())
    with np.load(args.output/'aot-t1/snapshot.npz') as saved:
        expected={name:saved[name] for name in saved.files}
    if len(expected['spike_i'])==0:raise AssertionError('inactive network is not a valid throughput workload')
    for level in args.levels:
        with np.load(args.output/f'cpp-t{level}/snapshot.npz') as actual:
            compare(actual,expected)
    samples={str(level):{'aot':[],'cpp':[]} for level in args.levels}
    with tempfile.TemporaryDirectory(prefix='flywire-replays-',dir=args.output) as temp:
        temp=Path(temp)
        for level in args.levels:
            for backend in ('aot','cpp'):
                build_dir=args.output/('aot-t1' if backend=='aot' else f'cpp-t{level}')
                replay(backend,build_dir,temp/f'warm-{backend}-{level}',level,expected,cleanup=True)
        for iteration in range(args.repeats):
            levels=args.levels[iteration%len(args.levels):]+args.levels[:iteration%len(args.levels)]
            for level in levels:
                for backend in (('aot','cpp') if iteration%2==0 else ('cpp','aot')):
                    print(f'[replay] {iteration+1}/{args.repeats} {backend} threads={level}',flush=True)
                    build_dir=args.output/('aot-t1' if backend=='aot' else f'cpp-t{level}')
                    result=replay(backend,build_dir,temp/f'r{iteration}-{backend}-{level}',level,expected,
                                  cleanup=True)
                    samples[str(level)][backend].append(result)
    summary={}
    for level,backends in samples.items():
        row={}
        for backend,values in backends.items():
            timings=[v['loop_seconds'] for v in values];median=statistics.median(timings)
            row[backend]={'median_seconds':median,'relative_spread':(max(timings)-min(timings))/median,
                          'native_peak_rss_bytes':max(v['peak_rss_bytes'] for v in values)}
        row['cpp_over_aot_speedup']=row['cpp']['median_seconds']/row['aot']['median_seconds']
        summary[level]=row
    dataset=json.loads((args.graph/'manifest.json').read_text())
    report={'schema':'b2-flywire-benchmark-v1','dataset':dataset,'duration_ms':args.duration_ms,
            'dt_ms':.1,'seed':args.seed,'threads':args.levels,'repeats':args.repeats,
            'model':'homogeneous excitatory LIF, tau=20ms, refractory=2ms, delay=.5ms; weight=.0001 per contact',
            'claim':'engineering workload on empirical weighted topology; no fitted fly biophysics or morphology',
            'correctness':'full final states/refractory, 16-neuron traces, all spike IDs/times/counts checked on every replay',
            'spikes':len(expected['spike_i']),'rustc':toolchain,'cxx':cxx_version,
            'host':platform.platform(),'builds':builds,'summary':summary,'samples':samples,
            'measurement_stable':all(row[backend]['relative_spread']<=.15
                                     for row in summary.values() for backend in ('aot','cpp'))}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# FlyWire full weighted-graph LIF benchmark','',
           f"{dataset['neurons']:,} neurons; {dataset['directed_pair_edges']:,} weighted edges; {dataset['biological_contacts']:,} biological contacts.",
           '',f'{args.duration_ms:g} simulated ms; dt=0.1 ms; {args.repeats} interleaved measured replays after warm-up.',
           '',report['claim'], '', 'Every replay passed '+report['correctness']+'.','',
           '| Threads | Rust AOT ms | C++ ms | C++ / Rust | Rust native MiB | C++ native MiB |',
           '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for level,row in summary.items():
        a,c=row['aot'],row['cpp']
        lines.append(f"| {level} | {a['median_seconds']*1000:.3f} | {c['median_seconds']*1000:.3f} | {row['cpp_over_aot_speedup']:.2f}x | {a['native_peak_rss_bytes']/2**20:.1f} | {c['native_peak_rss_bytes']/2**20:.1f} |")
    lines += ['', 'Preprocessing is measured once separately in the dataset manifest. Build end-to-end includes CSR validation, Brian object creation, topology materialisation, compilation, first simulation and result loading; native replay RSS excludes Python and compilers.', '',
              'Spreads and all raw samples are in report.json; values above 15% should be treated as noisy, not a stable release baseline.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['aot','cpp'],help=argparse.SUPPRESS)
    parser.add_argument('--threads',type=int,default=1,help=argparse.SUPPRESS)
    parser.add_argument('--levels',type=parse_levels,default=[1,4])
    parser.add_argument('--duration-ms',type=float,default=100)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--seed',type=int,default=783)
    parser.add_argument('--rustc',default=os.environ.get('B2_BENCHMARK_RUSTC','rustc'))
    parser.add_argument('--cxx',default=default_cxx())
    args=parser.parse_args();args.graph=args.graph.resolve();args.output=args.output.resolve()
    if not np.isfinite(args.duration_ms) or args.duration_ms<=0 or not np.isclose(args.duration_ms/.1,round(args.duration_ms/.1)):
        parser.error('duration must be a positive multiple of 0.1 ms')
    if not 3<=args.repeats<=10 or not 1<=args.threads<=256 or args.seed<0:
        parser.error('invalid repetitions/threads/seed')
    (build if args.backend else suite)(args)


if __name__=='__main__':main()
