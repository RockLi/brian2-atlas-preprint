"""Persistent benchmark worker: reset and replay compiled recurrent models."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

BACKENDS=('rust-f64','cpu-f32','cuda','brian2cuda','brian2genn','genn')
COMPILERS={'nvcc','rustc','cc','c++','gcc','g++','clang','clang++','make','cmake','ninja','genn-buildmodel.sh'}


@contextmanager
def forbid_compilation():
    original=subprocess.Popen
    def checked(command,*args,**kwargs):
        tokens=shlex.split(command) if isinstance(command,str) else [os.fsdecode(t) for t in command]
        if any(Path(t).name in COMPILERS for t in tokens):
            raise RuntimeError('Compilation attempted inside a precompiled replay: '+repr(tokens))
        return original(command,*args,**kwargs)
    subprocess.Popen=checked
    try:yield
    finally:subprocess.Popen=original


def artifact_hashes(directory):
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob('*')) if p.is_file() and
        (p.name in {'main','b2-native'} or p.suffix in {'.so','.cubin','.dylib'})}


def native_arrays(raw):
    p=raw['populations'][0]
    return dict(v=np.asarray(p['states']['v']).copy(),synaptic_current=np.asarray(p['states']['I_syn']).copy(),
                ticks=np.asarray(p['spike_ticks'],np.int64).copy(),indices=np.asarray(p['indices'],np.int64).copy())


class Replay:
    def __init__(self,backend,neurons,steps,degree,output):
        from gpu_recurrent import brian_run,genn_run
        self.backend,self.steps,self.output=backend,steps,output
        self.context={};self.count=0;self.executor=None
        started=time.perf_counter()
        if backend=='genn':
            result,timing=genn_run(neurons,steps,degree,output,False,policy='brian-euler',prepared=self.context)
        else:
            result,timing=brian_run('rust' if backend=='rust-f64' else backend,neurons,steps,degree,
                output,False,prepared=self.context,state_dtype=np.float64 if backend=='rust-f64' else None)
        if backend in {'cuda','rust-f64'}:
            self.model=json.loads((output/'project/model.json').read_text())
        if backend=='cuda':
            from brian2_rust.cuda import CudaExecutor
            self.executor=CudaExecutor(self.model,output/'project/cuda',numeric_mode='float32',event_delivery='sparse',dag_execution='auto')
        self.setup_seconds=time.perf_counter()-started;self.bootstrap=result;self.setup_timing=timing
        self.artifacts=artifact_hashes(output)
        if not self.artifacts:raise RuntimeError('No compiled artifact found after preparation')

    def _brian_arrays(self):
        import brian2 as b
        from gpu_baseline import canonical_result
        from gpu_recurrent import DT_MS
        c=self.context;p=c['population'];m=c['monitor'];dev=c['device']
        # These quantities must come from the new output files, not initialization
        # caches in old Brian standalone versions.
        for var in (p.variables['v'],p.variables['I_syn'],m.variables['t'],m.variables['i']):dev.array_cache[var]=None
        result=canonical_result(p.v[:],m.t[:]/b.ms,m.i[:],DT_MS)
        result['synaptic_current']=np.asarray(p.I_syn[:]).copy()
        return {k:v.copy() for k,v in result.items()}

    def run(self):
        self.count+=1;start=time.perf_counter();details={};result_directory=None
        with forbid_compilation():
            if self.backend=='cuda':
                raw=self.executor.run();result=native_arrays(raw)
                details=dict(runtime=raw['cuda_runtime'],timings=raw['timings'])
            elif self.backend=='cpu-f32':
                from brian2_rust.cuda import CudaExecutor
                raw=CudaExecutor._cpu_control(self.context['control'],512*1024**2,1)
                result=native_arrays(raw);details=dict(native_run_seconds=raw['run_seconds'])
            elif self.backend=='rust-f64':
                from brian2_rust.results import load_results
                artifact=self.context['device'].native_artifact
                directory=self.output/f'native-replay-{self.count}'
                if directory.exists():raise RuntimeError('Result directory already exists')
                subprocess.run([str(artifact['binary']),str(artifact['instance']),str(directory)],check=True,
                    stdout=subprocess.DEVNULL,env={**os.environ,'B2_NUM_THREADS':'1','B2_THREAD_AFFINITY':'auto'})
                result=native_arrays(load_results(self.model,directory));result_directory=directory
            elif self.backend in {'brian2cuda','brian2genn'}:
                dev=self.context['device'];project=self.output/'project'
                if self.backend=='brian2cuda':
                    name=f'replay-{self.count}'
                    if (project/name).exists():raise RuntimeError('Result directory already exists')
                    dev.run(directory=str(project),results_directory=name,with_output=False);result_directory=project/name
                else:
                    # GeNN 4's standalone wrapper hardcodes results/. It contains
                    # outputs only; static_arrays/ and compiled artifacts stay intact.
                    results=project/'results'
                    if results.is_symlink():raise RuntimeError('Unexpected results symlink')
                    if results.exists():shutil.rmtree(results)
                    results.mkdir()
                    dev.run(directory=str(project),use_GPU=True,with_output=False);result_directory=results
                result=self._brian_arrays()
                details=dict(backend_last_run_seconds=float(dev._last_run_time),scope='backend-specific diagnostic only')
            else:
                from gpu_baseline import canonical_result
                from gpu_recurrent import DT_MS
                model=self.context['model'];pop=self.context['population']
                # Loading the compiled model resets state, topology, queues, time
                # and recordings through the supported GeNN lifecycle API.
                model.load(num_recording_timesteps=self.steps)
                try:
                    for _ in range(self.steps):model.step_time()
                    model.pull_recording_buffers_from_device();pop.vars['v'].pull_from_device();pop.vars['I_syn'].pull_from_device()
                    times,indices=pop.spike_recording_data[0]
                    result=canonical_result(pop.vars['v'].current_values.copy(),times.copy(),indices.copy(),DT_MS)
                    result['synaptic_current']=pop.vars['I_syn'].current_values.copy().reshape(-1)
                finally:model.unload()
        elapsed=time.perf_counter()-start
        if result_directory is not None:
            details['output_file_bytes']=sum(p.stat().st_size for p in result_directory.rglob('*') if p.is_file())
        details['result_array_bytes']=sum(a.nbytes for a in result.values())
        return result,dict(wall_seconds=elapsed,details=details)

    def close(self):
        if self.executor is not None:self.executor.close()


def write_result(output,name,result):
    path=output/(name+'.npz');np.savez_compressed(path,**result)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        arrays={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in result.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=BACKENDS,required=True);parser.add_argument('--neurons',type=int,default=4096)
    parser.add_argument('--steps',type=int,default=2048);parser.add_argument('--degree',type=int,default=32)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from gpu_recurrent import validate_size,configuration
    if not 1<=args.neurons<=65536 or not 1<=args.steps<=4096:parser.error('bounded positive workload required')
    try:validate_size(args.neurons,args.degree)
    except ValueError as error:parser.error(str(error))
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    # Preserve a dedicated protocol descriptor, then direct all Python/native
    # compiler stdout to stderr. Compiler output cannot impersonate JSON replies.
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1);os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    replay=None
    def emit(value):protocol.write(json.dumps(value,allow_nan=False)+'\n');protocol.flush()
    try:
        replay=Replay(args.backend,args.neurons,args.steps,args.degree,args.output)
        emit(dict(event='ready',backend=args.backend,configuration=configuration(args.neurons,args.steps,args.degree),
            arithmetic=('full f64 storage/expressions' if args.backend=='rust-f64' else 'float32; explicit brian-euler' if args.backend=='genn' else 'float32'),
            lifecycle={'rust-f64':'fresh compiled native process','cpu-f32':'retained C++ library and initial topology, fresh writable arrays','cuda':'retained executor and device buffers, reset writable initial state','brian2cuda':'fresh standalone process and output directory','brian2genn':'fresh standalone process after clearing generated outputs','genn':'reload compiled model, run, synchronized pull, unload'}[args.backend],
            setup_seconds=replay.setup_seconds,setup_timing=replay.setup_timing,artifacts=replay.artifacts,
            bootstrap=write_result(args.output,'bootstrap',replay.bootstrap)))
        for line in sys.stdin:
            request=json.loads(line)
            if request['command']=='close':
                replay.close();replay=None;emit(dict(event='closed'));break
            if request['command']!='run':raise ValueError('Unknown worker command')
            result,timing=replay.run()
            # Hashing, NPZ compression and artifact verification are outside the
            # measured reset-to-result interval, and cannot hide a changed binary.
            if artifact_hashes(args.output)!=replay.artifacts:raise RuntimeError('Compiled artifact changed during replay')
            artifact=write_result(args.output,f'replay-{replay.count}',result)
            emit(dict(event='result',sample=request['sample'],**timing,result=artifact))
    except BaseException as error:
        emit(dict(event='error',error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        if replay is not None:replay.close()
        protocol.close()


if __name__=='__main__':main()
