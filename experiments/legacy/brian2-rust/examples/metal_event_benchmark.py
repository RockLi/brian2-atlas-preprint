"""Paired scan/sparse Metal delivery experiment; correctness is the only exit gate.

Synthetic event-transport workloads isolate silence, staggered 1% firing and
100% firing. These are route stress tests, not physiological network models.
The coupled case reuses the recurrent LIF + live summed benchmark.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

import brian2 as b
import brian2_rust
import numpy as np
from brian2_rust.metal import MetalExecutor
from brian2_rust.results import load_results
from metal_benchmark import compare, network, ROOT


def event_network(neurons, degree, steps, period, delay_ticks=0, delay_kind="uniform"):
    source = b.NeuronGroup(neurons,"v:1",threshold="v>1",refractory=max(1,period)*.1*b.ms,name="source")
    source.v = 0 if period == 0 else 2
    source.lastspike = -(np.arange(neurons) % max(1,period))*.1*b.ms
    target = b.NeuronGroup(neurons,"v:1",name="target")
    syn = b.Synapses(source,target,"w:1",on_pre="v_post+=w",clock=source.clock)
    indices=np.repeat(np.arange(neurons),degree)
    syn.connect(i=indices,j=(indices+np.tile(17*np.arange(degree)+1,neurons))%neurons)
    syn.w=1/1024
    syn.delay=(delay_ticks if delay_kind == "uniform" else np.arange(len(indices))%(delay_ticks+1))*.1*b.ms
    monitor=b.StateMonitor(target,"v",record=np.linspace(0,neurons-1,min(16,neurons),dtype=int))
    spikes=b.SpikeMonitor(source)
    return b.Network(source,target,syn,monitor,spikes),steps*.1*b.ms


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--case",choices=["silent","sparse","burst","coupled"],required=True)
    parser.add_argument("--neurons",type=int,default=8192)
    parser.add_argument("--degree",type=int,default=64)
    parser.add_argument("--steps",type=int,default=100)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--threads",default="1,4,8")
    parser.add_argument("--delay-ticks",type=int,default=0)
    parser.add_argument("--delay-kind",choices=["uniform","heterogeneous"],default="uniform")
    options=parser.parse_args()
    workers=sorted(set(map(int,options.threads.split(','))))
    if min(options.neurons,options.degree,options.steps,options.repeats)<1 or not workers or workers[0]<1 or workers[-1]>256:
        parser.error("positive workload sizes and workers within 1..256 required")
    if not 0 <= options.delay_ticks <= 1_000_000 or (options.case == "coupled" and options.delay_ticks):
        parser.error("delay ticks must be within 0..1,000,000; coupled case currently has zero delay")
    output=options.output.resolve();output.mkdir(parents=True,exist_ok=False)
    b.set_device("rust_standalone",engine="aot",directory=output/"rust-project",runner=ROOT/"target/release/b2-runner",threads=max(workers))
    b.seed(42)  # Fix the unused B2IR RNG identity as well as explicit initial values.
    if options.case == "coupled":
        net,duration=network("coupled",options.neurons,options.steps,options.degree)
    else:
        net,duration=event_network(options.neurons,options.degree,options.steps,{"silent":0,"sparse":100,"burst":1}[options.case],options.delay_ticks,options.delay_kind)
    net.run(duration)
    model=json.loads((output/"rust-project/model.json").read_text())
    reference=load_results(model,output/"rust-project/rust")
    files=[ROOT/"python/brian2_rust"/name for name in ("metal.py","metal_dag.py","metal_events.py","metal_delays.py","gpu_schedule.py","gpu_types.py","gpu_links.py","metal_runtime/bridge.m","metal_runtime/clocks.h","native.py","planner.py")]+[Path(__file__),ROOT/"examples/metal_benchmark.py"]
    report={"schema":"b2-metal-event-benchmark-v1","case":options.case,"neurons_per_group":options.neurons,
            "edges":options.neurons*options.degree,"steps":options.steps,"degree":options.degree,
            "repeats":options.repeats,"threads":workers,"model_hashes":model["protocol"]["layers"],
            "delay_ticks":options.delay_ticks,"delay_kind":options.delay_kind,
            "source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "host":platform.platform(),"rustc":subprocess.check_output(["rustc","-vV"],text=True),
            "trials":[],"correctness":{},"plans":{}}
    def persist():
        (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    with ExitStack() as stack:
        executors={route:stack.enter_context(MetalExecutor(model,output/route,numeric_mode="float32",event_delivery=route)) for route in ("scan","sparse")}
        report["device"]=executors["sparse"].device_name
        for route,executor in executors.items():
            report["plans"][route]={"sha256":executor.plan.sha256,"stages":len(executor.plan.dispatches),"compile_seconds":executor.compile_seconds,
                                    "dispatch_roles":[stage.role for stage in executor.plan.dispatches]}
        control=executors["scan"].run(compute="cpu-f32",workers=max(workers))
        report["storage_preparation_seconds"]={"scan":control["storage_preparation_seconds"]}
        for route,executor in executors.items():
            result=executor.run()
            report["storage_preparation_seconds"][route]=report["storage_preparation_seconds"].get(route,0)+result["storage_preparation_seconds"]
            report["correctness"][route+"_vs_f32"]=compare(result,control,exact=True)
            report["correctness"][route+"_vs_f64"]=compare(result,reference)
        mirror=executors["sparse"].run(compute="cpu-f32",workers=max(workers))
        report["correctness"]["sparse_cpu_vs_scan_cpu"]=compare(mirror,control,exact=True)
        report["delivered_events"]=sum(s["events"] for s in mirror["synapses"])
        persist()
        for repeat in range(options.repeats):
            cases=[("scan",0),("sparse",0)]+[("rust-f64",n) for n in workers]+[("cpu-f32",n) for n in workers]
            if repeat%2:cases.reverse()
            for backend,threads in cases:
                gpu_time=None
                if backend=="rust-f64":
                    destination=output/f"replay-{repeat}-{threads}"
                    started=time.perf_counter()
                    subprocess.run([str(output/"rust-project/native/b2-native"),str(output/"rust-project/native/instance.bin"),str(destination)],
                                   env={**os.environ,"B2_NUM_THREADS":str(threads)},check=True,capture_output=True,text=True)
                    wall=time.perf_counter()-started
                    result=load_results(model,destination)
                    timing=result["metadata"]["timings"]["simulation_and_recording_seconds"]
                    valid=compare(result,reference,exact=True)["passed"]
                    del result
                    shutil.rmtree(destination)
                else:
                    route="sparse" if backend=="cpu-f32" else backend
                    result=executors[route].run(compute="cpu-f32" if backend=="cpu-f32" else "metal",workers=max(1,threads))
                    wall=result["run_seconds"]
                    timing=sum(t["command_seconds"] for t in result["timings"])
                    gpu_time=sum(t["gpu_seconds"] for t in result["timings"]) if backend!="cpu-f32" else None
                    valid=compare(result,control,exact=True)["passed"]
                trial={"repeat":repeat,"backend":backend,"workers":threads,"simulation_seconds":timing,
                       "gpu_interval_seconds":gpu_time,"run_seconds":wall,"correctness_passed":valid}
                report["trials"].append(trial);persist();print(json.dumps(trial),flush=True)
        report["summary"]={}
        for backend,threads in cases:
            rows=[t for t in report["trials"] if (t["backend"],t["workers"])==(backend,threads)]
            samples=[t["simulation_seconds"] for t in rows]
            report["summary"][f"{backend}/{threads}"]={"median_seconds":statistics.median(samples),"min_seconds":min(samples),"max_seconds":max(samples),"run_median_seconds":statistics.median(t["run_seconds"] for t in rows)}
        summary=report["summary"];sparse=summary["sparse/0"]
        report["speedup_vs_scan"]=summary["scan/0"]["median_seconds"]/sparse["median_seconds"]
        report["conservative_speedup_vs_scan"]=summary["scan/0"]["min_seconds"]/sparse["max_seconds"]
        for backend in ("rust-f64","cpu-f32"):
            report["speedup_vs_best_"+backend]=min(summary[f"{backend}/{n}"]["median_seconds"] for n in workers)/sparse["median_seconds"]
        report["correctness_passed"]=all(c["passed"] for name,c in report["correctness"].items() if not name.endswith("f64")) and all(t["correctness_passed"] for t in report["trials"])
        report["strict_f64_equivalence_passed"]=report["correctness"]["sparse_vs_f64"]["passed"]
        persist();print(json.dumps({k:v for k,v in report.items() if k.startswith("speedup") or k.endswith("_passed")}),flush=True)
        if not report["correctness_passed"]:raise SystemExit("Sparse event correctness failed; inspect report.json")


if __name__ == "__main__":
    main()
