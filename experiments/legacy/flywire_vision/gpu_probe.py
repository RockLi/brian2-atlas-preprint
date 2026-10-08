"""Explicit full-graph Metal f32 probe and compiled CPU-f32 control."""
import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
from .run_experiment import diagnostic, save
from .simulation import SimulationConfig, encode_movie, build_model
from .stimuli import MovieConfig, movie


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-mib",type=int,default=1536)
    p.add_argument("--short",action="store_true",help="separate 60 ms full-graph numerical diagnostic")
    p.add_argument("--graph",type=Path)
    args=p.parse_args()
    from flywire_mnist.backends import GPU
    from brian2_rust.protocol import attach_protocol
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.artifact/"manifest.json").read_text())
    cfg=SimulationConfig(**manifest["config"])
    channels=json.loads((args.artifact/"channels.json").read_text())
    template=json.loads((args.artifact/"model.json").read_text())
    if args.short:
        if args.graph is None:p.error("--short requires --graph")
        from flywire_mnist.graph import load_graph
        cfg=replace(cfg,frames=4,warmup_ms=10.,tail_ms=10.)
        template=build_model(load_graph(args.graph),channels,cfg,Path(__file__).resolve().parents[2]/"target/release/b2-runner")
    pi=next(i for i,pop in enumerate(template["definition"]["populations"]) if pop["name"]=="visual_input")
    executor=GPU(template,args.output/"metal",backend="metal",max_bytes=args.max_mib*1024**2)
    report={"status":"running","scope":"60 ms full-graph numerical diagnostic; not the 600 ms viewer trial" if args.short else "full graph, two diagnostic clips; not full classification validation","config":asdict(cfg),"numeric_profile":"metal-f32 / compiled-cpu-f32", "trials":[]}
    save(args.output/"report.json",report)
    try:
        for kind in (("right",) if args.short else ("blank","right")):
            model=copy.deepcopy(template)
            frames=movie(kind,cfg.seed,MovieConfig(frames=cfg.frames)) if args.short else diagnostic(kind,cfg.seed)
            i,t,_=encode_movie(frames,channels,cfg)
            model["instance"]["populations"][pi]["spike_generator"]={"spike_indices":i.tolist(),"spike_ticks":t.tolist()}
            attach_protocol(model)
            print(json.dumps({"stage":"Metal","kind":kind}),flush=True)
            start=time.perf_counter()
            actual=executor.run(model,kind)
            metal_seconds=time.perf_counter()-start
            print(json.dumps({"stage":"CPU f32 control","kind":kind,"metal_seconds":metal_seconds}),flush=True)
            control=executor.previous.run(max_buffer_bytes=args.max_mib*1024**2,compute="cpu-f32")
            checks=[]
            for definition,a,b in zip(model["definition"]["populations"],actual["populations"],control["populations"]):
                checks.append({"population":definition["name"],"spikes":len(a["indices"]),
                               "events_exact":bool(np.array_equal(a["indices"],b["indices"]) and np.array_equal(a["spike_ticks"],b["spike_ticks"])),
                               "state_max_abs":{n:float(np.max(np.abs(v.astype(float)-b["states"][n].astype(float)),initial=0)) for n,v in a["states"].items()}})
            report["trials"].append({"kind":kind,"metal_prepare_and_run_seconds":metal_seconds,"checks":checks})
            save(args.output/"report.json",report)
            print(json.dumps(report["trials"][-1]),flush=True)
        report["status"]="complete"
        report["all_f32_events_and_states_exact"]=all(p["events_exact"] and all(v==0 for v in p["state_max_abs"].values()) for t in report["trials"] for p in t["checks"])
        save(args.output/"report.json",report)
    except Exception as error:
        report.update(status="failed",error=str(error))
        save(args.output/"report.json",report)
        raise
    finally:executor.close()


if __name__=="__main__":main()
