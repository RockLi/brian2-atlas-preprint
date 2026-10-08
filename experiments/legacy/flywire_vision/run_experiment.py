"""Run full-graph P1 CPU trials and export their real activity for the local viewer."""
import argparse
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .simulation import SimulationConfig, bind_graph, build_model, cut_model, encode_movie, summarize
from .stimuli import KINDS, movie


def save(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def array64(value, dtype):
    return base64.b64encode(np.asarray(value,dtype=dtype).tobytes()).decode()


def diagnostic(kind, seed):
    if kind == "blank": return np.full((40,48,48),.5,dtype=np.float32)
    if kind == "bright": return np.full((40,48,48),.9,dtype=np.float32)
    if kind == "dark": return np.full((40,48,48),.1,dtype=np.float32)
    if kind == "flash":
        frames=np.full((40,48,48),.5,dtype=np.float32)
        frames[5:10,21:27,21:27]=.9
        return frames
    return movie(kind,seed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources",type=Path,required=True)
    p.add_argument("--graph",type=Path,required=True)
    p.add_argument("--runner",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--weight",type=float,default=16.)
    args=p.parse_args()
    if not 1<=args.threads<=256: p.error("threads must be in 1..256")
    from flywire_mnist.graph import load_graph
    from flywire_mnist.backends import CPU
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    started=time.perf_counter()
    cfg=SimulationConfig(input_weight_mv=args.weight)
    args.output.mkdir(parents=True,exist_ok=False)
    graph=load_graph(args.graph)
    channels,groups,audit=bind_graph(graph,args.sources)
    print(json.dumps({"stage":"graph_validated","neurons":len(graph.root_ids),"edges":graph.edge_count,
                      "groups":{k:len(v) for k,v in groups.items()}}),flush=True)
    template=build_model(graph,channels,cfg,args.runner)
    cut=cut_model(template,channels)
    save(args.output/"model.json",template)
    save(args.output/"model-cut.json",cut)
    save(args.output/"channels.json",channels)
    save(args.output/"groups.json",{k:v.tolist() for k,v in groups.items()})
    manifest={"schema":"flywire-vision-p1-v1","status":"running","backend":"CPU / Rust AOT f64",
              "config":asdict(cfg),"threads":args.threads,"neurons":len(graph.root_ids),"edges":graph.edge_count,
              "graph_identity":graph.identity,"source_lock":audit["source_lock"],
              "definition":template["protocol"]["layers"]["definition"],
              "input_encoding":"one-to-one Mi1 positive / Tm1 negative luminance contrast; regular phase accumulator",
              "sources_sha256":{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in Path(__file__).parent.glob("*.py")},
              "scope":"Full-graph artificial visual drive; no trained classifier or natural-vision validation"}
    save(args.output/"manifest.json",manifest)
    compiler=CPU(template,args.output/"compile",threads=args.threads)
    runners={"intact":FrozenCPU(template,compiler.native,args.output/"intact",population="visual_input",threads=args.threads),
             "cut":FrozenCPU(cut,compiler.native,args.output/"cut",population="visual_input",threads=args.threads)}
    manifest["prepare_seconds"]=time.perf_counter()-started
    trials=[]
    sequence=[("intact",k) for k in ("blank","flash","bright","dark",*KINDS,"right")]
    sequence += [("cut",k) for k in ("blank","right","looming","bright","dark")]
    for index,(condition,kind) in enumerate(sequence):
        frames=diagnostic(kind,cfg.seed)
        spike_i,spike_t,strength=encode_movie(frames,channels,cfg)
        key=f"{index:02d}-{condition}-{kind}"
        result=runners[condition].run(spike_i,spike_t,key)
        summary,input_counts=summarize(result,template if condition=="intact" else cut,groups,channels,cfg)
        ni=next(i for i,pop in enumerate(template["definition"]["populations"]) if pop["name"]=="flywire_neurons")
        summary["final_state_sha256"]={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in result["populations"][ni]["states"].items()}
        trial={"id":key,"kind":kind,"condition":condition,"seed":cfg.seed,"summary":summary,
               "input_spikes":len(spike_i),"input_sha256":hashlib.sha256(np.stack([spike_t,spike_i],axis=1).astype('<i8').tobytes()).hexdigest(),
               "movie_sha256":hashlib.sha256(frames.tobytes()).hexdigest(),
               "frames_u8":array64(np.rint(frames*255),"u1"),"frame_shape":list(frames.shape),
               "strength_u8":array64(np.rint(strength*255),"u1"),
               "input_counts_u16":array64(input_counts,"<u2"),"count_shape":list(input_counts.shape)}
        if np.any(input_counts>65535):raise ValueError("display count overflow")
        trials.append(trial)
        save(args.output/(key+".json"),trial)
        print(json.dumps({"trial":key,"input_spikes":len(spike_i),"neural_spikes":summary["total_spikes"],
                          "active":summary["active_during_stimulus"],"seconds":summary["simulation_wall_seconds"]}),flush=True)
    by_key={(r["condition"],r["kind"]):r for r in trials}
    right=[t for t in trials if t["condition"]=="intact" and t["kind"]=="right"]
    reset=all(right[0]["summary"][k]==right[1]["summary"][k] for k in ("neural_event_sha256","final_state_sha256"))
    cut_blank=by_key["cut","blank"]["summary"]["noninput_event_sha256"]
    checks={"A_B_A_events_and_final_states_exact":reset,
            "cut_downstream_equals_cut_blank":{k:by_key["cut",k]["summary"]["noninput_event_sha256"]==cut_blank for k in ("right","looming","bright","dark")},
            "cut_preserves_external_input":{k:by_key["cut",k]["input_sha256"]==by_key["intact",k]["input_sha256"] for k in ("right","looming","bright","dark")}}
    passed=reset and all(checks["cut_downstream_equals_cut_blank"].values()) and all(checks["cut_preserves_external_input"].values())
    save(args.output/"checks.json",checks)
    manifest.update(status="complete" if passed else "failed_checks",total_seconds=time.perf_counter()-started,
                    trial_count=len(trials),checks=checks)
    save(args.output/"manifest.json",manifest)
    payload={"manifest":manifest,"channels":channels,"trials":trials}
    save(args.output/"viewer-data.json",payload)
    print(json.dumps({"status":manifest["status"],"checks":checks,"seconds":manifest["total_seconds"]}),flush=True)
    if not passed:raise RuntimeError("P1 paired checks failed")


if __name__=="__main__":main()
