#!/usr/bin/env python3
"""E5 native v5 CPU FP64 binary Q0 qualification, never performance.

Preparation is AST-only. Run only in a separately scheduled remote window.
No frozen source is changed. Oracle is the independent e5_oracle_r1 module.
"""
from __future__ import annotations
import argparse
import copy
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback
import e5_oracle_r1 as oracle

ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL = 1e-10, 1e-8
np = None
FROZEN = {
    "protocol/semantic-contracts.json": "eaea29ada59e3cb8395a99cbacad843ef13cb80369020e40c47d78454b513a0a",
    "evidence/e5-preparation-r1/fixture-contract.json": "0fec9d02b121dce91b202f0f51a7805063ba81c65f6b9f1b696d85f03e692a31",
    "tools/e5_oracle_r1.py": "e97e982996dea2303d6a6b5c9f75747ee48a58cc3f8845c8728fcb76ae05ea4b",
}


def dump(path, value):
    oracle.write_new(path, value)


def load_native():
    sys.path.insert(0, str(ROOT/"snapshot"))
    sys.path.insert(0, str(ROOT/"snapshot/brian2-rust/python"))
    from brian2_rust.training import lif_training_plan, NativeLIFTrainer
    from brian2_rust.training_dynamic import compile_dynamic_transform, dynamic_action
    from brian2_rust.training_equations import neuron_parameter_bank
    expected = ROOT/"snapshot/brian2-rust/python/brian2_rust/training.py"
    if Path(sys.modules["brian2_rust.training"].__file__).resolve() != expected:
        raise RuntimeError("Native source import outside frozen snapshot")
    return lif_training_plan, NativeLIFTrainer, compile_dynamic_transform, dynamic_action, neuron_parameter_bank


def builder(sizes, state_counts, banks, seed):
    lif, Trainer, compile_transform, make_action, bank = load_native()
    identity = [[[dict(op="state", index=i)] for i in range(count)] for count in state_counts]
    plan = lif(sizes, backend="cpu", beta=.995, threshold=1.,
        reset="subtract", detach_reset=True, surrogate_slope=5.,surrogate_scale=1.,
        optimizer="adam",learning_rate=.001,logit_scale=5.,seed=seed,
        projections=[bank(len(w)) for w in banks],state_equations=identity,
        state_resets=identity,clock=dict(origin=0.,dt=.0001),max_tape_bytes=64*1024**2)
    programs, actions, phases = [], [], []
    def add(code, names, parameters, reads, owner, phase, trigger=None, detach=False):
        transform = compile_transform(code, states=names, parameters=parameters)
        program_set = len(programs)
        programs.append(transform["programs"])
        actions.append(make_action(transform,reads,owner=owner,program_set=program_set,
                                   trigger=trigger,detach_trigger=detach))
        phases.append(dict(phase=phase,owner=owner,source=code))
    def threshold(owner, slot, *, inclusive=False, margin=False):
        actions.append(dict(owner=owner,reads=[slot],writes=[],program_set=None,
            threshold=owner,trigger=None,threshold_margin=margin,
            threshold_inclusive=inclusive))
        phases.append(dict(phase="threshold",owner=owner,slot=slot,
                           inclusive=inclusive,margin=margin))
    def finish(width, voltage):
        plan.update(schema="b2-dynamic-training-plan-v5",dynamic=dict(
            initial=[0.]*width,initial_parameters=[None]*width,detached=[False]*width,
            voltage=voltage,program_sets=programs,actions=actions))
        return plan, phases
    return Trainer, add, threshold, finish


def build_plan(case):
    spec,_ = oracle.configuration()
    D = oracle.dimensions(case["model"])
    H,O = 4,2
    out, margin = D*H, D*H+O
    Trainer,add,threshold,finish = builder([2,H,O],[D,1],case["weights"],case["seed"])
    model = case["model"]
    params = dict(spec[model].get("parameters",{}))
    params["dt"] = spec["common"]["dt_ms"]
    if model == "AdEx":
        names = dict(v=0,a=1,m=2)
        drift = ("ov=v\noa=a\n"
            "v=ov+dt*(gL_nS*(EL_mV-ov)+gL_nS*DeltaT_mV*exp((ov-VT_mV)/DeltaT_mV)+I_pA-oa)/C_pF\n"
            "a=oa+dt*(a_adapt_nS*(ov-EL_mV)-oa)/tau_a_ms\nm=(v-Vcut_mV)/2")
    elif model == "Izhikevich":
        names = dict(v=0,u=1,m=2)
        # 'a' is a parameter; u is the second state. Both outputs use old v/u.
        drift = ("ov=v\nou=u\nv=ov+dt*(.04*ov*ov+5*ov+140-ou+I)\n"
                 "u=ou+dt*a*(b*ov-ou)\nm=(v-30)/10")
    else:
        names = dict(v=0,a=1,p=2,q=3,m=4)
        drift = ("ov=v\noa=a\nop=p\noq=q\n"
                 "v=ov+dt*(-ov-.2*oa+.4*op)/2\n"
                 "a=oa+dt*(-oa)/10\np=op+dt*(oq-op)\n"
                 "q=oq+dt*(-oq)/.5\nm=v-1")
    for h in range(H):
        reads = [d*H+h for d in range(D)]+[margin+h]
        add(drift,names,params,reads,h,"drift")
    for j in range(O):
        add("v=.995*v",dict(v=0),{},[out+j],H+j,"drift")
    for h in range(H):
        threshold(h,margin+h,inclusive=model=="Izhikevich",margin=True)
    for j in range(O):
        threshold(H+j,out+j)
    target = 3*H if D==4 else 0
    for i in range(2):
        for h in range(H):
            add("v+=w",dict(v=0),dict(w=(0,i*H+h)),[target+h],h,
                "synapse",dict(external=True,index=i))
    for h in range(H):
        for j in range(O):
            add("v+=w",dict(v=0),dict(w=(1,h*O+j)),[out+j],H+j,
                "synapse",dict(external=False,index=h))
    for h in range(H):
        trigger=dict(external=False,index=h)
        if model=="AdEx":
            add("v=Vr_mV\na+=b_pA",dict(v=0,a=1),params,[h,H+h],h,
                "reset",trigger,True)
        elif model=="Izhikevich":
            add("v=c\nu+=d",dict(v=0,u=1),params,[h,H+h],h,
                "reset",trigger,True)
        else:
            add("v-=1",dict(v=0),{},[h],h,"reset",trigger,True)
            add("a+=.1",dict(a=0),{},[H+h],h,"reset",trigger,False)
    for j in range(O):
        add("v-=1",dict(v=0),{},[out+j],H+j,"reset",
            dict(external=False,index=H+j),True)
    plan,phases=finish(margin+H,list(range(H))+list(range(out,out+O)))
    order = {"drift":0,"threshold":1,"synapse":2,"reset":3}
    if [order[p["phase"]] for p in phases] != sorted(order[p["phase"]] for p in phases):
        raise RuntimeError("Action schedule violates frozen common order")
    return Trainer,plan,phases


def admission(plan,state,inputs,labels,initial,operation):
    """Frozen native storage accounting only; independent of oracle equations."""
    d=plan["dynamic"]
    if plan["backend"]!="cpu" or plan.get("mpi_ranks"):
        raise ValueError("This qualification is CPU only")
    if any(d.get(k) for k in ("delay_layout","migration","parameter_maps",
                             "threshold_references","clocks","integer_parameters")):
        raise ValueError("Unbudgeted dynamic extension")
    if any(a.get("indirect") for a in d["actions"]):
        raise ValueError("Unbudgeted indirect action")
    B,T,I=len(inputs),len(inputs[0]),plan["sizes"][0]
    N=sum(plan["sizes"][1:]);O=plan["sizes"][-1];S=len(d["initial"])
    P=sum(len(r) for r in plan["masks"])
    topology=sum(len(p["sources"])*24+128 for p in plan["projections"])
    reads=sum(len(a["reads"]) for a in d["actions"])
    nodes=sum(len(p) for ps in d["program_sets"] for p in ps)
    components=dict(spike_tape=B*T*N*32,live_neurons=B*N*32,logits=B*O*16,
        parameter_optimizer=P*48,topology=topology,
        v4_vector=B*T*S*16+B*S*64+sum(len(p) for p in plan["state_equations"])*128*128+8192,
        inputs=B*T*(I*8+48)+B*8,
        dynamic=reads*B*T*8+reads*32+len(d["actions"])*256+nodes*128+S*B*64+129*3*8)
    tape=sum(components.values());initial_bytes=S*B*64+P*48
    request=dict(plan=plan,state=state,operation=operation,inputs=inputs,
                 labels=labels,initial=initial,start_tick=0)
    wire=json.dumps(oracle.plain(request),sort_keys=True,separators=(",",":"),
                    ensure_ascii=False,allow_nan=False).encode()
    import hashlib
    accepted=len(wire)<=64*1024**2 and max(tape,initial_bytes,P*48+topology)<=plan["max_tape_bytes"]
    return dict(status="admitted" if accepted else "budget_rejected",
        request_json_bytes=len(wire),request_sha256=hashlib.sha256(wire).hexdigest(),
        exact_tape_bytes=tape,initial_bytes=initial_bytes,components=components,
        scope="software admission, not peak RSS")


def comparison(actual,expected,exact=False):
    a,e=np.asarray(actual),np.asarray(expected)
    if a.shape!=e.shape:
        return dict(passed=False,actual_shape=list(a.shape),expected_shape=list(e.shape))
    finite=bool(np.all(np.isfinite(a)) and np.all(np.isfinite(e)))
    passed=bool(np.array_equal(a,e)) if exact else bool(np.all(np.abs(a-e)<=ATOL+RTOL*np.abs(e)))
    return dict(passed=finite and passed,shape=list(e.shape),
                max_absolute_error=float(np.max(np.abs(a-e),initial=0.)) if finite else None)


def classify(error):
    text=str(error).lower()
    if isinstance(error,subprocess.TimeoutExpired):return "timeout"
    if isinstance(error,MemoryError):return "oom"
    if isinstance(error,FloatingPointError) or "nonfinite" in text or "non-finite" in text:return "numerical_divergence"
    if "budget" in text:return "budget_rejected"
    return "qualification_error"


class Calls:
    def __init__(self,directory):
        self.directory=directory;self.records=[]

    def run(self,trainer,inputs,labels,initial,operation,tag):
        record=dict(tag=tag,operation=operation,performance_run=False,status="started")
        path=self.directory/f"call-{len(self.records)+1:03d}-{tag}.json"
        self.records.append(path.name)
        try:
            check=admission(trainer.plan,trainer.state,inputs,labels,initial,operation)
            record["admission"]=check
            if check["status"]!="admitted":
                raise RuntimeError("software budget rejected")
            begin=time.perf_counter_ns()
            result=trainer.execute(inputs,labels,operation=operation,initial=initial,start_tick=0)
            record.update(status="executed",result=result,public_api_ns=time.perf_counter_ns()-begin,
                          tape_bytes_match=result["tape_bytes"]==check["exact_tape_bytes"])
            if result["backend"]!="cpu" or result.get("gpu_dispatches")!=0:
                raise RuntimeError("Unexpected backend")
            return result
        except Exception as error:
            record.update(status=classify(error),error_type=type(error).__name__,
                          error=str(error),traceback=traceback.format_exc())
            raise
        finally:
            dump(path,record)


def check_result(checks,prefix,result,expected):
    for key in ("loss","logits","spikes","final_state","initial_state_gradients"):
        checks[f"{prefix}_{key}"]=comparison(result[key],expected[key],exact=key=="spikes")
    for bank,gradient in enumerate(expected["gradients"]):
        checks[f"{prefix}_bank{bank}_gradient"]=comparison(result["gradients"][bank],gradient)
    checks[f"{prefix}_full_bptt"]=dict(passed=result["gradient_scope"]=="full-bptt")


def report_value(reference):
    value={k:v for k,v in reference.items() if k!="anchors"}
    value["hidden_margins"]=np.stack([t["mh"] for t in reference["anchors"]],axis=1)
    value["output_margins"]=np.stack([t["mo"] for t in reference["anchors"]],axis=1)
    return value


def reset_witness(case,plan,Trainer,runner,calls,checks):
    """Separate predeclared one-tick diagnostic, never replaces six Q0 cases."""
    model=case["model"]
    if not case["boundary"] or model=="synthetic_four_state":
        return dict(applicable=False)
    modified=copy.deepcopy(case)
    modified["inputs"]=[list(map(list,sample)) for sample in case["inputs"]]
    for b in range(2):
        modified["inputs"][b][0]=[1.,1.]
    expected=oracle.forward_vjp(modified)
    inputs=[[[1.,1.]],[[1.,1.]]]
    trainer=Trainer(plan,runner=runner,weights=case["weights"])
    actual=calls.run(trainer,inputs,case["labels"],case["initial"],"evaluate","reset-feed-witness")
    checks["reset_witness_full_state"]=comparison(actual["final_state"],expected["states"][:,0])
    checks["reset_witness_spikes"]=comparison(actual["spikes"],expected["spikes"][:,:1],exact=True)
    mask=expected["spikes"][:,0,:4].astype(bool)
    wi=np.array(case["weights"][0]).reshape(2,4)
    feed=np.broadcast_to(wi.sum(axis=0),(2,4))
    overlap=mask & (feed!=0)
    checks["reset_witness_has_nonzero_feed_overlap"]=dict(passed=bool(np.any(overlap)))
    reset=oracle.configuration()[0][model]["parameters"]["Vr_mV" if model=="AdEx" else "c"]
    values=np.asarray(actual["final_state"])[:,:4][overlap]
    checks["hard_reset_overrides_same_tick_input"]=comparison(values,np.full_like(values,reset),exact=True)
    return dict(applicable=True,scope="extra one-tick all-ones input; original boundary initial",
                overlap_count=int(overlap.sum()),hard_spikes=mask,feed=feed,
                physical_final_v=np.asarray(actual["final_state"])[:,:4],
                expected_reset=reset,no_tuning_after_results=True)


def qualify_case(case,directory,runner):
    directory.mkdir(parents=True,exist_ok=False)
    dump(directory/"fixture.json",case)
    calls=Calls(directory)
    report=dict(case=case["id"],performance_run=False,status="unqualified",
                calls=calls.records,checks={})
    checks=report["checks"]
    try:
        expected=oracle.forward_vjp(case)
        dump(directory/"oracle.json",report_value(expected))
        fd=oracle.local_surrogate_fd(case)
        dump(directory/"oracle-local-surrogate-fd.json",fd)
        checks["oracle_local_surrogate_fd"]=dict(passed=fd["passed"])
        if not fd["passed"]:
            report["status"]="oracle_selfcheck_failure"
            return report
        Trainer,plan,phases=build_plan(case)
        dump(directory/"plan.json",plan);dump(directory/"action-source.json",phases)
        trainer=Trainer(plan,runner=runner,weights=case["weights"])
        actual=calls.run(trainer,case["inputs"],case["labels"],case["initial"],"gradients","gradients")
        check_result(checks,"initial",actual,expected)
        S=len(case["initial"][0])
        checks["initial_ghost_margin_VJP_zero"]=comparison(
            np.asarray(actual["initial_state_gradients"])[:,-4:],np.zeros((2,4)),exact=True)
        checks["initial_weights_unchanged"]=comparison(actual["state"]["weights"],case["weights"],exact=True)
        base_state=copy.deepcopy(trainer.state)
        states=[]
        for length in range(1,33):
            trainer.state=copy.deepcopy(base_state)
            prefix=calls.run(trainer,[x[:length] for x in case["inputs"]],
                case["labels"],case["initial"],"evaluate",f"prefix-{length:02d}")
            states.append(prefix["final_state"])
        state_array=np.stack(states,axis=1)
        checks["full_physical_state_trace"]=comparison(state_array[:,:,:S-4],expected["states"][:,:,:S-4])
        checks["full_margin_scratch_trace"]=comparison(state_array[:,:,-4:],expected["states"][:,:,-4:])
        # Save all actual prefix states together; per-call evidence retains native results.
        dump(directory/"native-state-trace.json",dict(states=state_array))
        report["positive_margin_spike_check"]=dict(
            hidden_positive_count=int(sum(np.sum(t["mh"]>0) for t in expected["anchors"])),
            output_positive_count=int(sum(np.sum(t["mo"]>0) for t in expected["anchors"])),
            exact_all_spikes_passed=checks["initial_spikes"]["passed"],
            boundary_is_separate=bool(case["boundary"]))
        report["simultaneous_euler_evidence"]=dict(
            old_state_local_temporaries=True,
            all_physical_states_checked_at_all_32_ticks=True,
            phase_order="all old-state drifts, all thresholds, all synapses, all resets",
            passed=checks["full_physical_state_trace"]["passed"])
        report["reset_feed_witness"]=reset_witness(case,plan,Trainer,runner,calls,checks)
        # SGD uses a fresh optimizer state, same initial arrays and same gradients.
        sgd_plan=copy.deepcopy(plan);sgd_plan["optimizer"]["kind"]="sgd"
        sgd=Trainer(sgd_plan,runner=runner,weights=case["weights"])
        result=calls.run(sgd,case["inputs"],case["labels"],case["initial"],"train","sgd-1")
        check_result(checks,"sgd_1",result,expected)
        for bank,g in enumerate(expected["gradients"]):
            checks[f"sgd_1_weights_{bank}"]=comparison(result["state"]["weights"][bank],
                np.array(case["weights"][bank])-.001*g)
        checks["sgd_1_counter"]=dict(passed=result["state"]["step"]==1)
        trainer=Trainer(plan,runner=runner,weights=case["weights"])
        weights=[np.array(w,dtype=np.float64) for w in case["weights"]]
        first=[np.zeros_like(w) for w in weights];second=[np.zeros_like(w) for w in weights]
        for step in range(1,4):
            reference=oracle.forward_vjp(case,weights)
            result=calls.run(trainer,case["inputs"],case["labels"],case["initial"],"train",f"adam-{step}")
            check_result(checks,f"adam_{step}",result,reference)
            for bank,g in enumerate(reference["gradients"]):
                first[bank]=.9*first[bank]+(1.-.9)*g
                second[bank]=.999*second[bank]+(1.-.999)*g*g
                weights[bank]-=.001*(first[bank]/(1.-.9**step))/(
                    np.sqrt(second[bank]/(1.-.999**step))+1e-8)
                for key,ref in (("weights",weights[bank]),("first_moment",first[bank]),("second_moment",second[bank])):
                    checks[f"adam_{step}_{key}_{bank}"]=comparison(result["state"][key][bank],ref)
            checks[f"adam_{step}_counter"]=dict(passed=result["state"]["step"]==step)
        for name in calls.records:
            raw=json.loads((directory/name).read_text())
            checks[f"{name}_native_tape_bytes"]=dict(passed=raw.get("tape_bytes_match",False))
        report["status"]="qualified" if all(c["passed"] for c in checks.values()) else "unqualified"
        report["scope"]="CPU FP64 binary Q0 only; 32-tick full states, exact spikes, CE, both synapse banks, initial physical/scratch VJP, SGD and three Adam. Not formal E5/performance/original migration."
    except Exception as error:
        report.update(status=classify(error),error_type=type(error).__name__,
                      error=str(error),traceback=traceback.format_exc())
    finally:
        dump(directory/"report.json",report)
    return report


def threshold_boundary(directory,runner):
    """Native direct normalized context exactly -2^-20,0,+2^-20."""
    directory.mkdir(parents=True,exist_ok=False)
    reports=[]
    for inclusive in (False,True):
        target=directory/("inclusive" if inclusive else "strict")
        target.mkdir()
        calls=Calls(target)
        report=dict(inclusive=inclusive,status="unqualified",performance_run=False,
                    calls=calls.records,checks={})
        checks=report["checks"]
        try:
            weights=[[0.]]
            Trainer,add,threshold,finish=builder([1,1,3],[1,1],weights,20261004)
            for neuron in range(4):
                add("m=v",dict(v=0,m=1),{},[neuron,4+neuron],neuron,"drift")
            for neuron in range(4):
                threshold(neuron,4+neuron,inclusive=inclusive if neuron else False,margin=True)
            plan,phases=finish(8,[0,1,2,3])
            d=2.**-20
            initial=[[-1.,-d,0.,d,7.,-3.,21.,-11.]]
            expected_spikes=np.array([[[0.,0.,float(inclusive),1.]]])
            output_spikes=expected_spikes[0,0,1:]
            logits=5.*output_spikes
            shifted=logits-logits.max();ex=np.exp(shifted);prob=ex/ex.sum()
            loss=float(np.log(ex.sum())-shifted[0])
            derivative=(prob-np.array([1.,0.,0.]))*5./(1.+5.*np.abs([-d,0.,d]))**2
            expected_initial=np.zeros((1,8));expected_initial[0,1:4]=derivative
            dump(target/"plan.json",plan)
            trainer=Trainer(plan,runner=runner,weights=weights)
            actual=calls.run(trainer,[[[0.]]],[0],initial,"gradients","threshold")
            checks["literal_context_spikes"]=comparison(actual["spikes"],expected_spikes,exact=True)
            checks["logits"]=comparison(actual["logits"],logits[None])
            checks["loss"]=comparison(actual["loss"],loss)
            checks["initial_state_VJP"]=comparison(actual["initial_state_gradients"],expected_initial)
            checks["scratch_initial_VJP_zero"]=comparison(np.asarray(actual["initial_state_gradients"])[:,4:],np.zeros((1,4)),exact=True)
            checks["unused_weight_VJP_zero"]=comparison(actual["gradients"],[[0.]],exact=True)
            raw=json.loads((target/calls.records[0]).read_text())
            checks["native_tape_bytes"]=dict(passed=raw["tape_bytes_match"])
            report["actual_context"]=np.asarray(actual["final_state"])[:,5:]
            report["status"]="qualified" if all(c["passed"] for c in checks.values()) else "unqualified"
        except Exception as error:
            report.update(status=classify(error),error_type=type(error).__name__,
                          error=str(error),traceback=traceback.format_exc())
        finally:
            dump(target/"report.json",report)
            reports.append(report)
    return dict(status="qualified" if all(r["status"]=="qualified" for r in reports) else "unqualified",cases=reports)


def main():
    global np
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id",required=True)
    p.add_argument("--runner",type=Path,default=ROOT/"runtime/arm64-r2/b2-train")
    args=p.parse_args()
    oracle.guard_remote()
    from arm64_artifact_gate_r1 import gate_artifact
    if platform.machine() != 'arm64':raise RuntimeError('ARM64 Python required')
    gate_artifact(args.runner,'runner','1dc6fac642e1829a942105ec6288fe24e5632f66ca54cf12a671258f6014e5a4')
    for relative,expected in FROZEN.items():
        if oracle.sha(ROOT/relative)!=expected:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        p.error("Use an independent simple run id")
    import numpy
    np=numpy
    directory=ROOT/"evidence/e5-runs"/args.run_id
    directory.mkdir(parents=True,exist_ok=False)
    summary=dict(status="started",performance_run=False,cases=[],native_boundary=None,
                 denominator=6,scope="six frozen binary Q0 cases, not full E5",
                 host=platform.node(),python=sys.version,source_hashes={})
    try:
        for source in [Path(__file__),Path(oracle.__file__),oracle.CONTRACT,oracle.FIXTURE_CONTRACT,args.runner]:
            summary["source_hashes"][str(source.relative_to(ROOT))]=oracle.sha(source)
        fixtures=oracle.make_fixtures()
        dump(directory/"fixtures.json",dict(cases=fixtures,performance_run=False))
        if len(fixtures)!=6 or {c["id"] for c in fixtures}!={f"E5-Q0-{m}-{suffix}" for m in oracle.MODELS for suffix in ("base","boundary")}:
            raise RuntimeError("Fixed denominator mismatch")
        for case in fixtures:
            result=qualify_case(case,directory/case["id"],args.runner)
            summary["cases"].append(dict(case=case["id"],status=result["status"],
                report=str((directory/case["id"]/"report.json").relative_to(ROOT))))
        summary["native_boundary"]=threshold_boundary(directory/"direct-threshold-boundary",args.runner)
        summary["qualified_cases"]=sum(c["status"]=="qualified" for c in summary["cases"])
        summary["status"]="qualified_binary_Q0" if summary["qualified_cases"]==6 and summary["native_boundary"]["status"]=="qualified" else "unqualified"
    except Exception as error:
        summary.update(status=classify(error),error_type=type(error).__name__,
                       error=str(error),traceback=traceback.format_exc())
    finally:
        dump(directory/"summary.json",summary)
        print(json.dumps(oracle.plain({k:v for k,v in summary.items() if k!="native_boundary"}),ensure_ascii=False))
    return 0 if summary["status"]=="qualified_binary_Q0" else 1


if __name__=="__main__":
    raise SystemExit(main())
