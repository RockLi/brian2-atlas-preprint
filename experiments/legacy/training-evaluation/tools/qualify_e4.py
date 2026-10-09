#!/usr/bin/env python3
"""E4 adaptive-threshold Q0: independent explicit reverse VJP vs Atlas v5.

No engine tests or engine equations implement the oracle. Native helpers are
imported only in build_plan()/qualify(). prepare/selfcheck need NumPy alone.
All native runs are tiny qualifications, never performance measurements.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"evidence/e4"
ATOL=1e-10
RTOL=1e-8


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dynamics(case,weights):
    if case["mode"]=="fixed_dynamics":return .95,.995,.1,np.zeros(3)
    z=np.asarray(weights[2],dtype=np.float64)
    sigma=1./(1.+np.exp(-z))
    tv=5.+95.*sigma[0];ta=20.+980.*sigma[1];k=2.*sigma[2]
    # Derivatives are wrt the shared raw z bank, not physical tau/k.
    jac=np.array([95.*sigma[0]*(1.-sigma[0])/tv**2,
                  980.*sigma[1]*(1.-sigma[1])/ta**2,2.*sigma[2]*(1.-sigma[2])])
    return 1.-1./tv,1.-1./ta,k,jac


def fixtures():
    seed=20261004;rng=np.random.default_rng(seed)
    inputs=(rng.random((2,32,2))<.45).astype(np.float64)
    weights=[np.array([[.35,.8,.65,.25],[.95,.3,.4,.9]],dtype=np.float64).reshape(-1),
             np.array([[.65,.2],[.45,.7],[.9,-.15],[-.2,.8]],dtype=np.float64).reshape(-1)]
    raw=np.log(np.array([(20.-5.)/(100.-20.),(200.-20.)/(1000.-200.),.1/(2.-.1)]))
    cases=[]
    for mode in ("fixed_dynamics","weights_plus_shared_hidden_tau_v_tau_a_k"):
        banks=[w.copy() for w in weights]+([raw.copy()] if mode!="fixed_dynamics" else [])
        for boundary in (False,True):
            case={"id":f"E4-Q0-{'boundary' if boundary else 'zero'}-{mode}","mode":mode,
                  "sizes":[2,4,2],"B":2,"T":32,"inputs":inputs.copy(),"labels":[0,1],
                  "weights":copy.deepcopy(banks),"seed":seed,"performance_run":False,
                  "input_coverage":"binary count bins only; count>1 remains unqualified on the v5 external trigger ABI",
                  "initial":np.zeros((2,14),dtype=np.float64)}
            # m slots are overwritten before thresholds and must have zero VJP.
            case["initial"][:,8:12]=np.array([[7.,-3.,21.,-11.],[-4.,12.,-8.,33.]])
            if boundary:
                bv,ba,k,_=dynamics(case,banks)
                adaptation=np.array([[0.,0.,0.,.4],[.5,.7,.1,.2]])
                margins=np.array([[0.,1e-7,-1e-7,.2],[-.4,.1,-.1,.3]])
                case["initial"][:,:4]=(1.+ba*adaptation+margins)/bv
                case["initial"][:,4:8]=adaptation
                case["initial"][:,12:]=np.array([[1.,1.+1e-7],[1.-1e-7,1.3]])/.95
                case["intended_initial_hidden_margins"]=margins
                case["boundary_note"]="Report actual float64 drift margins, including nominal zero rounding; no epsilon threshold policy."
            cases.append(plain(case))
    return cases


def forward_vjp(case,weights=None,initial=None,*,anchors=None):
    """Explicit equations and hand-derived reverse, with independent batch mean.

    anchors is used ONLY for a local surrogate-linearized differentiable check.
    Reset gates stay at their anchor hard values; adaptation spikes stay live.
    Ordinary qualification calls use anchors=None and strict hard thresholds.
    """
    x=np.asarray(case["inputs"],dtype=np.float64);B,T,I=x.shape;H=4;O=2
    w=[np.asarray(a,dtype=np.float64) for a in (case["weights"] if weights is None else weights)]
    wi=w[0].reshape(I,H);wo=w[1].reshape(H,O)
    initial=np.array(case["initial"] if initial is None else initial,dtype=np.float64)
    vh=initial[:,:4].copy();a=initial[:,4:8].copy();vo=initial[:,12:].copy()
    bv,ba,k,jac=dynamics(case,w)
    tape=[];spikes=[];states=[]
    for t in range(T):
        oldv=vh.copy();olda=a.copy();oldo=vo.copy()
        uv=bv*oldv;ua=ba*olda;uo=.95*oldo
        # Literal public equation; the adapter's m=uv-ua representation has
        # a disclosed reassociation near floating-point threshold boundaries.
        mh=uv-(1.+ua);mo=uo-1.
        sh=(mh>0.).astype(np.float64);so=(mo>0.).astype(np.float64)
        gateh=sh.copy();gateo=so.copy()
        if anchors is not None:
            q=anchors[t]
            gateh=q["sh"];gateo=q["so"]
            sh=gateh+(mh-q["mh"])/(1.+5.*np.abs(q["mh"]))**2
            so=gateo+(mo-q["mo"])/(1.+5.*np.abs(q["mo"]))**2
        # Explicit source order, matching the common arrays, not BLAS reassociation.
        vh=uv.copy();vo=uo.copy()
        for i in range(I):vh+=x[:,t,i,None]*wi[i]
        for h in range(H):vo+=sh[:,h,None]*wo[h]
        vh-=(1.+ua)*gateh
        a=ua+k*sh
        vo-=gateo
        tape.append(dict(oldv=oldv,olda=olda,oldo=oldo,mh=mh,mo=mo,sh=sh,so=so,gateh=gateh))
        spikes.append(np.concatenate([sh,so],axis=1))
        states.append(np.concatenate([vh,a,uv-ua,vo],axis=1))
    spike_array=np.stack(spikes,axis=1)
    logits=5.*spike_array[:,:,-O:].mean(axis=1)
    shifted=logits-logits.max(axis=1,keepdims=True);exponent=np.exp(shifted)
    probabilities=exponent/exponent.sum(axis=1,keepdims=True)
    labels=np.asarray(case["labels"],dtype=np.int64)
    loss=np.mean(np.log(exponent.sum(axis=1))-shifted[np.arange(B),labels])
    dlogits=probabilities.copy();dlogits[np.arange(B),labels]-=1.;dlogits/=B
    dv=np.zeros((B,H));da=np.zeros((B,H));dout=np.zeros((B,O))
    dwi=np.zeros_like(wi);dwo=np.zeros_like(wo);dbv=dba=dk=0.
    for t in range(T-1,-1,-1):
        q=tape[t]
        # Synapse weights see the final-state carries; reset gates are detached.
        dwi+=x[:,t,:].T@dv
        dwo+=q["sh"].T@dout
        dsh=dout@wo.T+k*da
        dso=(5./T)*dlogits
        ph=1./(1.+5.*np.abs(q["mh"]))**2
        po=1./(1.+5.*np.abs(q["mo"]))**2
        duv=dv+ph*dsh
        dua=da-q["gateh"]*dv-ph*dsh
        duo=dout+po*dso
        dbv+=np.sum(duv*q["oldv"])
        dba+=np.sum(dua*q["olda"])
        dk+=np.sum(da*q["sh"])
        dv=bv*duv;da=ba*dua;dout=.95*duo
    gradients=[dwi.reshape(-1),dwo.reshape(-1)]
    if case["mode"]!="fixed_dynamics":gradients.append(np.array([dbv,dba,dk])*jac)
    initial_vjp=np.concatenate([dv,da,np.zeros((B,H)),dout],axis=1)
    return {"loss":float(loss),"logits":logits,"spikes":spike_array,
            "states":np.stack(states,axis=1),"final_state":states[-1],
            "initial_state_gradients":initial_vjp,"gradients":gradients,
            "physical_dynamics":{"beta_v":bv,"beta_a":ba,"k":k},"anchors":tape}


def selfcheck(cases):
    """Check hand reverse against FD of an explicit local surrogate linearization.

    This does NOT differentiate the discontinuous hard-spike function.
    The manual VJP above remains the qualification oracle.
    """
    results=[]
    for case in cases:
        base=forward_vjp(case);anchors=base["anchors"];checks=[]
        for bank,values in enumerate(case["weights"]):
            for index,value in enumerate(values):
                h=1e-6;plus=copy.deepcopy(case["weights"]);minus=copy.deepcopy(case["weights"])
                plus[bank][index]+=h;minus[bank][index]-=h
                fd=(forward_vjp(case,plus,anchors=anchors)["loss"]-forward_vjp(case,minus,anchors=anchors)["loss"])/(2*h)
                expected=base["gradients"][bank][index]
                checks.append({"field":f"bank{bank}[{index}]","fd":fd,"manual":expected,
                               "passed":abs(fd-expected)<=2e-7+2e-5*abs(expected)})
        for batch in range(2):
            for index in range(14):
                h=1e-6;plus=np.asarray(case["initial"]).copy();minus=plus.copy()
                plus[batch,index]+=h;minus[batch,index]-=h
                fd=(forward_vjp(case,initial=plus,anchors=anchors)["loss"]-forward_vjp(case,initial=minus,anchors=anchors)["loss"])/(2*h)
                expected=base["initial_state_gradients"][batch,index]
                checks.append({"field":f"initial[{batch},{index}]","fd":fd,"manual":expected,
                               "passed":abs(fd-expected)<=2e-7+2e-5*abs(expected)})
        results.append({"case":case["id"],"passed":all(x["passed"] for x in checks),"checks":checks,
                        "first_hidden_margin":anchors[0]["mh"],
                        "initial_margin_vjp_exact_zero":bool(np.all(base["initial_state_gradients"][:,8:12]==0))})
    return {"method":"central differences of explicit anchor-linearized surrogate graph; NOT hard-spike finite differences",
            "native_execution":False,"driver_sha256":sha(__file__),"numpy_version":np.__version__,"results":results}


def load_native():
    sys.path.insert(0,str(ROOT/"snapshot"))
    sys.path.insert(0,str(ROOT/"snapshot/brian2-rust/python"))
    from brian2_rust.training import lif_training_plan,NativeLIFTrainer
    from brian2_rust.training_dynamic import compile_dynamic_transform,dynamic_action
    from brian2_rust.training_equations import neuron_parameter_bank
    assert Path(sys.modules["brian2_rust.training"].__file__).resolve()==ROOT/"snapshot/brian2-rust/python/brian2_rust/training.py"
    return lif_training_plan,NativeLIFTrainer,compile_dynamic_transform,dynamic_action,neuron_parameter_bank


def build_plan(case):
    lif,_,compile_transform,make_action,parameter_bank=load_native()
    banks=case["weights"];identity=[[[{"op":"state","index":0}]]]*2
    plan=lif([2,4,2],backend="cpu",beta=.95,threshold=1.,detach_reset=True,
             projections=[parameter_bank(len(w)) for w in banks],
             state_equations=identity,state_resets=identity,
             clock={"origin":0.,"dt":.001},learning_rate=.001,seed=case["seed"],
             max_tape_bytes=1024**3)
    programs=[];actions=[]
    def add(code,states,parameters,reads,owner,trigger=None,detach=False):
        compiled=compile_transform(code,states=states,parameters=parameters)
        number=len(programs);programs.append(compiled["programs"])
        actions.append(make_action(compiled,reads,owner=owner,program_set=number,
                                   trigger=trigger,detach_trigger=detach))
    trainable=case["mode"]!="fixed_dynamics"
    params={"zv":(2,0),"za":(2,1)} if trainable else {"bv":.95,"ba":.995}
    drift=("v=(1-1/(5+95/(1+exp(-zv))))*v\na=(1-1/(20+980/(1+exp(-za))))*a\nm=v-a"
           if trainable else "v=bv*v\na=ba*a\nm=v-a")
    for h in range(4):add(drift,{"v":0,"a":1,"m":2},params,[h,4+h,8+h],h)
    for j in range(2):add("v=.95*v",{"v":0},{},[12+j],4+j)
    for neuron,slot in enumerate([8,9,10,11,12,13]):
        actions.append({"owner":neuron,"reads":[slot],"writes":[],"program_set":None,
                        "threshold":neuron,"trigger":None})
    for i in range(2):
        for h in range(4):add("v+=w",{"v":0},{"w":(0,i*4+h)},[h],h,{"external":True,"index":i})
    for h in range(4):
        for j in range(2):add("v+=w",{"v":0},{"w":(1,h*2+j)},[12+j],4+j,{"external":False,"index":h})
    for h in range(4):add("v-=1+a",{"v":0,"a":1},{},[h,4+h],h,{"external":False,"index":h},True)
    for h in range(4):
        add("a+=2/(1+exp(-zk))" if trainable else "a+=k",{"a":0},
            {"zk":(2,2)} if trainable else {"k":.1},[4+h],h,{"external":False,"index":h},False)
    for j in range(2):add("v-=1",{"v":0},{},[12+j],4+j,{"external":False,"index":4+j},True)
    plan.update(schema="b2-dynamic-training-plan-v5",dynamic={
        "initial":[0.]*14,"initial_parameters":[None]*14,"detached":[False]*14,
        "voltage":[8,9,10,11,12,13],"program_sets":programs,"actions":actions})
    return plan


def admission(plan,state,inputs,labels,initial,operation):
    """Exact guard formula for these small CPU plans; reject unsupported extensions.

    Mirrors only storage accounting in frozen training.rs/dynamic.rs, not math.
    No MPI, indirect addressing, dynamic clocks, delay/migration or maps allowed.
    """
    d=plan["dynamic"]
    assert not plan.get("mpi_ranks") and plan["backend"]=="cpu" and plan["sizes"]==[2,4,2]
    assert not any(d.get(k) for k in ("delay_layout","migration","parameter_maps","threshold_references","clocks","integer_parameters"))
    assert all(not a.get("indirect") for a in d["actions"])
    B=len(inputs);T=len(inputs[0]);I=2;N=6;O=2;S=len(d["initial"]);P=sum(len(r) for r in plan["masks"])
    topology=sum(len(p["sources"])*24+128 for p in plan["projections"])
    reads=sum(len(a["reads"]) for a in d["actions"])
    nodes=sum(len(program) for program_set in d["program_sets"] for program in program_set)
    components={"spike_tape":B*T*N*32,"live_neurons":B*N*32,"logits":B*O*16,
                "parameter_optimizer":P*48,"topology":topology,
                "v4_vector":B*T*S*16+B*S*64+sum(len(p) for p in plan["state_equations"])*128*128+8192,
                "inputs":B*T*(I*8+48)+B*8,
                "dynamic":reads*B*T*8+reads*32+len(d["actions"])*256+nodes*128+S*B*64+129*3*8}
    tape=sum(components.values());initial_bytes=S*B*64+P*48
    request={"plan":plan,"state":state,"operation":operation,"inputs":plain(inputs),
             "labels":plain(labels),"initial":plain(initial),"start_tick":0}
    wire=json.dumps(request,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    reasons=[]
    if len(wire)>64*1024**2:reasons.append("request_json_exceeds_64MiB")
    if max(tape,initial_bytes,P*48+topology)>plan["max_tape_bytes"]:reasons.append("native_software_memory_budget")
    return {"status":"budget_rejected" if reasons else "admitted","reasons":reasons,
            "request_json_bytes":len(wire),"request_sha256":hashlib.sha256(wire).hexdigest(),
            "exact_tape_bytes":tape,"initial_bytes":initial_bytes,"components":components,
            "scope":"software admission, not peak RSS"}


def compare(actual,expected,exact=False):
    a=np.asarray(actual);e=np.asarray(expected)
    if a.shape!=e.shape:return {"passed":False,"actual_shape":list(a.shape),"expected_shape":list(e.shape)}
    return {"passed":bool(np.array_equal(a,e) if exact else np.all(np.isfinite(a)) and np.all(np.abs(a-e)<=ATOL+RTOL*np.abs(e))),
            "max_absolute_error":float(np.max(np.abs(a-e),initial=0)),"shape":list(e.shape)}


def qualify(case,directory,runner):
    _,Trainer,_,_,_=load_native();directory.mkdir(parents=True,exist_ok=False)
    plan=build_plan(case);save(directory/"plan.json",plan);save(directory/"fixture.json",case)
    trainer=Trainer(plan,weights=case["weights"],runner=runner)
    index=0;calls=[]
    def call(operation,inputs=None):
        nonlocal index
        index+=1;data=case["inputs"] if inputs is None else inputs
        check=admission(trainer.plan,trainer.state,data,case["labels"],case["initial"],operation)
        record={"operation":operation,"admission":check,"status":check["status"],"performance_run":False}
        begin=time.perf_counter_ns()
        try:
            if check["status"]!="admitted":raise ValueError("software admission rejected")
            result=trainer.execute(data,case["labels"],operation=operation,initial=case["initial"])
            record.update(status="executed",result=result,public_api_ns=time.perf_counter_ns()-begin,
                          tape_bytes_match=result["tape_bytes"]==check["exact_tape_bytes"])
        except Exception as error:
            msg=str(error).lower()
            status=("oom" if isinstance(error,MemoryError) else "timeout" if isinstance(error,subprocess.TimeoutExpired)
                    else "numerical_divergence" if "nonfinite" in msg or "non-finite" in msg
                    else "budget_rejected" if "budget" in msg else "software_rejected")
            record.update(status=status,
                          error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                          public_api_ns=time.perf_counter_ns()-begin)
        path=directory/f"call-{index:03d}-{operation}.json";save(path,record);calls.append(path.name)
        return record
    report={"case":case["id"],"performance_run":False,"status":"unqualified","calls":calls,"checks":{}}
    try:
        expected=forward_vjp(case);actual=call("gradients")
        if actual["status"]!="executed":
            report["status"]=actual["status"];return report
        checks=report["checks"];result=actual["result"]
        checks["tape_bytes"]={"passed":actual["tape_bytes_match"]}
        for key in ("loss","logits","spikes","final_state","initial_state_gradients"):
            checks[key]=compare(result[key],expected[key],exact=key=="spikes")
        checks["initial_margin_cotangent_zero"]=compare(np.asarray(result["initial_state_gradients"])[:,8:12],np.zeros((2,4)),exact=True)
        for b,gradient in enumerate(expected["gradients"]):checks[f"parameter_vjp_{b}"]=compare(result["gradients"][b],gradient)
        # Real v/a live slots, not final_membrane (which includes diagnostic m).
        state=[];base_state=copy.deepcopy(trainer.state)
        for length in range(1,33):
            trainer.state=copy.deepcopy(base_state)
            prefix=call("evaluate",[sample[:length] for sample in case["inputs"]])
            if prefix["status"]!="executed":raise RuntimeError(f"prefix {length}: {prefix['status']}")
            state.append(prefix["result"]["final_state"])
        state=np.stack(state,axis=1)
        checks["full_state_trace"]=compare(state,expected["states"])
        checks["physical_v_a_trace"]=compare(state[:,:,:8],expected["states"][:,:,:8])
        trainer.state=copy.deepcopy(base_state)
        weights=[np.asarray(w,dtype=np.float64) for w in case["weights"]]
        m=[np.zeros_like(w) for w in weights];v=copy.deepcopy(m)
        for step in range(1,4):
            reference=forward_vjp(case,weights);update=call("train")
            if update["status"]!="executed":raise RuntimeError(f"Adam {step}: {update['status']}")
            checks[f"adam_{step}_loss"]=compare(update["result"]["loss"],reference["loss"])
            for b,g in enumerate(reference["gradients"]):
                checks[f"adam_{step}_gradient_{b}"]=compare(update["result"]["gradients"][b],g)
                m[b]=.9*m[b]+.1*g;v[b]=.999*v[b]+.001*g*g
                weights[b]-=.001*(m[b]/(1.-.9**step))/(np.sqrt(v[b]/(1.-.999**step))+1e-8)
                for key,target in (("weights",weights[b]),("first_moment",m[b]),("second_moment",v[b])):
                    checks[f"adam_{step}_{key}_{b}"]=compare(update["result"]["state"][key][b],target)
            checks[f"adam_{step}_counter"]={"passed":update["result"]["state"]["step"]==step}
        report["status"]="qualified" if all(v["passed"] for v in checks.values()) else "unqualified"
        report["qualification_scope"]="Binary inputs; explicit E4 v/a/m representation; CE surrogate VJP; shared raw z gradients; initial-state VJP; 3 Adam updates; CPU FP64 only."
    except Exception as error:
        report.update(status="qualification_error",exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:save(directory/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","selfcheck","qualify"])
    parser.add_argument("--run-id",default="r1")
    parser.add_argument("--allow-host")
    parser.add_argument("--runner",type=Path,default=ROOT/"runtime/b2-train")
    args=parser.parse_args();cases=fixtures()
    if args.mode=="prepare":
        target=OUT/"fixtures-r1.json"
        if target.exists():raise FileExistsError(target)
        save(target,{"schema":"E4-Q0-fixtures-r1","cases":cases,"source_contract_sha256":sha(ROOT/"protocol/semantic-contracts.json")})
    elif args.mode=="selfcheck":
        target=OUT/f"oracle-selfcheck-{args.run_id}.json"
        if target.exists():raise FileExistsError(target)
        stored=OUT/"fixtures-r1.json"
        if not stored.exists():raise SystemExit("Run prepare once before the oracle selfcheck.")
        checked=selfcheck(json.loads(stored.read_text())["cases"])
        checked["fixture_sha256"]=sha(stored)
        save(target,checked)
    else:
        if not args.allow_host or args.allow_host!=platform.node():raise SystemExit("Specify the coordinated target hostname; no native execution on the MacBook Air.")
        stored=OUT/"fixtures-r1.json"
        if not stored.exists():raise SystemExit("Run prepare once, then reuse the stored common arrays.")
        frozen=json.loads(stored.read_text())
        assert frozen["source_contract_sha256"]==sha(ROOT/"protocol/semantic-contracts.json")
        check_path=OUT/f"oracle-selfcheck-{args.run_id}.json"
        if not check_path.exists():raise SystemExit("Run the independent oracle selfcheck before native qualification.")
        independent=json.loads(check_path.read_text())
        if (independent["driver_sha256"]!=sha(__file__) or independent.get("fixture_sha256")!=sha(stored)
            or len(independent["results"])!=4 or not all(r["passed"] for r in independent["results"])):
            raise SystemExit("Oracle selfcheck failed or belongs to a different driver version.")
        cases=frozen["cases"]
        folder=OUT/"runs"/args.run_id;folder.mkdir(parents=True,exist_ok=False)
        report={"performance_run":False,"source_contract_sha256":frozen["source_contract_sha256"],
                "fixture_sha256":sha(stored),"driver_sha256":sha(__file__),"runtime_sha256":sha(args.runner),
                "host":platform.node(),"python":sys.version,"numpy_version":np.__version__,
                "oracle_selfcheck_sha256":sha(check_path),
                "source_sha256":{path:sha(ROOT/"snapshot/brian2-rust"/path) for path in
                    ("python/brian2_rust/training.py","python/brian2_rust/training_dynamic.py",
                     "python/brian2_rust/training_equations.py","src/training.rs","src/training/dynamic.rs")},
                "oracle":"independent NumPy forward + hand-derived reverse VJP; no engine tests imported",
                "results":[]}
        for case in cases:
            try:result=qualify(case,folder/case["id"],args.runner)
            except Exception as error:
                result={"case":case["id"],"status":"dependency_failed" if isinstance(error,(ImportError,ModuleNotFoundError)) else "preparation_failed",
                        "performance_run":False,"error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()}
                save(folder/case["id"]/"report.json",result)
            report["results"].append(result)
            save(folder/"summary.json",report)
        print(json.dumps([{"case":r["case"],"status":r["status"]} for r in report["results"]]))


if __name__=="__main__":main()
