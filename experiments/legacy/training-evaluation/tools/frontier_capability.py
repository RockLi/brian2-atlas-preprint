#!/usr/bin/env python3
"""F-HH1 capability probes; deliberately not an engine performance benchmark.

No installation is performed by this script. Run each engine from its own
hash-locked interpreter. The `contract` mode imports no numerical packages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/frontier"
VERSIONS = {"jaxley": "0.14.0", "braincell": "0.1.0", "brian2modelfitting": "0.4"}
LOCKS = {"jaxley": "jaxley-py312-macos-arm64.lock", "braincell": "braincell-py312-macos-arm64.lock",
         "brian2modelfitting": "brian2modelfitting-published-py312.lock"}
CONTRACT = {
    "id": "F-HH1-r1", "performance_run": False,
    "scope": "Predeclared standalone single-compartment HH capability probe; not any member of the 12-model Brian migration corpus.",
    "status_at_freeze": "prepared_not_executed",
    "model": {
        "name": "classical HH1952 point membrane in absolute-voltage coordinates",
        "states": ["v_mV", "m", "h", "n"], "reset": None, "surrogate": None,
        "voltage_ode": "C*dv/dt = gNa*m^3*h*(50-v) + 36*n^4*(-77-v) + 0.3*(-54.387-v) + I",
        "gate_ode": "dx/dt = alpha_x(v)*(1-x)-beta_x(v)*x",
        "alpha_m_per_ms": "0.1*(v+40)/(1-exp(-(v+40)/10))",
        "beta_m_per_ms": "4*exp(-(v+65)/18)",
        "alpha_h_per_ms": "0.07*exp(-(v+65)/20)",
        "beta_h_per_ms": "1/(1+exp(-(v+35)/10))",
        "alpha_n_per_ms": "0.01*(v+55)/(1-exp(-(v+55)/10))",
        "beta_n_per_ms": "0.125*exp(-(v+65)/80)",
        "capacitance_uF_per_cm2": 1.0, "initial_voltage_mV": -65.0,
        "initial_gates": "alpha/(alpha+beta) at -65 mV, materialized once in shared fixture",
        "geometry_for_point_current_conversion_um": {"radius": 10.0, "length": 10.0, "area": "2*pi*radius*length"},
        "temperature_kinetic_scale": 1.0,
    },
    "input": {"dt_ms": 0.025, "steps": 800, "duration_ms": 20.0,
              "current_uA_per_cm2": "10 for 5 <= t_ms < 15, otherwise 0",
              "recording": "start-of-step voltage: t = 0,0.025,...,19.975 ms", "dtype": "float64"},
    "target": {"gNa_mS_per_cm2": 120.0,
               "source": "independent NumPy float64 RK4 oracle; 10 substeps per output step, piecewise-constant shared current",
               "artifact": "fixtures/F-HH1-r1.json, SHA256 recorded in every probe result",
               "loss": "mean((voltage_mV-target_mV)^2), all 800 points"},
    "fit": {"trainable": ["gNa_mS_per_cm2"], "initial": 100.0, "bounds": [80.0, 160.0],
            "jax_ad_update": "one SGD update: g -= 0.1*clip(dloss/dg,-1,1)",
            "brian2modelfitting_update": "SkoptOptimizer(random_state=20261004), n_samples=2, one fit round; then refine(calc_gradient=True,max_nfev=2)",
            "quality_goal": "none: interface/finite-value capability only, no time-to-quality claim"},
    "numerical_profiles": {
        "jaxley": "Jaxley HH; native exponential gate update and fwd_euler voltage solver; all physical parameters explicit; initial record retained; x64",
        "braincell": "SingleCompartment + native Na_HH1952/K_HH1952/IL, rk4; explicit gK=36, gLeak=0.3, V_sh=-45, q10=1; sodium current multiplier exposed as ParamState; x64",
        "brian2modelfitting": "published package's TraceFitter, explicit Brian HH equations, rk4, float64 NumPy runtime; default package dependencies in private environment",
        "cross_engine_equality": "Not asserted: native integration profiles differ; each trace and discrepancy to common RK4 target are reported.",
    },
    "checks": {
        "forward": "800 finite start-of-step values; report target RMSE and max absolute error, no same-semantics pass implied",
        "autodiff": "For JAX engines, compare scalar continuous-HH loss gradient with central differences h=0.001 and 0.0005 mS/cm2; atol=1e-6, rtol=1e-3",
        "finite_difference_scope": "Continuous voltage loss only. No hard-spike finite difference or surrogate claim.",
        "fit_interface": "Persist actual API result and finite parameter/trace checks; refinement separately classified",
    },
    "budgets": {"timeout_seconds_per_engine": 600, "cpu_threads": 1, "gpu": False,
                "memory_limit_bytes": None, "memory_note": "No OS-enforced memory cap; tiny one-cell case, peak resource accounting may be added by coordinator."},
    "not_executed_scope": ["12 original Brian models", "spatial cable fitting", "multi-cell biological networks", "deep SNNs", "GPU/multi-GPU", "converged fitting", "formal engine performance ranking"],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temp.replace(path)


def write_contract():
    save(OUT / "F-HH1-contract.json", CONTRACT)
    entries = []
    for name, version in VERSIONS.items():
        path = OUT / "official-sources" / f"{name}-{version}-pypi.json"
        metadata = json.loads(path.read_text())
        artifacts = []
        for artifact in metadata["urls"]:
            local = path.parent / artifact["filename"]
            if local.exists():
                assert sha(local) == artifact["digests"]["sha256"]
                artifacts.append({"filename": artifact["filename"], "url": artifact["url"],
                                  "sha256": sha(local), "verified_against_pypi": True})
        extracted = path.parent / f"{name}-{version}"
        files = {str(p.relative_to(extracted)): sha(p) for p in sorted(extracted.rglob("*")) if p.is_file()}
        entries.append({"name": name, "version": version,
                        "metadata_url": f"https://pypi.org/pypi/{name}/{version}/json",
                        "metadata_sha256": sha(path), "artifacts": artifacts,
                        "extracted_file_sha256": files})
    save(OUT / "source-manifest.json", {"download_date": "2026-10-04", "packages": entries})
    locks = {p.name: sha(p) for p in sorted((OUT / "dependencies").glob("*.lock"))}
    save(OUT / "preparation-status.json", {
        "performance_run": False, "status": "prepared_not_executed", "engine_count": 3,
        "official_source_archives_verified": True, "source_api_inspection_completed": True,
        "dependency_lock_sha256": locks, "resolved_for": "CPython 3.12 / macOS arm64",
        "installation_tested": False, "numerical_probe_executed": False,
        "potential_compatibility_issue": "brian2modelfitting 0.4 imports nevergrad.instrumentation and pins nevergrad<=0.3; the resolved modern dependency set is not an import/compatibility qualification.",
        "driver_sha256": sha(__file__), "contract_sha256": sha(OUT / "F-HH1-contract.json"),
    })


def fixture():
    """Generate once on the test host, then every engine reads identical arrays."""
    import numpy as np
    path = OUT / "fixtures/F-HH1-r1.json"
    if path.exists():
        obj = json.loads(path.read_text())
        assert obj["contract_sha256"] == sha(OUT / "F-HH1-contract.json")
        return obj, path

    def ratio(x):
        return 1.0 - x/2.0 + x*x/12.0 if abs(x) < 1e-7 else x/np.expm1(x)

    def rates(v):
        return (ratio(-(v+40)/10), 4*np.exp(-(v+65)/18),
                .07*np.exp(-(v+65)/20), 1/(1+np.exp(-(v+35)/10)),
                .1*ratio(-(v+55)/10), .125*np.exp(-(v+65)/80))

    a,b,c,d,e,f = rates(-65.)
    initial = [-65., a/(a+b), c/(c+d), e/(e+f)]
    dt = CONTRACT["input"]["dt_ms"]
    times = np.arange(CONTRACT["input"]["steps"], dtype=np.float64)*dt
    current = np.where((times >= 5) & (times < 15), 10., 0.)

    def derivative(state, amperage):
        v,m,h,n = state
        am,bm,ah,bh,an,bn = rates(v)
        return np.array([120*m**3*h*(50-v)+36*n**4*(-77-v)+.3*(-54.387-v)+amperage,
                         am*(1-m)-bm*m, ah*(1-h)-bh*h, an*(1-n)-bn*n], dtype=np.float64)

    def solve(substeps):
        state = np.array(initial, dtype=np.float64)
        trace = []
        small_dt = dt/substeps
        for amperage in current:
            trace.append(float(state[0]))
            for _ in range(substeps):
                k1=derivative(state,amperage);k2=derivative(state+small_dt*k1/2,amperage)
                k3=derivative(state+small_dt*k2/2,amperage);k4=derivative(state+small_dt*k3,amperage)
                state += small_dt*(k1+2*k2+2*k3+k4)/6
        return np.asarray(trace)

    target = solve(10)
    refinement = solve(20)
    error = float(np.max(np.abs(target-refinement)))
    assert error < 1e-5, f"Oracle refinement inconsistency: {error}"
    obj = {"contract_sha256": sha(OUT / "F-HH1-contract.json"), "dtype": "float64",
           "times_ms": times.tolist(), "current_uA_per_cm2": current.tolist(),
           "initial": initial, "target_voltage_mV": target.tolist(),
           "oracle_refinement_max_abs_mV": error, "performance_run": False}
    save(path, obj)
    return obj, path


def forward_stats(trace, data, filename):
    import numpy as np
    voltage = np.asarray(trace, dtype=np.float64).reshape(-1)
    target = np.asarray(data["target_voltage_mV"], dtype=np.float64)
    assert voltage.shape == target.shape, (voltage.shape, target.shape)
    assert np.all(np.isfinite(voltage)), "nonfinite voltage"
    np.save(filename, voltage)
    return {"status": "finite_forward_pass", "samples": len(voltage),
            "dtype": str(voltage.dtype), "voltage_file": Path(filename).name,
            "sha256": sha(filename), "target_rmse_mV": float(np.sqrt(np.mean((voltage-target)**2))),
            "target_max_abs_mV": float(np.max(np.abs(voltage-target))),
            "same_semantics_qualification": False}


def finish_ad(loss, gradient, initial_loss, report):
    import numpy as np
    g = float(gradient)
    l0 = float(initial_loss)
    assert np.isfinite(g) and np.isfinite(l0)
    estimates = {str(h): (float(loss(100+h))-float(loss(100-h)))/(2*h) for h in (1e-3,5e-4)}
    qualified = all(abs(g-v) <= 1e-6 + 1e-3*abs(v) for v in estimates.values())
    new_parameter = 100.0 - 0.1*np.clip(g, -1., 1.)
    l1 = float(loss(float(new_parameter)))
    assert np.isfinite(l1)
    report["gradient"] = {"status": "continuous_gradient_qualified" if qualified else "gradient_mismatch",
                          "autodiff": g, "central_difference": estimates, "atol": 1e-6, "rtol": 1e-3}
    report["optimizer"] = {"status": "finite_one_step_pass", "parameter_before": 100.,
                           "parameter_after": float(new_parameter), "loss_before": l0, "loss_after": l1,
                           "loss_decreased": l1 < l0, "convergence_claim": False}


def jaxley_probe(data, report, publish, directory):
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jaxley as jx
    from jaxley.channels import HH
    cell = jx.Compartment()
    cell.insert(HH())
    for key,value in {"radius":10., "length":10., "capacitance":1., "v":-65.,
                      "HH_gNa":.1, "HH_gK":.036, "HH_gLeak":.0003,
                      "HH_eNa":50., "HH_eK":-77., "HH_eLeak":-54.387,
                      "HH_m":data["initial"][1], "HH_h":data["initial"][2], "HH_n":data["initial"][3]}.items():
        cell.set(key,value)
    # Density (uA/cm2) * 2*pi*r*l (um2) * 1e-8 cm2/um2 * 1000 nA/uA.
    current_nA = np.asarray(data["current_uA_per_cm2"])*(2*np.pi*10*10)*1e-5
    cell.stimulate(jnp.asarray(current_nA), verbose=False)
    cell.record("v", verbose=False)
    cell.make_trainable("HH_gNa", verbose=False)
    target = jnp.asarray(data["target_voltage_mV"], dtype=jnp.float64)
    def forward(conductance):
        params = [{"HH_gNa": jnp.asarray([conductance/1000.], dtype=jnp.float64)}]
        voltage = jx.integrate(cell, params=params, delta_t=.025, solver="fwd_euler")
        # Jaxley records the initial value as the first sample.
        return voltage[0,:len(target)]
    def loss(conductance):
        return jnp.mean((forward(conductance)-target)**2)
    report["forward"] = forward_stats(forward(100.), data, directory/"voltage.npy")
    report["devices"] = [str(x) for x in jax.devices()]
    publish()
    value,gradient = jax.jit(jax.value_and_grad(loss))(100.)
    finish_ad(jax.jit(loss),gradient,value,report)


def braincell_probe(data, report, publish, directory):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import braincell as bc
    import brainstate as bst
    import brainunit as u
    import braintools as bt
    bst.environ.set(precision=64)
    # Only expose the native sodium conductance as a differentiable state.
    # Native p/q gates, currents, temperature handling, cell and integrator remain.
    class TrainableNa(bc.channel.Na_HH1952):
        def __init__(self):
            super().__init__(1, g_max=1.*u.mS/u.cm**2, V_sh=-45.*u.mV, q10=1.)
            self.gna = bst.ParamState(jnp.asarray(100.,dtype=jnp.float64))
        def current(self,V,Na):
            return self.gna.value*super().current(V,Na)
    cell = bc.SingleCompartment(1,length=10.*u.um,radius=10.*u.um,C=1.*u.uF/u.cm**2,
                                 V_initializer=bt.init.Constant(-65.*u.mV),solver="rk4")
    cell.na = bc.ion.SodiumFixed(1,E=50.*u.mV)
    sodium=TrainableNa();cell.na.add(INa=sodium)
    cell.k = bc.ion.PotassiumFixed(1,E=-77.*u.mV)
    potassium=bc.channel.K_HH1952(1,g_max=36.*u.mS/u.cm**2,V_sh=-45.*u.mV,q10=1.)
    cell.k.add(IK=potassium)
    cell.IL=bc.channel.IL(1,g_max=.3*u.mS/u.cm**2,E=-54.387*u.mV)
    cell.init_state()
    current=jnp.asarray(data["current_uA_per_cm2"],dtype=jnp.float64)
    target=jnp.asarray(data["target_voltage_mV"],dtype=jnp.float64)
    def forward():
        cell.reset_state()
        cell.V.value=jnp.asarray([-65.])*u.mV
        sodium.p.value=jnp.asarray([data["initial"][1]])
        sodium.q.value=jnp.asarray([data["initial"][2]])
        potassium.p.value=jnp.asarray([data["initial"][3]])
        def step(index,amperage):
            voltage=cell.V.value.to_decimal(u.mV)[0]
            with bst.environ.context(t=index*.025*u.ms,dt=.025*u.ms):
                cell.update(amperage*u.uA/u.cm**2)
            return voltage
        return bst.transform.for_loop(step,jnp.arange(len(current)),current)
    def loss():
        return jnp.mean((forward()-target)**2)
    report["forward"]=forward_stats(forward(),data,directory/"voltage.npy")
    report["devices"]=[str(x) for x in jax.devices()]
    report["adapter_note"]="Native sodium current multiplied by explicit scalar ParamState; no channel or integrator equations reimplemented."
    publish()
    gradient,value=bst.transform.grad(loss,grad_states={"gNa":sodium.gna},return_value=True)()
    def evaluate(g):
        sodium.gna.value=jnp.asarray(g,dtype=jnp.float64)
        return loss()
    finish_ad(evaluate,gradient["gNa"],value,report)


BRIAN_EQUATIONS = """
dv/dt = (gNa*m**3*h*(50*mV-v) + 36*msiemens/cm**2*n**4*(-77*mV-v) + 0.3*msiemens/cm**2*(-54.387*mV-v) + I)/(1*ufarad/cm**2) : volt
dm/dt = alpham*(1-m)-betam*m : 1
dh/dt = alphah*(1-h)-betah*h : 1
dn/dt = alphan*(1-n)-betan*n : 1
alpham = 1/exprel(-(v+40*mV)/(10*mV))/ms : Hz
betam = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
alphah = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
betah = 1/(1+exp(-(v+35*mV)/(10*mV)))/ms : Hz
alphan = 0.1/exprel(-(v+55*mV)/(10*mV))/ms : Hz
betan = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
gNa : siemens/metre**2 (constant)
"""


def fitting_probe(data, report, publish, directory):
    import numpy as np
    import brian2 as b2
    from brian2modelfitting import TraceFitter, SkoptOptimizer, MSEMetric
    b2.start_scope();b2.seed(20261004);b2.prefs.codegen.target="numpy"
    conductance_unit=b2.msiemens/b2.cm**2
    fitter=TraceFitter(model=BRIAN_EQUATIONS,input_var="I",output_var="v",
                      input=np.asarray(data["current_uA_per_cm2"])[None,:]*b2.uamp/b2.cm**2,
                      output=np.asarray(data["target_voltage_mV"])[None,:]*b2.mV,
                      dt=.025*b2.ms,n_samples=2,method="rk4",
                      param_init={"v":-65*b2.mV,"m":data["initial"][1],"h":data["initial"][2],"n":data["initial"][3]})
    # Public documented n_rounds=0 establishes the metric and parameter bounds.
    fitter.fit(optimizer=SkoptOptimizer(random_state=20261004),metric=MSEMetric(),n_rounds=0,
               callback=None,gNa=[80*conductance_unit,160*conductance_unit])
    trace=fitter.generate_traces(params={"gNa":100*conductance_unit})/b2.mV
    report["forward"]=forward_stats(trace,data,directory/"voltage.npy");publish()
    params,error=fitter.fit(optimizer=SkoptOptimizer(random_state=20261004),metric=MSEMetric(),
                            n_rounds=1,callback=None,gNa=[80*conductance_unit,160*conductance_unit])
    estimate=float(params["gNa"]/conductance_unit)
    assert np.isfinite(estimate) and np.isfinite(float(error))
    report["fit"]={"status":"finite_fit_round_pass","n_samples":2,"n_rounds":1,
                   "gNa_mS_per_cm2":estimate,"error_in_SI_voltage_squared":float(error)}
    publish()
    try:
        refined,information=fitter.refine(params={"gNa":100*conductance_unit},calc_gradient=True,
                                         max_nfev=2,callback=None)
        value=float(refined["gNa"]/conductance_unit)
        assert np.isfinite(value)
        report["refine"]={"status":"analytic_sensitivity_interface_pass","gNa_mS_per_cm2":value,
                          "nfev":getattr(information,"nfev",None),"success":getattr(information,"success",None),
                          "gradient_independently_qualified":False,"convergence_claim":False}
    except Exception as error:
        report["refine"]={"status":"refine_probe_failed","exception_type":type(error).__name__,
                          "message":str(error),"traceback":traceback.format_exc()}


def worker(args):
    directory=Path(args.directory)
    report={"engine":args.engine,"version_expected":VERSIONS[args.engine],"performance_run":False,
            "model_contract":"F-HH1-r1","status":"started","driver_sha256":sha(__file__),
            "host":platform.node(),"platform":platform.platform(),"python":sys.version,
            "utc_start":datetime.now(timezone.utc).isoformat()}
    def publish():save(directory/"result.json",report)
    publish();begin=time.perf_counter()
    try:
        actual=importlib.metadata.version(args.engine)
        if actual!=VERSIONS[args.engine]:raise RuntimeError(f"version mismatch: {actual}")
        report["installed_distributions"]={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}
        lock=OUT/"dependencies"/LOCKS[args.engine]
        mismatch=[]
        for line in lock.read_text().splitlines():
            pin=re.match(r"^([A-Za-z0-9_.-]+)==([^ ;\\]+)",line)
            if pin:
                try:installed=importlib.metadata.version(pin[1])
                except importlib.metadata.PackageNotFoundError:installed=None
                if installed!=pin[2]:mismatch.append({"package":pin[1],"expected":pin[2],"actual":installed})
        report["lock_sha256"]=sha(lock)
        report["dependency_lock_mismatches"]=mismatch
        if mismatch:raise ImportError("installed packages do not match the frozen engine dependency lock")
        data,path=fixture();report["fixture_sha256"]=sha(path)
        report["contract_sha256"]=sha(OUT/"F-HH1-contract.json");publish()
        {"jaxley":jaxley_probe,"braincell":braincell_probe,"brian2modelfitting":fitting_probe}[args.engine](data,report,publish,directory)
        report["status"]="capability_probe_completed"
    except Exception as error:
        report.update(status="dependency_failure" if isinstance(error,(ImportError,ModuleNotFoundError,importlib.metadata.PackageNotFoundError)) else "probe_failed",
                      exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    report["diagnostic_elapsed_seconds"]=time.perf_counter()-begin
    publish()


def run(args):
    if not args.allow_host or platform.node()!=args.allow_host:
        raise SystemExit("Run only on the coordinated test host via --allow-host <hostname>.")
    directory=OUT/"runs"/args.run_id/args.engine
    directory.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),"worker","--engine",args.engine,"--directory",str(directory)]
    env=dict(os.environ,JAX_PLATFORMS="cpu",JAX_ENABLE_X64="true",OMP_NUM_THREADS="1",
             OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",MPLBACKEND="Agg",
             MPLCONFIGDIR=str(directory/"matplotlib"),PYTHONDONTWRITEBYTECODE="1",
             BRAINEVENT_CACHE_DIR=str(directory/"cache/brainevent"),
             JAX_COMPILATION_CACHE_DIR=str(directory/"cache/jax"),XDG_CACHE_HOME=str(directory/"cache"))
    env.pop("PYTHONPATH",None)  # import only this engine's isolated package environment
    save(directory/"command.json",{"argv":command,"timeout":args.timeout,"performance_run":False,
                                  "environment":{k:env[k] for k in ("JAX_PLATFORMS","JAX_ENABLE_X64","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS")}})
    with (directory/"stdout.log").open("w") as stdout,(directory/"stderr.log").open("w") as stderr:
        process=subprocess.Popen(command,cwd=directory,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        try:process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            result=directory/"result.json"
            row=json.loads(result.read_text()) if result.exists() else {"performance_run":False,"engine":args.engine}
            row["status"]="timeout";row["timeout_seconds"]=args.timeout;save(result,row)
    print((directory/"result.json").read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["contract","run","worker"])
    parser.add_argument("--engine",choices=list(VERSIONS))
    parser.add_argument("--run-id",default="r1")
    parser.add_argument("--timeout",type=int,default=600)
    parser.add_argument("--allow-host")
    parser.add_argument("--directory")
    args=parser.parse_args()
    if args.mode=="contract":write_contract()
    elif args.mode=="worker":worker(args)
    else:run(args)


if __name__=="__main__":main()
