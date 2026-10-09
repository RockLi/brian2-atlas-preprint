#!/usr/bin/env python3
"""E5 independent NumPy forward/reverse oracle; preparation is stdlib-only.

No engine code implements this oracle. --prepare only writes declared fixtures
when explicitly run later; current preparation does not call it. All numerical
commands reject hosts other than the scheduled evaluation Mac Studio.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "protocol/semantic-contracts.json"
FIXTURE_CONTRACT = ROOT / "evidence/e5-preparation-r1/fixture-contract.json"
MODELS = ("AdEx", "Izhikevich", "synthetic_four_state")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_remote():
    if socket.gethostname().split(".")[0] != "rock-mac-studio-1":
        raise RuntimeError("Numerical E5 preparation is scheduled only on 100.90.28.27")


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(plain(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def configuration():
    d = json.loads(CONTRACT.read_text())
    return d["E5"], d["shared"]


def dimensions(model):
    if model not in MODELS:
        raise ValueError("Unknown E5 model")
    return 4 if model == "synthetic_four_state" else 2


def validate_case(case, np):
    if case["semantic_contract_sha256"] != sha(CONTRACT):
        raise ValueError("Frozen semantic contract hash mismatch")
    if case["fixture_contract_sha256"] != sha(FIXTURE_CONTRACT):
        raise ValueError("Frozen fixture contract hash mismatch")
    if case["sizes"] != [2, 4, 2] or case["B"] != 2 or case["T"] != 32:
        raise ValueError("Only declared Q0 is prepared")
    x = np.asarray(case["inputs"], dtype=np.float64)
    if x.shape != (2, 32, 2) or not np.all((x == 0) | (x == 1)):
        raise ValueError("Q0 native external trigger coverage is binary only")
    D = dimensions(case["model"])
    initial = np.asarray(case["initial"], dtype=np.float64)
    if initial.shape != (2, D * 4 + 2 + 4):
        raise ValueError("Initial layout is physical hidden, output, then margin scratch")
    if not np.all(np.isfinite(initial)):
        raise ValueError("Nonfinite initial state")
    if list(case["labels"]) != [0, 1]:
        raise ValueError("Declared Q0 labels must be [0,1]")
    return D, x, initial


def forward_vjp(case, weights=None, initial=None, *, anchors=None):
    """Hand-written simultaneous Euler and reverse VJP, in declared units.

    anchors only defines an auxiliary smooth local-surrogate function for FD.
    Its hard reset gates stay fixed at base hard values. No finite differences
    of the discontinuous spike function are used to assert surrogate gradients.
    """
    import numpy as np
    spec, shared = configuration()
    D, x, init = validate_case(case, np)
    B, T, I = x.shape
    H, O = 4, 2
    model = case["model"]
    p = spec[model].get("parameters", {})
    dt = spec["common"]["dt_ms"]
    beta_o = 1. - dt / 20.
    w = [np.asarray(a, dtype=np.float64) for a in
         (case["weights"] if weights is None else weights)]
    if len(w) != 2 or w[0].size != I * H or w[1].size != H * O:
        raise ValueError("Weights require two source-major dense banks")
    wi, wo = w[0].reshape(I, H), w[1].reshape(H, O)
    init = np.array(init if initial is None else initial, dtype=np.float64)
    if init.shape != (B, D * H + O + H) or not np.all(np.isfinite(init)):
        raise ValueError("Invalid initial override")
    hidden = init[:, :D*H].reshape(B, D, H).copy()
    output = init[:, D*H:D*H+O].copy()
    tape, spikes, states = [], [], []
    for t in range(T):
        old = hidden.copy()
        v = old[:, 0]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if model == "AdEx":
                a = old[:, 1]
                exponential = np.exp((v-p["VT_mV"])/p["DeltaT_mV"])
                uv = v + dt*(p["gL_nS"]*(p["EL_mV"]-v)
                     + p["gL_nS"]*p["DeltaT_mV"]*exponential+p["I_pA"]-a)/p["C_pF"]
                ua = a + dt*(p["a_adapt_nS"]*(v-p["EL_mV"])-a)/p["tau_a_ms"]
                drift = np.stack([uv, ua], axis=1)
                mh = (uv-p["Vcut_mV"])/2.
                hard_h = (uv > p["Vcut_mV"]).astype(np.float64)
                width = 2.
            elif model == "Izhikevich":
                u = old[:, 1]
                uv = v + dt*(.04*v*v + 5*v + 140-u+p["I"])
                uu = u + dt*p["a"]*(p["b"]*v-u)
                drift = np.stack([uv, uu], axis=1)
                mh = (uv-30.)/10.
                hard_h = (uv >= 30.).astype(np.float64)
                width = 10.
            else:
                a, pf, q = old[:, 1], old[:, 2], old[:, 3]
                uv = v + dt*(-v-.2*a+.4*pf)/2.
                ua = a + dt*(-a)/10.
                up = pf + dt*(q-pf)
                uq = q + dt*(-q)/.5
                drift = np.stack([uv, ua, up, uq], axis=1)
                mh = uv-1.
                hard_h = (uv > 1.).astype(np.float64)
                width = 1.
        if not np.all(np.isfinite(drift)):
            raise FloatingPointError("numerical_divergence: nonfinite Euler drift")
        uo = beta_o * output
        mo = uo - 1.
        hard_o = (uo > 1.).astype(np.float64)
        gate_h, gate_o = hard_h.copy(), hard_o.copy()
        sh, so = hard_h.copy(), hard_o.copy()
        if anchors is not None:
            anchor = anchors[t]
            gate_h, gate_o = anchor["gate_h"], anchor["gate_o"]
            sh = gate_h + (mh-anchor["mh"])/(1.+5.*np.abs(anchor["mh"]))**2
            so = gate_o + (mo-anchor["mo"])/(1.+5.*np.abs(anchor["mo"]))**2
        hidden = drift.copy()
        # Source-major addition matches the declared operation order.
        target = 3 if model == "synthetic_four_state" else 0
        for i in range(I):
            hidden[:, target] += x[:, t, i, None] * wi[i]
        output = uo.copy()
        for h in range(H):
            output += sh[:, h, None] * wo[h]
        if model == "AdEx":
            hidden[:, 0] = (1.-gate_h)*hidden[:, 0] + gate_h*p["Vr_mV"]
            hidden[:, 1] += p["b_pA"]*gate_h
        elif model == "Izhikevich":
            hidden[:, 0] = (1.-gate_h)*hidden[:, 0] + gate_h*p["c"]
            hidden[:, 1] += p["d"]*gate_h
        else:
            hidden[:, 0] -= gate_h
            hidden[:, 1] += .1*sh
        output -= gate_o
        if not np.all(np.isfinite(hidden)) or not np.all(np.isfinite(output)):
            raise FloatingPointError("numerical_divergence: nonfinite next state")
        tape.append(dict(old=old, mh=mh, mo=mo, sh=sh, so=so,
                         gate_h=gate_h, gate_o=gate_o, width=width))
        spikes.append(np.concatenate([sh, so], axis=1))
        states.append(np.concatenate([hidden.reshape(B, -1), output, mh], axis=1))
    spike_array = np.stack(spikes, axis=1)
    logits = np.zeros((B, O), dtype=np.float64)
    for t in range(T):
        logits += spike_array[:, t, -O:] * (5./T)
    shifted = logits-logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = exponent/exponent.sum(axis=1, keepdims=True)
    labels = np.asarray(case["labels"], dtype=np.int64)
    loss = np.mean(np.log(exponent.sum(axis=1))-shifted[np.arange(B), labels])
    dlogits = probabilities.copy()
    dlogits[np.arange(B), labels] -= 1.
    dlogits /= B
    dh = np.zeros((B, D, H))
    dout = np.zeros((B, O))
    dwi, dwo = np.zeros_like(wi), np.zeros_like(wo)
    for t in range(T-1, -1, -1):
        q = tape[t]
        dv = dh[:, 0]
        da = dh[:, 1]
        feed_bar = dh[:, 3] if D == 4 else (1.-q["gate_h"])*dv
        dwi += x[:, t, :].T @ feed_bar
        dwo += q["sh"].T @ dout
        dsh = dout @ wo.T
        if model == "synthetic_four_state":
            dsh += .1*da
        ph = 1./(1.+5.*np.abs(q["mh"]))**2/q["width"]
        po = 1./(1.+5.*np.abs(q["mo"]))**2
        duv = (dv if D == 4 else feed_bar) + ph*dsh
        duo = dout + po*(5./T)*dlogits
        old_v = q["old"][:, 0]
        previous = np.zeros_like(dh)
        if model == "AdEx":
            ex = np.exp((old_v-p["VT_mV"])/p["DeltaT_mV"])
            previous[:, 0] = duv*(1.+dt*p["gL_nS"]*(ex-1.)/p["C_pF"]) + da*dt*p["a_adapt_nS"]/p["tau_a_ms"]
            previous[:, 1] = -dt/p["C_pF"]*duv + (1.-dt/p["tau_a_ms"])*da
        elif model == "Izhikevich":
            previous[:, 0] = duv*(1.+dt*(.08*old_v+5.)) + da*dt*p["a"]*p["b"]
            previous[:, 1] = -dt*duv + (1.-dt*p["a"])*da
        else:
            dp, dq = dh[:, 2], dh[:, 3]
            previous[:, 0] = (1.-dt/2.)*duv
            previous[:, 1] = -dt*.2/2.*duv + (1.-dt/10.)*da
            previous[:, 2] = dt*.4/2.*duv + (1.-dt)*dp
            previous[:, 3] = dt*dp + (1.-dt/.5)*dq
        dh, dout = previous, beta_o*duo
    initial_vjp = np.concatenate([dh.reshape(B, -1), dout, np.zeros((B, H))], axis=1)
    if not all(np.all(np.isfinite(a)) for a in (dwi, dwo, initial_vjp)):
        raise FloatingPointError("numerical_divergence: nonfinite reverse")
    return dict(loss=float(loss), logits=logits, spikes=spike_array,
                states=np.stack(states, axis=1), final_state=states[-1],
                gradients=[dwi.reshape(-1), dwo.reshape(-1)],
                initial_state_gradients=initial_vjp, anchors=tape)


def make_fixtures():
    """Predeclared tiny arrays. Never change weights after observing outcomes."""
    import numpy as np
    spec, _ = configuration()
    declared = json.loads(FIXTURE_CONTRACT.read_text())
    rng = np.random.default_rng(declared["seed"])
    inputs = (rng.random((2, 32, 2)) < .45).astype(np.float64)
    cases = []
    dt = spec["common"]["dt_ms"]
    for model in MODELS:
        D = dimensions(model)
        p = spec[model].get("parameters", {})
        for boundary in (False, True):
            initial = np.zeros((2, D*4+2+4))
            if model == "AdEx":
                initial[:, :4] = p["EL_mV"]
            elif model == "Izhikevich":
                initial[:, :4] = -65.
                initial[:, 4:8] = -13.
            if boundary:
                if model == "AdEx":
                    vs = np.array([p["Vcut_mV"], p["Vcut_mV"]-1e-7,
                                   p["Vcut_mV"]+1e-7, p["EL_mV"]])
                    second = np.zeros(4)
                    second[:3] = (p["gL_nS"]*(p["EL_mV"]-vs[:3])
                        + p["gL_nS"]*p["DeltaT_mV"]*np.exp((vs[:3]-p["VT_mV"])/p["DeltaT_mV"])
                        + p["I_pA"])
                elif model == "Izhikevich":
                    vs = np.array([30.,30.-1e-7,30.+1e-7,-65.])
                    second = np.array([.04*v*v+5*v+140+p["I"] for v in vs])
                    second[3] = -13.
                else:
                    center = np.float64(1./.95)
                    candidate = center
                    for _ in range(8):
                        candidate = np.nextafter(candidate, -np.inf)
                    hits = []
                    for _ in range(17):
                        uv = candidate + dt*(-candidate-.2*0.+.4*0.)/2.
                        if uv == 1.:
                            hits.append(float(candidate))
                        candidate = np.nextafter(candidate, np.inf)
                    if not hits:
                        raise RuntimeError("predeclared_boundary_construction_failure: no exact literal-Euler value within ±8 ULP; do not tune")
                    exact = hits[0]
                    vs = np.array([exact, exact-1e-7, exact+1e-7, 0.])
                    second = np.zeros(4)
                initial[0, :4], initial[0, 4:8] = vs, second
                initial[1, :4] = np.roll(vs, 1)
                initial[1, 4:8] = np.roll(second, 1)
                initial[:, D*4:D*4+2] = [[1./.995,1./.995+1e-7],
                                         [1./.995-1e-7,.2]]
            # Scratch is overwritten each tick; deliberately nonzero initial values.
            initial[:, D*4+2:] = [[7.,-3.,21.,-11.],[-4.,12.,-8.,33.]]
            cases.append(dict(id=f"E5-Q0-{model}-{'boundary' if boundary else 'base'}",
                model=model, boundary=boundary, seed=declared["seed"], sizes=[2,4,2],
                B=2,T=32,inputs=inputs,labels=[0,1],initial=initial,
                weights=[np.array(declared["Wi"]).reshape(-1),np.array(declared["Wo"]).reshape(-1)],
                semantic_contract_sha256=sha(CONTRACT),
                fixture_contract_sha256=sha(FIXTURE_CONTRACT),performance_run=False))
    return plain(cases)


def local_surrogate_fd(case):
    import numpy as np
    base = forward_vjp(case)
    anchors = base["anchors"]
    checks = []
    for bank, values in enumerate(case["weights"]):
        for j in range(len(values)):
            delta = 1e-6
            plus, minus = copy.deepcopy(case["weights"]), copy.deepcopy(case["weights"])
            plus[bank][j] += delta
            minus[bank][j] -= delta
            fd = (forward_vjp(case, plus, anchors=anchors)["loss"] -
                  forward_vjp(case, minus, anchors=anchors)["loss"])/(2.*delta)
            expected = base["gradients"][bank][j]
            checks.append(dict(field=f"weight[{bank}][{j}]", finite_difference=fd,
                manual=expected, passed=bool(abs(fd-expected)<=2e-7+2e-5*abs(expected))))
    for b in range(2):
        for j in range(len(case["initial"][b])):
            delta = 1e-6
            plus, minus = np.array(case["initial"]), np.array(case["initial"])
            plus[b,j] += delta
            minus[b,j] -= delta
            fd = (forward_vjp(case, initial=plus, anchors=anchors)["loss"] -
                  forward_vjp(case, initial=minus, anchors=anchors)["loss"])/(2.*delta)
            expected = base["initial_state_gradients"][b,j]
            checks.append(dict(field=f"initial[{b}][{j}]", finite_difference=fd,
                manual=expected, passed=bool(abs(fd-expected)<=2e-7+2e-5*abs(expected))))
    return dict(scope="FD of fixed-anchor smooth surrogate extension only",
                passed=all(q["passed"] for q in checks), checks=checks)


def adam_reference(case, steps=3):
    import numpy as np
    _, shared = configuration()
    opt = shared["optimizer"]
    weights = [np.array(w, dtype=np.float64) for w in case["weights"]]
    m, v = [np.zeros_like(w) for w in weights], [np.zeros_like(w) for w in weights]
    trace = []
    for step in range(1,steps+1):
        value = forward_vjp(case, weights)
        for bank, gradient in enumerate(value["gradients"]):
            m[bank] = opt["beta1"]*m[bank]+(1-opt["beta1"])*gradient
            v[bank] = opt["beta2"]*v[bank]+(1-opt["beta2"])*gradient*gradient
            weights[bank] -= opt["learning_rate"]*(m[bank]/(1-opt["beta1"]**step))/(
                np.sqrt(v[bank]/(1-opt["beta2"]**step))+opt["epsilon"])
        trace.append(dict(step=step, loss_before_update=value["loss"],
            gradients=value["gradients"], weights=[a.copy() for a in weights],
            m=[a.copy() for a in m], v=[a.copy() for a in v]))
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--case", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--adam", action="store_true")
    args = parser.parse_args()
    if not args.output:
        parser.error("--output is required and must be new")
    guard_remote()
    if args.prepare:
        if args.case or args.selfcheck or args.adam:
            parser.error("--prepare cannot run a model")
        write_new(args.output, dict(status="fixtures_generated_not_qualified",
                                  performance_run=False,cases=make_fixtures()))
        return
    if not args.case:
        parser.error("--case is required")
    case = json.loads(args.case.read_text())
    result = forward_vjp(case)
    if args.selfcheck:
        result["local_surrogate_fd"] = local_surrogate_fd(case)
    if args.adam:
        result["adam"] = adam_reference(case)
    # Keep observed FP64 first-tick margins; nominal zero is not asserted.
    result["first_hidden_margin"] = result["anchors"][0]["mh"]
    result["first_output_margin"] = result["anchors"][0]["mo"]
    del result["anchors"]
    result.update(status="independent_oracle_computed_native_not_executed",
                  performance_run=False,case=case["id"])
    write_new(args.output,result)


if __name__ == "__main__":
    main()
