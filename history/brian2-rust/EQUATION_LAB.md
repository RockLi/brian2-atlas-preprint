# Equation Lab: frontend model authoring

The lab remains a static site. Equation Lab, AdEx and Quadratic IF accept editable
models and execute locally in a Worker. No Python service, notebook server,
remote compiler, dynamic JavaScript evaluation or Pyodide is required.

Select **Equation Lab**, choose an example, edit the **Equations** tab and press
**Run experiment** (or Cmd/Ctrl+Enter). Parameter, equation and backend edits leave
the previous result visible until Run. **Save design** writes a small JSON file;
**Load design** restores it without executing it. The existing full experiment
export also includes the custom source in `experiment_config.custom`.

## Supported language

This is a Brian2-style equation subset, not a Python/Jupyter kernel:

```text
dv/dt = (drive - v - w) / (tau * ms) : 1 (unless refractory)
dw/dt = -w / (tau_w * ms) : 1
```

Parameters (numeric JSON): `{"tau":20,"tau_w":120,"jump":0.06}`.
Initial state: `{"v":0,"w":0}`. Threshold: `v > 1`.
Reset: `v = 0` followed by `w += jump`. Fixed refractory period: 2 ms.

- One independent population, 1–4 dimensionless state coordinates, including `v`.
  Custom coordinate systems can represent numerical mV or other normalized units;
  the compiler does not treat such coordinates as Brian physical voltage units.
- Explicit simultaneous Euler: every derivative reads the old state; updates then
  commit together. Resets execute sequentially. Start-of-tick probes match Brian2.
- Up to 24 numeric parameters and one numeric initial value per state. `drive` is
  a reserved input parameter, varied by a seeded uniform spread across cells.
  Identical input disables spread; custom initial values are identical across cells.
- `ms`, `second`, `t` and `dt` carry physical time dimensions. Dividing by `ms`
  converts a derivative written in numerical millisecond coordinates. Numeric
  user parameters are dimensionless; use `tau * ms` for a time constant.
- `+ - * / **`, parentheses, `exp`, `exprel`, comparisons and `and/or/not`.
  Powers require a literal integer 0–8. Reset operators: `=`, `+=`, `-=`.
- Only states marked `(unless refractory)` freeze. Thresholds are tested every
  step outside the fixed refractory period. Without a reset/rearming mechanism,
  a sustained suprathreshold state produces repeated spike events.
- No Python statements, imports, user functions, stochastic equation calls,
  linked populations, synapses, general physical units or expression refractory.
  FlyWire remains an existing connected WASM model, outside this editor.

`equations.js` tokenizes and parses expressions into a bounded AST. It constructs
B2IR states, parameters, simultaneous-update temporaries, monitoring and effect
ordering. WASM independently validates types, units, effects and scheduling before
building hashes and an execution plan. WebGPU lowers that validated model into
f32 WGSL, with no silent fallback. The original built-in HH uses exponential Euler;
custom models use explicit Euler and may require a smaller timestep.

Text, token, nesting and expression-expansion limits bound authoring work.
Non-finite traces and final states fail the run. Both backends retain the existing
four-million-spike output ceiling; WebGPU keeps the 256 MiB buffer/staging/copy
budget and checks actual device limits. These are application limits, not a
measurement or guarantee of available browser memory. Custom equation complexity
can make equal neuron-tick counts take very different amounts of time.

## Added examples

AdEx uses numerical voltage in mV, current and adaptation divided by leak
conductance, a -30 mV cutoff, -65 mV reset and a fixed refractory period. It is a
parameterized demonstration, not a reproduction of a particular Naud figure.
See the [Brian2 AdEx example](https://brian2.readthedocs.io/en/stable/examples/frompapers.Naud_et_al_2008_adex_firing_patterns.html).

Quadratic IF uses `tau * dv/dt = v**2 + drive` with finite cutoff/reset at ±2,
normalized coordinates and a fixed refractory period. Finite cutoffs change firing
rates relative to the infinite-cutoff idealization. See
[Neuronal Dynamics, section 5.3](https://neuronaldynamics.epfl.ch/online/Ch5.S3.html).

Equation Lab also offers editable adaptive LIF and Izhikevich starting points.
Their initialization differs from the original fixed-template models; do not
compare cross-model seeds as if they defined identical initial arrays.

## Expanded scales and verification (2026-09-09)

| Model | WASM maximum | WebGPU maximum |
| --- | ---: | ---: |
| Adaptive LIF | 32,768 | 98,304 |
| Izhikevich | 8,192 | 32,768 |
| Hodgkin–Huxley | 4,096 | 16,384 |
| AdEx / Quadratic IF / Equation Lab | 8,192 | 32,768 |
| FlyWire | 4,096 | Unsupported |

All six independent modes completed at these maxima in the in-app Chrome 152
browser at their default durations/timesteps. Example single-run Worker timings:
LIF/WebGPU 98,304: 702 ms; HH/WebGPU 16,384: 213 ms; HH/WASM 4,096: 10,751 ms.
These are smoke timings including compilation and transfer, excluding initial
asset fetches and charts; they are not repeated performance estimates or promises
for other hardware. Longer durations/smaller timesteps can exceed work/memory limits.

51 pytest cases passed, including 11 new comparisons with independently constructed
Brian2 NumPy models. Spike indices/ticks matched; traces matched at rtol=1e-9,
atol=1e-8. The original WASM/native/FlyWire suite also passed with new maximum
reference sizes. Browser checks covered ten editable presets on both backends,
exact four-state coupled Euler with a binary clock, and twelve maximum-size runs.
All ten editable presets had identical per-cell spike counts in this sample; this
does not establish general f32/f64 equivalence or exact spike-time agreement.
See [WEBGPU.md](WEBGPU.md) for the existing measured f32 limitations.

```sh
node brian2-rust/wasm/check-equations.mjs brian2-rust/output/wasm /tmp/equation-checks
python -m pytest brian2-rust/tests/test_browser_equations.py \
  brian2-rust/tests/test_wasm.py brian2-rust/tests/test_browser_scales.py \
  brian2-rust/tests/test_flywire_browser.py -q
```

Open `check-equations-browser.html` on the static site and click **Run browser
checks** for hardware verification. Reuse the in-app browser; the checker does not
launch or leave behind standalone Chrome sessions.
