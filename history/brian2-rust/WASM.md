# ExecutionPlan browser executor

The WASM backend executes the shared `LogicalPlan` in a browser Worker. Python
builds an immutable `WasmPlan` with `build_execution_plan(model, backend="wasm")`.
The browser independently validates the B2IR hashes and semantics, re-derives
completed effects/dependencies, checks every plan field and uses its verified
node ordinals as the runtime dispatch order. B2IR v1 is unchanged. CPU fusion
and GPU physical plans are not browser input.

`src/model.rs` contains the existing wire types and validator moved from the
CLI. `src/executor.rs` owns the shared compiled programs, typed state, clocks,
linked inputs, event queues and monitors. Native and WASM execute this same
runtime. `src/wasm_plan.rs` validates the browser plan; `src/wasm.rs` exports the
JavaScript interface. The native CLI and binary result protocol are preserved.

## Build and open the example

Use the Rust toolchain configured for this worktree (tested with Rust 1.98.1),
Python with this repository's Brian2 dependencies, and wasm-bindgen 0.2.100:

```sh
rustup target add wasm32-unknown-unknown --toolchain 1.98.1
cargo install wasm-bindgen-cli --version 0.2.100 --locked
rustup run 1.98.1 python brian2-rust/tools/build_wasm.py
python -m http.server 8765 --bind 127.0.0.1 --directory brian2-rust/output/wasm
```

Open `http://127.0.0.1:8765/` for **Neural Lab**, the interactive single-page
experiment workbench. It automatically runs the saved configuration or a
160-neuron Adaptive LIF model on first load. Model selection, presets and edits
only change the configuration; click **Run experiment** (or Cmd/Ctrl+Enter) to
compute new results. The Run button stays at the top of the parameter panel.
All interface text is English and the desktop layout uses the available width.

Three independent-cell models have their own equations, parameters, presets,
timestep bounds, units and trace labels:

| Model | Presets | State traces | Integration |
| --- | --- | --- | --- |
| Adaptive LIF | Asynchronous, synchronous, adaptation, subthreshold | Normalized voltage v and adaptation w | Euler, default 0.1 ms |
| Izhikevich (2003) | Regular spiking, bursting, fast spiking, resting | Voltage v (mV coordinate), recovery u (model units) | Euler, default 0.1 ms |
| Hodgkin–Huxley (1952) | Tonic, strong drive, synchronous, resting | Voltage (mV), Na activation m, Na inactivation h, K activation n | Exponential Euler, default 0.025 ms |

Izhikevich follows the [Brian2 example](https://brian2.readthedocs.io/en/stable/examples/frompapers.Izhikevich_2003.html),
with the algebraically equivalent factored polynomial `0.04*v*(v+125)+140`
to avoid cross-libm square differences being amplified by threshold resets.
The bursting preset uses c = −50 and d = 2 (chattering-style bursts).
HH uses the [1952 equations](https://brian2.readthedocs.io/en/stable/examples/compartmental.hodgkin_huxley_1952.html)
as isopotential cells with voltage shifted by −65 mV, classic squid-axon
kinetics, Cm = 1 µF/cm² and ENa/EK/EL = 50/−77/−54.387 mV. Injected current
is in µA/cm²; conductances are in mS/cm². It has no voltage reset. Spike
monitoring detects v > 0 and rearms when v <= −40, while all states continue
to integrate. It does not implement a spatial axon or temperature control.

Model selection loads that model's defaults. Sliders or editable JSON control
its parameters, cell count, duration, timestep and seed. JSON may change the
model when the complete corresponding configuration is provided. This is a
parameter editor for the displayed equations, not a Python interpreter or an
arbitrary equation compiler. The numerical timestep is bounded per model;
all configurations are additionally limited to 40,000 steps. Neuron-count and
neuron-tick caps are model-specific (see the scale table below). Non-finite traces are rejected before replacing existing results.

A fourth **FlyWire · DM1** mode executes real induced olfactory subgraphs
of 240, 1,024 or 4,096 cells, with up to 284,105 weighted edges. It includes stimulation and matched sensory-
output-cut controls, schematic network replay, group firing rates and exact
root-ID inspection. See [FLYWIRE_BROWSER.md](FLYWIRE_BROWSER.md) for extraction,
attribution, model assumptions and verification. This mode has recurrent signed
synapses; the independent-cell description below applies to the three classic
single-cell model demos.

The page renders real WASM results as a spike raster, population firing-rate
curve, per-neuron rate histogram and selected-probe state traces. Rate-bin
width, recorded probe selection and time replay update the existing results
without re-running the simulation. The previous run's rate curve is shown in
grey when simulation durations match. Voltage and auxiliary states use separate
axes; negative voltage and gate fractions have appropriate scales. A min/max
envelope preserves narrow spikes during trace downsampling. Invalid JSON/configuration leaves previous results intact; Stop terminates
the Worker. Up to twelve evenly spaced state probes are recorded. Cells are
independent; synchronized firing is driven by identical initial states/input,
not network connectivity. Export downloads a verified bundle with the last
executed configuration attached; the raster can also be saved as PNG.

Browser editing explicitly calls the WASM `compile_model(draft_json)` authoring
API. It validates the edited B2IR semantics, generates new canonical wire hashes
and derives the complete logical execution plan before returning a bundle.
The existing `BrowserExecutor` constructor remains strict and rejects stale
hashes. `runDraft(draft, options)` performs authoring and execution in a Worker;
`runBundle(bundle, options)` continues to load existing verified bundles.

Build output includes JavaScript, TypeScript declarations and
`pkg/b2_runner_bg.wasm`; it is ignored by Git. `--output` and `--wasm-bindgen`
allow custom output/tool paths. Static HTTP hosting is sufficient after build;
Python, a simulation service, WASI and cross-origin isolation are not needed by
the running simulation. The old two-neuron `sample.json` remains available for
low-level API checks.

On this Mac's custom 1.98.1 installation, the linker requires
`DYLD_LIBRARY_PATH=/atlas-home/0004/.rustup/toolchains/1.98.1-aarch64-apple-darwin/lib`
during the build because its bundled `rust-lld` cannot locate `libLLVM.dylib`.
This is a local toolchain issue, not a browser runtime dependency.

## Export and execute a model

The lab now includes **Load WASM model** (`import.html`): choose a Brian2 export,
press Run, then switch populations and recorded variables. See [WASM_IMPORT.md](WASM_IMPORT.md)
for a complete Python example, viewer limits and result export instructions.

From an existing B2IR model dictionary (for example, a Device's `model.json`):

```python
from brian2_rust import build_execution_plan, export_wasm_bundle, explain_plan

plan = build_execution_plan(model, backend="wasm")
print(explain_plan(plan))
export_wasm_bundle(model, "network.browser.json", plan=plan)
```

The export and public planner use the independent native Rust validator.
`export_wasm_bundle` rejects a stale or modified plan. The bundle carries
`model_json` and `plan_json` as strings: **do not parse and re-stringify the
model in JavaScript**. B2IR contains integer seeds/counters that can exceed
JavaScript's exact Number range. Rust parses the original JSON directly.

The high-level browser helper creates one isolated Worker for each run:

```javascript
import { runBundle } from './runtime.js';
const bundle = await (await fetch('./network.browser.json')).json();
const controller = new AbortController();
const { results, events, summary } = await runBundle(bundle, {
  batchTicks: 128,
  signal: controller.signal,
  onProgress: ({ batches }) => console.log(batches),
});
// controller.abort() terminates even a Worker still compiling a model.
// results/events are Uint8Arrays; summary is a JSON object bound to plan_sha256.
```

For direct integration, `pkg/b2_runner.js` exports `BrowserExecutor`:

```javascript
import init, { BrowserExecutor } from './pkg/b2_runner.js';
await init();
const executor = new BrowserExecutor(bundle.model_json, bundle.plan_json);
try {
  while (!executor.finished) {
    executor.step(128);
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  const results = executor.results();
  const events = executor.events();
  const summary = JSON.parse(executor.summary());
} finally {
  executor.free();
}
```

`step(n)` advances at most `n` scheduler instants, activating all coincident
clocks in plan order, and returns whether execution is complete. `n` must be
positive. Pending events and RNG counter identities survive batch boundaries.
An execution error makes the executor unusable; partial results are rejected.
Completed results are repeatable and retain the native little-endian dump
format, including exact 64-bit integer arrays. `bind_execution_plan` accepts
WASM summaries and checks their backend, numeric profile and plan hash.

## Scope and numerical contract

This is a serial tiled WASM executor using the reference numeric profile. It
supports the portable B2IR operations implemented by the shared reference
engine, including multiple clocks, delayed pre/post pathways, synaptic state
updates, summed variables, custom events, linked variables, typed monitors,
refractory behavior, portable Functions, TimedArray, RNG and inline/procedural
topology. The tests exercise these through real exported models. This is not
WebGPU and makes no GPU acceleration claim.

Native-only Functions and filesystem-backed binary CSR topology are rejected;
export portable Function bodies and inline/procedural topology for browsers.
The existing reference validator's model/array/run limits still apply, with
an additional 16 MiB plan limit and the WASM model's 128 MiB JSON limit.
WASM32 indices and clocks must fit `usize`; 64-bit typed state and RNG seeds
remain 64-bit. Memory is per Worker and subject to the browser's WASM32 limit.
There is no checkpoint export or continuation past the bundle's declared run;
B2IR segments with nonzero clocks and pending events can be loaded, and a
loaded run can be stepped incrementally.

IEEE f64 operations and explicit typed storage are preserved. Native and WASM
transcendental libraries can differ in the last few bits; statistical/RNG
normal values are compared with tight numerical tolerance, not a cross-libm
bitwise promise. Different tick batch sizes must produce identical WASM bytes.

## Verification

After building, run from the repository root with `brian2-rust/python` on
`PYTHONPATH`:

```sh
python -m pytest brian2-rust/tests/test_wasm.py -q \
  --basetemp=brian2-rust/output/wasm-tests
python brian2-rust/tools/prepare_wasm_browser_checks.py
```

Open `/check-browser.html` on the example server and click **Run verification**.
The browser suite executes the native-validated test corpus through Workers,
checks all result/event bytes against the Node execution of the same WASM,
rejects altered plans/bundles, cancels during execution, tests pre-aborted runs
and checks isolation of concurrent Workers. This page reports pass/fail visibly.
The Python suite compares decoded state, spikes, event streams, synapses and
monitor data against native execution for every case and runs each WASM model
with batch sizes 1, 7 and 10,000. It also tests malformed input, plan drift,
large integer seeds, runtime failure and result lifecycle.

### SPA acceptance

`test_browser_authoring_and_experiment_metrics` runs `wasm/check-spa.mjs` and
checks each generated model against native Rust, including all result/event
bytes. The check covers changed plan identities, strict stale-hash rejection,
invalid semantic drafts, configuration validation, zero-spike behavior,
synchronous tick alignment, adaptation and exact normalization of a partial
final rate bin. `compile_model` never bypasses B2IR semantic validation.

Observed default result: 3,706 spikes in 600 ms across 160 neurons, or
38.6041666667 Hz/neuron. Browser acceptance covers JSON editing/error recovery,
changing drive, bin/probe controls without execution, scrubbing, playback,
cancellation and layouts at 1440 px and 390 px. Screenshots are generated under
`output/playwright/`; reproducible test results are recorded in
`execution-plan-evidence/wasm/spa-verification.json`.

### Classic-model verification

`wasm/check-models.mjs` executes all twelve presets in WASM, checks finite
traces, resting silence, burst intervals, fast-spiking rates, HH voltage
excursions and gate bounds, and writes model/results for differential tests.
`tests/test_wasm.py` compares their spike ticks and indices exactly with the
native executor, and state/trace values within 1e-9 absolute / 1e-10 relative
tolerance. Four additional cases compare against Brian2's independent NumPy
runtime: Izhikevich regular/bursting and HH tonic/resting, with identical
spike ticks and trace tolerance 1e-8. These checks cover the shipped presets;
custom parameter choices remain fixed-step numerical experiments.

## Browser population scales

| Model | Maximum neurons | Maximum neuron-ticks |
| --- | ---: | ---: |
| Adaptive LIF | 32,768 | 200,000,000 |
| Izhikevich | 8,192 | 40,000,000 |
| Hodgkin–Huxley | 4,096 | 20,000,000 |
| FlyWire DM1 | 4,096 | 40,000,000 |
| AdEx / Quadratic IF / Equation Lab | 8,192 | 100,000,000 |

The Population scale selector is next to Run; classic models also accept custom
integer counts. FlyWire selects exact 240 / 1,024 / 4,096-cell assets. Presets
preserve the selected count, while model switching and Reset restore defaults.
Scale changes require Run. Assets load on demand and Stop aborts loading or the
Worker. Limits bound work, not elapsed time; duration and dt still affect cost.
All cells and spikes contribute to statistics. Only 12 state probes are recorded;
the raster uniformly samples at most 60,000 events and labels that sampling.
Network drawings sample at most 96 cells per group and 600 edges between those
cells, with displayed and simulated counts labelled. Rates are cached for replay;
only the preceding run's spike times are retained for its comparison curve.

`tests/test_browser_scales.py` verifies native parity at every model's maximum
count, both expanded FlyWire circuits, and nested connectivity/asset identities.
HH uses the existing native/WASM libm tolerance; all spike ticks and IDs are exact.


## WebGPU option (2026-09-09)

The lab also exposes an explicit experimental WebGPU/f32 backend for independent
built-in and Equation Lab models. FlyWire remains WASM-only. See [WEBGPU.md](WEBGPU.md)
for actual numerical deviations, same-machine timings, browser/resource limits
and supported scope. Larger WebGPU scales do not raise the WASM limits above.
Both Workers cap accepted spike output at four million events. Static builds now
version local module/CSS URLs and the WASM binary together to avoid mixed caches.

Frontend custom equations, added models and expanded-scale verification: [EQUATION_LAB.md](EQUATION_LAB.md).
