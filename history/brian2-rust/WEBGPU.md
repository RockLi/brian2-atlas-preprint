# Browser WebGPU: bounded independent-cell backend

Status (2026-09-09): **experimental, executable WebGPU/f32**, alongside the
existing WASM/f64 reference. The lab's Compute backend selector is explicit;
parameter, scale and backend edits require **Run experiment**. Errors preserve
previous results and do not silently select another backend.

## Supported scope

Adaptive LIF, Izhikevich, single-compartment Hodgkin–Huxley, AdEx, Quadratic IF
and bounded Equation Lab models run on the GPU. See [EQUATION_LAB.md](EQUATION_LAB.md)
for frontend custom authoring and its explicit language restrictions.
The Worker first validates the authored B2IR with the shared Rust/WASM compiler,
then lowers its scalar/vector expressions and schedule into WGSL. It preserves
start-of-tick monitoring, simultaneous-update temporaries, threshold/reset order,
fixed refractory ticks and HH expression-based rearming. HH uses cancellation-safe
`exprel`, including the finite-result/overflow split from the native GPU work.

Each invocation owns one independent cell and advances at most 256 ticks per
submission. All state remains resident between submissions. A cell-owned bitmap
records every spike without float atomics; readback reconstructs stable time/cell
order and full counts. Twelve or fewer state probes are recorded. This architecture
benefits independent cells; it does not measure a recurrent-network GPU scheduler.

**FlyWire remains WASM-only.** This WebGPU profile does not implement synapses,
delayed event queues, STDP, random input, multiple clocks, linked variables, custom
events, native Functions or arbitrary Brian2 programs. Existing Metal/CUDA support
for those features does not imply WebGPU support. Unsupported operations fail.

The export schema is `b2-webgpu-lab-v1`, with the WGSL, execution hash, adapter/device
limits and a separately labelled `reference_bundle`. The embedded reference plan
is a WASM/f64 reference, not a WebGPU physical plan. The WebGPU execution identity
binds the model, generated WGSL and `webgpu-f32-independent-v1` profile.

## Verified browser limits versus hardware limits

| Model | WASM neuron cap | WebGPU neuron cap | WebGPU neuron-tick cap |
| --- | ---: | ---: | ---: |
| Adaptive LIF | 32,768 | 98,304 | 600,000,000 |
| Izhikevich | 8,192 | 32,768 | 150,000,000 |
| Hodgkin–Huxley | 4,096 | 16,384 | 70,000,000 |
| FlyWire | 4,096 | Unsupported | — |
| AdEx / Quadratic IF / Equation Lab | 8,192 | 32,768 | 400,000,000 |

These are **lab budgets, not universal browser maxima**. Both paths retain a
40,000-step ceiling. The WebGPU path checks actual granted device buffer and
compute limits before allocation, and budgets three times the storage-buffer
size at 256 MiB for GPU buffers, staging and copies. This does not measure total
browser-process memory, which also includes the B2IR, WASM validation, JavaScript
objects and results. The spike bitmap alone takes
`neurons × ceil(steps / 32) × 4 bytes`, regardless of firing rate.

Both browser Workers reject more than **4 million recorded spikes** before
building/transferring the full result. WASM checks between bounded step batches;
WebGPU checks while decoding its bitmap, before allocating event arrays. This
protects the JavaScript result path from high-drive runs that fit the compute
budget but generate excessive output. It is not a bound on arbitrary custom-event
monitors exposed through the lower-level WASM API. Raster drawings sample at most
60,000 events; statistics still include all events of an accepted run.

The tested adapter advertised almost 4 GiB per buffer, but the requested device
actually granted **256 MiB maxBufferSize / 128 MiB maxStorageBufferBindingSize**.
The allocator checks the latter. Increasing advertised limits alone will not
remove browser memory pressure, GPU watchdogs or readback overhead. Stop terminates
the Worker; already-submitted bounded GPU work may finish before its resources
are reclaimed. Device-loss and validation/allocation errors are surfaced.

WebGPU needs a secure context (HTTPS or localhost), an available adapter and a
supported browser/driver. No SharedArrayBuffer, COOP/COEP or WASM threads are needed
for this backend. Offline operation requires the local assets to be served; opening
`index.html` directly as a file is not the supported Worker-loading path. Adapter
availability is checked at runtime. This delivery tested desktop Chromium and the
in-app browser on this Apple GPU, not Windows, Android, Safari or other GPU vendors.

References: [WebGPU specification](https://www.w3.org/TR/webgpu/),
[WGSL specification](https://www.w3.org/TR/WGSL/),
[Chrome WebGPU overview](https://developer.chrome.com/docs/web-platform/webgpu/overview).
WGSL provides f32 (and optional f16), not the reference engine's f64 arithmetic.

## Numerical results and limits

The hardware suite exercises all 12 classic presets, three shared maximum sizes,
a fixed-clock LIF fixture, and HH's removable-singularity voltages. It compares
full per-cell spike counts/coordinates and recorded traces with WASM/f64 and the
existing native CPU/f32 control. Passing the suite means execution, finite-state,
gate, fixture, resource and lifecycle checks passed. **It does not mean universal
numerical equivalence.** The raw report retains every discrepancy.

- The fixed-clock LIF fixture preserves exact spike coordinates and agrees across
  GPU chunk sizes 17 and 256. Its voltage error versus f64 is below 1e-6.
- Default LIF at 16,384 cells preserves all spike counts; a few spike coordinates
  differ by one tick. HH preserves counts in the tested cases, with up to one tick
  of timing difference and small continuous voltage/gate errors.
- Izhikevich bursting differs by one spike in one of 80 cells and up to four ticks
  versus f64. Fast spiking differs in eight cell counts (at most one per cell),
  with up to 29 ticks / **2.9 ms** timing drift over 400 ms. Reset-aligned voltage
  differences can approach 95 model mV. Native CPU/f32 also differs: six cell
  counts and up to 28 ticks in that fast-spiking comparison.

There is no threshold epsilon or hidden precision fallback. Use WASM/f64 when
exact reference spike timing is required. A f32 label alone does not imply
bitwise agreement across GPU compilers, fused operations or math libraries.

## Same-machine timing

Chrome 152 reported vendor `apple`, architecture `metal-3`. Paired measurements
use the same authored model: one warmup, then five alternating-order replays per
backend. Both timings include Worker startup, model validation/compilation,
execution, readback/transfer and result decoding; they exclude initial HTTP asset
fetch and chart drawing. They are wall-clock measurements, not kernel timestamps. The timing series was
collected before the final four-million-spike guard and cosmetic UI changes;
the final guard was independently exercised in both Workers.

| Model / cells | Duration / dt | WASM median (range), ms | WebGPU median (range), ms |
| --- | --- | ---: | ---: |
| LIF / 16,384 | 600 / 0.1 ms | 2,245 (2,209–2,270) | 137 (134–139) |
| Izhikevich regular / 4,096 | 400 / 0.1 ms | 395 (393–397) | 52 (50–53) |
| HH / 2,048 | 100 / 0.025 ms | 5,424 (5,399–5,431) | 56 (55–59) |

Larger WebGPU-only smoke runs completed at 65,536 LIF cells in 458 ms, 16,384
Izhikevich cells in 126 ms, and 8,192 HH cells in 129 ms. These are single runs,
not five-round performance estimates. The in-app LIF/65,536 run completed in
631 ms including its asset/loading path, producing 1,509,999 spikes. The final
guard-enabled in-app rerun completed in 489 ms with the same spike count. Results from
independent cells cannot predict the cost of FlyWire, delay queues or STDP.

## Reproduction and evidence

```sh
python brian2-rust/tools/build_wasm.py --wasm-bindgen /path/to/wasm-bindgen
python -m pytest brian2-rust/tests/test_wasm.py \
  brian2-rust/tests/test_flywire_browser.py brian2-rust/tests/test_browser_scales.py -q
node brian2-rust/wasm/check-webgpu-node.mjs brian2-rust/output/wasm
python brian2-rust/tools/prepare_webgpu_checks.py
python -m http.server 8765 --bind 127.0.0.1 --directory brian2-rust/output/wasm
```

Open `check-webgpu.html` and click Run hardware checks. CPU/f32 controls are local
build artifacts generated by `prepare_webgpu_checks.py`, not fetched from a
service. `check-webgpu-cancel.html` independently tests cancellation after an
actual GPU chunk and excessive-spike rejection in both Workers. Regular builds
version all module/CSS URLs and the WASM binary together to prevent stale mixed
browser builds.

See [2026-09-09 evidence](execution-plan-evidence/wasm/webgpu-20260909/README.md).
The existing 40 WASM tests pass on the current GPU branch; Rust library tests
also pass. Five maximum-size models were additionally compared against a separate
build of committed GPU HEAD `a97a92edd`, using exact spike arrays and the existing
HH libm tolerance. This checks the resumable shared-engine refactor against the
committed native engine rather than only comparing two bindings of the same code.

The later Equation Lab extension passed 51 pytest cases and twelve expanded-scale
browser smoke runs. Earlier numerical and timing evidence above retains its
original sizes; see [EQUATION_LAB.md](EQUATION_LAB.md) for the new measurements.
