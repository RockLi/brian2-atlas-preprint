# Load a Brian2 model in the browser

Open **Load WASM model** in the lab header (`import.html`). Choose or drop a
`.browser.json` file. Loading reads metadata; **Run imported model** verifies the
model/plan and starts a dedicated WASM Worker. No Python backend is used.

The import is a `b2-wasm-bundle-v0` model/plan bundle for the shared engine, not an
arbitrary executable `.wasm` binary. Models are executed as exported; importing
never recompiles a modified draft, changes its parameters or converts it to WebGPU.

## Export a network

Install the repository's Python dependencies, add `brian2-rust/python` to
`PYTHONPATH`, and build the native validator:

```sh
cargo build --manifest-path brian2-rust/Cargo.toml --release --bin b2-runner
PYTHONPATH=brian2-rust/python:. python brian2-rust/wasm/export-browser-model.py \
  --output browser-export
```

This writes `browser-export/import-example.browser.json` and `model.json`, without
simulating the network. The complete example builds 24 input neurons, 12 output
neurons, 24 delayed feedforward edges, SpikeMonitors for both populations, and
StateMonitors with normalized `v` and physical-voltage `vm` coordinates.

For an existing compatible Brian2 Network:

```python
import brian2 as b
from brian2_rust import export_network, export_wasm_bundle

b.set_device("rust_standalone", build_on_run=False)
# Construct groups, synapses and monitors here, then network = b.Network(...).
model = export_network(network, 300*b.ms, "model.json", namespace={})
export_wasm_bundle(model, "network.browser.json")
```

Pass any external equation constants/functions in `namespace`, or attach them to
the Brian groups. Synapses must share their source group's clock in the currently
supported exporter. Portable Function bodies are required for browser execution.
The exporter reports unsupported Brian2 constructs rather than silently omitting
them. This is the existing Rust Device/B2IR supported subset, not every Brian2 API.

The browser page provides a downloadable complete Python example and its generated
bundle. **Load example** selects that bundle without running it. The static build
creates the example automatically; Python/Rust are needed at export/build time,
not while the browser runs the model.

## Results and scientific scope

- Select a population, recorded variable and probe. Raster/rate/distribution
  statistics refer to that population's **recording window**, which is stated
  explicitly. Replay scrubs the existing result; it does not resimulate.
- Each population retains its own timestep and trailing recording window. Plot
  time starts at zero relative to that window; its absolute start/end are shown.
- No `v` name or dimensionless-voltage assumption: state selection accepts exported
  monitor variables. Values use SI units or dimensionless numerical coordinates.
- Missing SpikeMonitor data is labelled as unrecorded, not interpreted as silence.
  Missing StateMonitor data hides the trace panel. Runtime synaptic-delivery counts
  refer to the whole imported network.
- f64/f32, bool and integer recordings decode according to their declared types.
  i64/u64 variables outside ±(2^53−1) are omitted from plots with an explicit note;
  exact values remain in the unmodified binary result.
- Download the verified source bundle, `results.bin`, `events.bin`, and
  `summary.json`. The supplied Python `model.json` plus these three run outputs
  can be read with `brian2_rust.results.load_results(model, result_directory)`.
- Run failures and Stop preserve the preceding completed result. Downloads remain
  bound to that completed result even if another file has been selected.

## Compatibility and application limits

The importer supports compatible single/multiple populations, multiple clocks,
inline or procedural connectivity, delays and portable functions. It rejects
external BinaryCSR paths, native-only functions and custom EventMonitor recordings
at preflight. The underlying WASM executor supports a broader API; these restrictions
bound the interactive viewer and avoid unbounded custom event recordings.

| Input/resource | Import limit |
| --- | ---: |
| Bundle file | 32 MiB |
| Embedded model / plan JSON | 24 / 4 MiB |
| Populations / clocks / synapse objects | 16 / 16 / 64 |
| Total neurons / total edges | 100,000 / 1,000,000 |
| Sum of population neuron-ticks | 200,000,000 |
| Steps / simulated duration per population | 40,000 / 60 seconds |
| Probes / trace variables per population | 64 / 32 |
| Total recorded state values | 2,000,000 |
| Estimated trace, final state and refractory storage | 64 MiB |
| Accepted recorded spikes | 4,000,000 |

These limits do not measure total browser heap or guarantee runtime for a dense
or computationally expensive model. The Worker can be terminated with Stop.
Existing Rust validation remains authoritative for types, units, scheduling,
clock alignment, effects and protocol/hash consistency.

The importer parses `model_json` only for metadata and decoding. It **never
re-stringifies it for execution**. Original model and plan strings pass directly
to `BrowserExecutor`, preserving 64-bit seeds/counters that JavaScript Numbers
cannot represent exactly. Outer bundle serialization is safe because those
fields remain strings. Hashes detect drift; they are not publisher signatures.

## Verification

`tests/test_browser_import.py` exports four fixtures from Brian2, executes each
unchanged bundle in Node/WASM and the independent native runner, and checks the
binary and browser-decoded spikes/traces. Fixtures cover delayed connectivity,
multiple clocks with trailing windows, a full-width RNG seed and f32/i64 states.
Malformed bundles, stale model/plan hashes and resource limits are also checked.

`check-import-browser.html` exercises actual File objects, no automatic execution,
multi-population rendering, non-`v` SI traces, tampered-plan rejection, preserved
results, Worker cancellation after progress and successful replay after errors.
The example records 742 total spikes across its two populations.

```sh
python -m pytest brian2-rust/tests/test_browser_import.py \
  brian2-rust/tests/test_browser_equations.py brian2-rust/tests/test_wasm.py \
  brian2-rust/tests/test_browser_scales.py brian2-rust/tests/test_flywire_browser.py -q
```

2026-09-09: 55 tests passed, and the in-app Chrome 152 acceptance page passed.
Browser tests reuse the in-app browser and do not launch standalone Chrome sessions.
