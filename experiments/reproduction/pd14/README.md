# Pinned PD14 reproduction

This is a new, bounded execution of the archived PD14 feasibility program with Atlas commit `6677a5bafd3b703ab56b6ed176e9aad70f4638cd`. The original research source stays byte-identical in `../../legacy/brian2-rust/src/bin/pd14.rs`; Cargo consumes the engine directly from the product repository. `Cargo.lock` fixes the complete Rust dependency resolution. No engine implementation is copied into this repository.

Use Python 3.11 or later, Rust/Cargo 1.98.1 and Git. The Git dependency uses SSH, so the running account needs GitHub read access to `RockLi/brian2-atlas`. With a clean, not-yet-created output directory:

```sh
export CARGO_NET_GIT_FETCH_WITH_CLI=true
python experiments/reproduction/pd14/reproduce.py --output output/pd14-smoke
```

Run this command from the repository root. Build outputs default to the requested output directory; `--target-dir /path/to/cargo-target` can reuse an existing Cargo cache. Configure `TMPDIR` and `CARGO_HOME` before running to place scratch files and downloads on another disk. No GPU or Python neuroscience package is needed for this Rust consumer.

The script builds with `--locked`, runs the archived full-scale topology-shape unit test, and executes two fresh processes. Each process performs two identical deterministic repetitions. The bounded simulation uses 1% neuron and indegree scales, 100 ms duration and seed 55, with a 0.25 GiB topology limit. It checks nonempty topology, spikes and delivered events, exact cross-process results, source identities and artifact hashes. Outputs include both raw reports, build/test logs and `validation.json`.

The accepted macOS arm64 run contains 772 neurons, 29,885 synapses and 91 delivered events; spike counts are `[1, 0, 1, 0, 1, 0, 0, 0]`. See [the execution record](../../../migration/pd14-pinned-reproduction.json).

This validates the new pinned build-to-simulation workflow. It does not reproduce full-scale firing-rate results or historical performance measurements. Historical experiments must use their recorded code, inputs and environments; the current Atlas revision cannot replace those identities.
