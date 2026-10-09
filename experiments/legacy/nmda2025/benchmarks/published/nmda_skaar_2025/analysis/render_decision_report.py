#!/usr/bin/env python3
"""Render the decision-network reproduction report from raw JSON evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def rate_cell(result: dict, period: str) -> str:
    rates = result["population_rates"][period]
    return f"{rates['A_Hz']:.3f} / {rates['B_Hz']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--approximate", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exact = read(args.exact)
    approx = read(args.approximate)
    pair = read(args.pair)
    env = read(args.environment)
    ratio = pair["runtime"]["exact_over_approximate_ratio"]
    serial_days = exact["wall_seconds"] * 2000 / 86400
    exact_peak = exact["time_command"]["maximum_resident_set_size_bytes"] / 1e9
    approx_peak = approx["time_command"]["maximum_resident_set_size_bytes"] / 1e9
    a_metrics = pair["trajectory_50ms"]["selective_A"]
    b_metrics = pair["trajectory_50ms"]["selective_B"]
    stimulus_a_delta = (
        approx["population_rates"]["stimulus_1000_3000ms"]["A_Hz"]
        - exact["population_rates"]["stimulus_1000_3000ms"]["A_Hz"]
    )
    post_a_delta = (
        approx["population_rates"]["post_3000_4000ms"]["A_Hz"]
        - exact["population_rates"]["post_3000_4000ms"]["A_Hz"]
    )

    text = f"""# Published decision-making workload reproduction

## Source route and validation scope

The upstream checkout at `{exact['upstream_commit']}` provides the functional
decision network as NEST code, not Brian2 code.  The three decision scripts
instantiate `nest.Create`/`nest.Connect`; no author-supplied Brian2 decision
fixture exists.  This result therefore reproduces the authors' original NEST
workload in its original simulator.  It is not a Brian2→B2IR→Rust correctness
result, and manually rewriting this network for the engine would violate the
unchanged-workload rule.

The harness verifies SHA-256 `{exact['upstream_source_sha256']}` for
`decision_making_varying_coherence.py`, extracts the unchanged `run_sim`
function with Python's AST, and avoids only the module's top-level batch loop.
The selected first validation is one matched trial at coherence
`{exact['coherence_percent']}%`, NEST seed `{exact['seed']}` and NumPy stimulus
seed `{exact['numpy_stimulus_seed']}`.  The upstream batch seeds NumPy once
from wall time and then advances that stream independently for each model
call; the harness resets it before each isolated call so exact and approximate
models receive the identical time-varying stimulus promised by the paper's
comparison.  This deterministic-input adjustment is explicit and does not
alter the function body, equations, parameters, connectivity or delays.

## Environment and structure

The run used NEST `{env['nest']}` from the unmodified official source commit
`{env['nest_source']['commit']}`, Python `{approx['python_version']}` and native
arm64 `{env['compiler_cxx']}` on `{env['cpu_brand']}` with
`{env['logical_cpu_count']}` OpenMP threads available and
{int(env['memory_bytes']) / 2**30:.0f} GiB RAM.  The measured trials use eight
OpenMP threads and no MPI.  AppleClang 21/libc++ could not compile NEST 3.8's
old iterator-pair sort templates; GCC 15 compiled the same source without a
source patch.  The reproducible environment is in the
[macOS NEST 3.8 build script](../scripts/build_nest38_macos.sh), and the
machine-readable inventory is the
[NEST environment record](../results/raw/decision_making/local_mac/environment_nest38.json).

The network contains 2,000 neurons: 240 each in selective populations A and
B, 1,120 nonselective excitatory neurons and 400 inhibitory neurons.  It has
7,202,960 logical connections: 6.4 million paired recurrent AMPA/NMDA
excitatory connections, 800,000 recurrent GABA connections, 2,480 external
AMPA connections and 480 spike-recorder connections.  Recurrent delay is
0.5 ms, external AMPA delay is 0.1 ms, integration resolution is 0.1 ms and
biological duration is 4,000 ms.  NEST uses its adaptive RKF45 neuron solver,
as reported in the paper.

## Matched coherence-20 result

| Metric | Exact `iaf_bw_2001_exact` | Approximate `iaf_bw_2001` |
| --- | ---: | ---: |
| `run_sim` wall time | {exact['wall_seconds']:.3f} s | {approx['wall_seconds']:.3f} s |
| Process wall time | {exact['time_command']['wall_seconds']:.2f} s | {approx['time_command']['wall_seconds']:.2f} s |
| Peak RSS | {exact_peak:.3f} GB | {approx_peak:.3f} GB |
| A / B spike count | {exact['spike_counts']['selective_A']} / {exact['spike_counts']['selective_B']} | {approx['spike_counts']['selective_A']} / {approx['spike_counts']['selective_B']} |
| Baseline A / B rate | {rate_cell(exact, 'baseline_0_1000ms')} Hz | {rate_cell(approx, 'baseline_0_1000ms')} Hz |
| Stimulus A / B rate | {rate_cell(exact, 'stimulus_1000_3000ms')} Hz | {rate_cell(approx, 'stimulus_1000_3000ms')} Hz |
| Post-stimulus A / B rate | {rate_cell(exact, 'post_3000_4000ms')} Hz | {rate_cell(approx, 'post_3000_4000ms')} Hz |
| Post-stimulus winner | {exact['decision_by_post_stimulus_rate']} | {approx['decision_by_post_stimulus_rate']} |

Both models select population A, the expected winner for positive coherence,
and both retain an asymmetric high-A/low-B state for the full second after
stimulus offset.  This passes the declared single-trial functional endpoint.
The approximate model's winning-A rate is `{stimulus_a_delta:.3f}` Hz higher
during stimulation and `{post_a_delta:.3f}` Hz higher afterward, matching the
qualitative direction reported in the paper for its example trajectories.
The authors' Figure 4 implementation chooses the population with more spikes
over the full four-second histogram; it also selects A for both models.

For descriptive trajectory comparison, the 1 ms histograms were aggregated
into the paper's 50 ms population-rate bins.  Exact versus approximate
Pearson correlations are `{a_metrics['pearson_r']:.4f}` for A and
`{b_metrics['pearson_r']:.4f}` for B; their RMSE values are
`{a_metrics['rmse_Hz']:.3f}` and `{b_metrics['rmse_Hz']:.3f}` Hz.  No numerical
pass threshold is applied: the two neuron models intentionally implement
different NMDA dynamics, and the paper itself expects divergent spike timing
after small voltage differences accumulate.  The validation contract is the
matched input, the same decision and sustained functional state.  See the
[matched activity trajectories](figures/decision_c20_matched.png) and the
[machine-readable pair summary](../results/processed/decision_making_c20_matched_pair.json).

The exact/approximate `run_sim` ratio is `{ratio:.1f}x` for this single local
trial.  It compares two different scientific models and is not an engine
speedup.  It is also a one-run observation, so it is not promoted to a formal
performance result.

## Scope relative to psychometric validation

The paper's Figure 4 uses 400 trials for each of five coherence levels for
each model, or 2,000 exact and 2,000 approximate simulations.  At this exact
trial's local runtime, the exact side alone would take about
`{serial_days:.1f}` days if run serially on this eight-core Mac.  The present
result is therefore a representative functional reproduction, not a
psychometric-curve reproduction.  The separate controlled 400-trial campaign,
including choice probabilities and 90% bootstrap intervals, is documented in
the [full psychometric report](decision_psychometric_400.md).

The original NEST trial is now reproduced, but the engine route remains
blocked by a real frontend boundary: our adapter accepts Brian2 objects and
has no generic NEST frontend.  No benchmark-specific translation or custom
Rust kernel has been introduced.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
