# Official Brian2 example compatibility scan

This report records the bounded scan run on 2026-09-29 against the 100 Python
files under `examples/`.  It distinguishes backend compatibility from external
data, multiprocessing wrappers, long-running benchmarks, and analysis code that
cannot consume deliberately shortened simulations.

## Method

`tools/scan_official_examples.py` applies four layers:

1. compile every Python source file;
2. statically identify missing imports;
3. force `rust_standalone`/reference and validate the first network with
   `run(0*ms)` in an isolated process;
4. for build-pass scripts, cap each run to 1 ms and stop after at most 20 runs.

Each probe has a process-group timeout.  Generated Rust artifacts are removed
after each probe, while the JSON report is written atomically after every
result.  This keeps the scan restartable and bounded in both time and disk.

Command used:

```console
MPLCONFIGDIR=/tmp/mpl-diehl PYTHONPATH=brian2-rust/python \
  .venv/bin/python brian2-rust/tools/scan_official_examples.py \
  --timeout 90 --workers 2 --smoke-ms 1 --max-runs 20 \
  --output /tmp/brian2-official-example-scan-20260929.json
```

## Raw bounded-scan result

| Layer | Result |
| --- | --- |
| Syntax | 100 passed, 0 failed |
| Statically missing dependencies | 0 examples |
| Rust zero-duration build | 86 passed, 8 did not call a network, 3 wrapper/resource failures, 3 timeouts |
| Short-run smoke | 78 passed, 8 post-analysis failures, 14 not run |
| Confirmed Rust semantic failures | **0** |

The eight smoke failures all occurred after a successful Rust run.  Seven were
expected consequences of replacing a long simulation by 1 ms: fixed-index
plotting read beyond the shortened monitor, time vectors retained their
original length, empty spike arrays reached reductions, or plotting saw
non-finite ranges.  The remaining failure exposed NumPy 2 rejecting a generator
passed to its exported ``sum``; the benchmark now explicitly uses Python's
built-in ``sum``.  All eight were subsequently qualified as follows:

- `advanced/float_32_64_benchmark.py`: all 36 one-second simulations and final
  plotting completed in 182 seconds after the NumPy 2 correction.
- `advanced/modelfitting_sbi.py`: a two-parameter, full 350 ms simulation
  produced a `(2, 7000)` voltage trace and finite `(2, 4)` summary statistics.
  The external 10,000-sample SNPE training remains a separate ML workload.
- `compartmental/hodgkin_huxley_1952.py`: all three segments (153 ms total),
  cable solve, monitor indexing, and plotting completed.
- `frompapers/Brunel_Wang_2001.py`: the full 1,000-neuron/four-second AOT run
  completed in 33 seconds, including spike/rate post-processing.
- `frompapers/Spreizer_et_al_2019/Spreizer_et_al_2019.py`: all four landscapes,
  12 run segments, rate histograms, and animation data completed in 113 seconds.
- `frompapers/Wang_2002.py`: the full 2,000-neuron/four-second AOT run completed
  in eight seconds, including profiling and monitor post-processing.
- `synapses/homeostatic_stdp_at_inhibitory_synapes.py`: the complete five-second
  linked-variable/event-driven STDP protocol and analysis completed.
- `synapses/spike_based_homeostasis.py`: the complete 40-second Poisson,
  ``on_post``, StateMonitor, and PopulationRateMonitor run completed.

Forty-two smoke-pass scripts now run their complete default duration as
automated regressions.  They cover single and segmented runs, `stop()`, SDEs,
custom events, independent monitor clocks, compartmental/SpatialNeuron solves,
standalone build/replay, GSL method aliases, and neuron/synapse protocols from
20 ms through 10 seconds.  The first complete profiling reduction in the float
benchmark, the 350 ms SBI core, and reduced Gaussian/Graupner/Maass end-to-end
paths are also automated.  Larger AOT and topology runs remain recorded
qualification checks rather than default unit tests.

This expansion exposed one real scalability bug in
`advanced/compare_GSL_to_conventional.py`: physical StateMonitor budgeting and
preallocation used the source population's increasingly fine clock instead of
the monitor's independent 100 us clock.  The validator and executor now use
the number of samples the monitor schedule can actually emit.  The example can
therefore complete its adaptive GSL versus successively refined fixed-step
comparison without weakening the 100-million-value output budget.

The eight `no_network` files are helpers, construction/visualisation examples,
or parent processes whose simulations run in workers:

- `advanced/exprel_function.py`
- `compartmental/morphotest.py`
- `frompapers/Brette_2012/params.py`
- `frompapers/Stimberg_et_al_2018/plot_utils.py`
- `multiprocessing/01_using_cython.py`
- `multiprocessing/03_standalone_joblib.py`
- `synapses/spatial_connections.py`
- `synapses/state_variables.py`

## Adjudication of the six build non-passes

- `advanced/opencv_movie.py` initially attempted its default network download.
  A later authorized online check downloaded the official 1.93 MiB AVI
  (`SHA-256 78884f64b564a3b06dc6ee731ed33b60c6d8cd864cea07f21d94ba0f90c7b310`).
  Its container reports 1400 frames, while
  OpenCV 4.13 decodes 1179; the example now counts decodable frames instead of
  reading past EOF.  The complete 168,960-neuron, 38.907-second biological-time
  Rust run finished successfully in 9m30s.  The synthetic-video regression also
  deliberately over-reports its frame count and passes.  A subsequent macOS
  hardware check opened camera 0 at 1920x1080/15 fps, resized three frames to
  64x36, ran the Rust model, and displayed the resulting spike animation.  Its
  pause/resume, restart, and close controls were exercised without warnings;
  a bounded fake-camera regression covers the same streaming path in CI.
- `frompapers/Diehl_Cook_2015.py` defaulted to test mode without staged data.
  Its checked MNIST train/observe/test Rust integration test passes.
- `multiprocessing/02_using_standalone.py` starts fresh worker interpreters,
  outside the in-process monkeypatch used by the generic scanner.  Its dedicated
  Rust multiprocessing integration test passes.
- `frompapers/Graupner_Brunel_2012.py` launches 41 multiprocessing jobs.  The
  complete default protocol ran with eight Rust AOT workers: every point used
  1,000 DOWN and 1,000 UP synapses for 60 seconds of biological time, all 41
  points and the final STDP analysis completed in 269 seconds.  Environment
  overrides for point/repetition counts now support a three-point, four-repeat
  full-duration regression without changing the official defaults.
- `frompapers/Maass_Natschlaeger_Markram_2002.py` performs expensive stimulus
  collection and multiprocessing before its first run.  Its original nested
  worker closure and implicit global neuron reference were made safe for the
  Python 3.14 macOS multiprocessing model.  The complete default protocol then
  finished all 1,600 restore/replay simulations (200 pairs at four distances,
  two stimuli each), with 135 neurons and 500 ms per simulation, on eight Rust
  AOT workers in 807 seconds.  The new opt-in `retain_run_artifacts=False`
  device mode retained only the latest successful artifact per worker (about
  1.45 MiB each) instead of exhausting disk space with 1,600 artifacts.  A
  reduced two-worker end-to-end replay is automated.
- `synapses/efficient_gaussian_connectivity.py` is a topology performance
  benchmark, not a simulation.  The unmodified benchmark sizes completed all
  naive, range-generator, and exact split-sample variants through `N=20,000`
  in 50 seconds; the largest naive case evaluated 400 million candidate pairs.
  The example is now import-safe, avoids unused-object warnings, and has a
  reduced three-variant construction regression.

The focused optional-example, Diehl-Cook, and multiprocessing suites passed
`6/6` after the scan.

## Interpretation

This scan finds no new backend semantic blocker in the official example set.
It is deliberately not a claim that every paper simulation was run at its
published duration or reproduced its scientific figures.  Long-duration
accuracy/statistical validation and platform-specific GPU/MPI execution remain
separate qualification work.
