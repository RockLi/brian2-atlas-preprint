# Brunel full-scale results

Date: 2026-09-06

All Rust runs below used 10,000 excitatory and 2,500 inhibitory neurons,
15,625,000 exact-indegree recurrent synapses, `dt=0.1 ms`, seed 20260906,
four workers, and the AOT backend.

## Figure 8 quantitative reproduction

| Regime | Biological time | Mean rate | Mean ISI-CV | Spectral peak | AOT simulation | End-to-end wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SR `(g=3, eta=2)` | 0.6 s | 315.12 Hz | 0.0005 | 380 Hz | 1.291 s | 3.854 s |
| fast SI `(g=6, eta=4)` | 1.2 s | 60.70 Hz | 0.774 | 170 Hz | 2.910 s | 4.612 s |
| AI `(g=5, eta=2)` | 1.2 s | 37.55 Hz | 0.354 | weak peak | 3.395 s | 5.486 s |
| slow SI `(g=4.5, eta=0.9)` | 1.2 s | 5.07 Hz | 0.228 | 25 Hz | 2.689 s | 4.456 s |

The paper reports 60.7 Hz and 180 Hz for fast SI, 37.7 Hz for AI, and 5.5 Hz
with a 22 Hz population oscillation for slow SI. The Rust results reproduce
those rates and frequencies to the resolution and finite-window variation of
these runs.

The former fast-SI mismatch (74.7 Hz, 280 Hz) exposed a backend bug:
`PoissonInput` writes were not gated by the target state's
`(unless refractory)` rule. External events accumulated during the absolute
refractory interval and advanced the next synchronous wave. Conditional
writes now apply to every relevant population code object while reset remains
unconditional. A reference/AOT/NumPy regression covers this behavior.

At the fast-SI point, Brian2 C++ Standalone with the same exact-indegree rule
produced 61.14 Hz and 175 Hz. Its measured cold frontend/compile/run/load wall
time was 22.65 s on this host. Rust's corresponding end-to-end wall time was
4.61 s. This is a single cold-run comparison; it is not presented as a
general steady-state C++ speedup.

## Broad-delay AR experiment

Holding `g=3` and `eta=2` fixed while replacing `D=1.5 ms` with
`D ~ Uniform(0, 3 ms)` changes the activity from globally synchronized SR to
asynchronous regular firing:

| Backend | Mean rate | Mean ISI-CV | Population Fano, 0.1 ms | End-to-end wall |
| --- | ---: | ---: | ---: | ---: |
| Rust AOT | 319.35 Hz | 0.0451 | 1.029 | 29.04 s |
| C++ Standalone | 319.68 Hz | 0.0447 | 1.065 | 91.35 s |

The agreement is strong across all three state markers. The broad-delay Rust
path is currently much slower than its uniform-delay path because per-edge
delay routing is used for 15.625 million edges. Delay-bucket CSR reuse is the
next performance optimization; the scientific semantics are already covered
by reference/AOT tests.

The rendered comparison is `output/brunel-delay-comparison.png`.

## Artifact reuse

A corrected full-scale 2 by 2 sweep reused one compiled AI artifact:

| `eta` | `g` | Mean rate | AOT simulation |
| ---: | ---: | ---: | ---: |
| 1.8 | 4.8 | 37.28 Hz | 2.166 s |
| 1.8 | 5.2 | 27.77 Hz | 2.106 s |
| 2.2 | 4.8 | 48.96 Hz | 2.449 s |
| 2.2 | 5.2 | 38.25 Hz | 2.428 s |

All four instances shared native source SHA-256
`498c122eb433c50e3ee001c3d224cdc14c9670b201d2461dd3b895e516d470d9`.
The complete sweep, including validated result loading and analysis, took
12.33 s. A 20 by 20 orchestration smoke test at `network_scale=0.01` completed
all 400 artifact-reused points in 5.24 s and rendered the three-panel heatmap.

## Full-scale 20 by 20 phase grid

The scientific grid completed all 400 points at `network_scale=1`, reusing one
native source definition for 480 aggregate seconds of biological time. Total
wall time was 2459.65 s (40 min 59.65 s). The 400 instance files had 400 unique
hashes and shared source SHA-256
`498c122eb433c50e3ee001c3d224cdc14c9670b201d2461dd3b895e516d470d9`.

Across the grid, mean rates ranged from 0 to 417.22 Hz, mean ISI-CV from
approximately 0 to 0.897, and 0.1 ms population-count Fano factors from 0 to
7083.91. The final figure uses `log10(1 + Fano)` so both the synchronized and
irregular regions remain visible.

The first attempt retained every full `results.bin` until the end and filled
the temporary volume at point 150. The sweep now removes each point dump and
instance immediately after extracting statistics, keeping temporary storage
bounded. The completed rerun validates this cleanup over all 400 points.
