# GPU comparator accounting

Generated from retained cohort reports. All times are complete replay medians in milliseconds; failed qualifications have no ranked time. Modified adapters remain explicitly labeled.

| Host | Ring neurons | Worker | Median (ms) | Qualification |
|---|---:|---|---:|---|
| M3 | 4096 | Rust CPU f64 (serial slot) | 37.35 | passed matched gate |
| M3 | 4096 | Compiled f32 expression control | 66.30 | passed matched gate |
| M3 | 4096 | metal | 34.37 | passed matched gate |
| M3 | 4096 | metal-prefix | 43.55 | passed matched gate |
| M3 | 4096 | metal-bitset | 36.78 | passed matched gate |
| M3 | 4096 | metal-prefix-bitset | 39.74 | passed matched gate |
| M3 | 16384 | Rust CPU f64 (serial slot) | 85.73 | passed matched gate |
| M3 | 16384 | Compiled f32 expression control | 267.51 | passed matched gate |
| M3 | 16384 | metal | 67.19 | passed matched gate |
| M3 | 16384 | metal-prefix | 74.64 | passed matched gate |
| M3 | 16384 | metal-bitset | 67.11 | passed matched gate |
| M3 | 16384 | metal-prefix-bitset | 79.36 | passed matched gate |
| L4 | 4096 | Rust CPU f64 (serial slot) | 60.18 | passed matched gate |
| L4 | 4096 | Compiled f32 expression control | 191.51 | passed matched gate |
| L4 | 4096 | cuda | 13.16 | passed matched gate |
| L4 | 4096 | cuda-prefix | 13.45 | passed matched gate |
| L4 | 4096 | brian2cuda | 608.84 | passed matched gate |
| L4 | 4096 | Direct GeNN original ring adapter | 275.98 | passed matched gate |
| L4 | 4096 | GeNN barrier variant | 276.30 | passed matched gate |
| L4 | 4096 | GeNN barrier + gather variant | 173.77 | passed matched gate |
| L4 | 4096 | Brian2GeNN schedule-corrected variant | 509.72 | passed matched gate |
| L4 | 4096 | Brian2GeNN f32-factor variant | 498.68 | passed matched gate |
| L4 | 4096 | cuda-bitset | 13.49 | passed matched gate |
| L4 | 4096 | cuda-prefix-bitset | 13.73 | passed matched gate |
| L4 | 16384 | Rust CPU f64 (serial slot) | 216.02 | passed matched gate |
| L4 | 16384 | Compiled f32 expression control | 826.66 | passed matched gate |
| L4 | 16384 | cuda | 34.82 | passed matched gate |
| L4 | 16384 | cuda-prefix | 35.47 | passed matched gate |
| L4 | 16384 | brian2cuda | 1701.10 | passed matched gate |
| L4 | 16384 | Direct GeNN original ring adapter | — | excluded: failed/incomplete qualification |
| L4 | 16384 | GeNN barrier variant | 1035.31 | passed matched gate |
| L4 | 16384 | GeNN barrier + gather variant | 622.10 | passed matched gate |
| L4 | 16384 | Brian2GeNN schedule-corrected variant | 1205.45 | passed matched gate |
| L4 | 16384 | Brian2GeNN f32-factor variant | 1184.50 | passed matched gate |
| L4 | 16384 | cuda-bitset | 35.40 | passed matched gate |
| L4 | 16384 | cuda-prefix-bitset | 36.04 | passed matched gate |
| A100 | 4096 | Rust CPU f64 (serial slot) | 55.69 | passed matched gate |
| A100 | 4096 | Compiled f32 expression control | 112.23 | passed matched gate |
| A100 | 4096 | cuda | 12.89 | passed matched gate |
| A100 | 4096 | cuda-prefix | 11.14 | passed matched gate |
| A100 | 4096 | brian2cuda | 983.39 | passed matched gate |
| A100 | 4096 | Direct GeNN original ring adapter | 261.24 | passed matched gate |
| A100 | 4096 | GeNN barrier variant | 258.99 | passed matched gate |
| A100 | 4096 | GeNN barrier + gather variant | 190.55 | passed matched gate |
| A100 | 4096 | Brian2GeNN schedule-corrected variant | 601.41 | passed matched gate |
| A100 | 4096 | Brian2GeNN f32-factor variant | 569.61 | passed matched gate |
| A100 | 4096 | cuda-bitset | 11.91 | passed matched gate |
| A100 | 4096 | cuda-prefix-bitset | 11.82 | passed matched gate |
| A100 | 16384 | Rust CPU f64 (serial slot) | 181.80 | passed matched gate |
| A100 | 16384 | Compiled f32 expression control | 449.68 | passed matched gate |
| A100 | 16384 | cuda | 32.58 | passed matched gate |
| A100 | 16384 | cuda-prefix | 32.43 | passed matched gate |
| A100 | 16384 | brian2cuda | 3385.01 | passed matched gate |
| A100 | 16384 | Direct GeNN original ring adapter | — | excluded: failed/incomplete qualification |
| A100 | 16384 | GeNN barrier variant | 1011.78 | passed matched gate |
| A100 | 16384 | GeNN barrier + gather variant | 730.59 | passed matched gate |
| A100 | 16384 | Brian2GeNN schedule-corrected variant | 1008.61 | passed matched gate |
| A100 | 16384 | Brian2GeNN f32-factor variant | 1036.75 | passed matched gate |
| A100 | 16384 | cuda-bitset | 30.93 | passed matched gate |
| A100 | 16384 | cuda-prefix-bitset | 31.89 | passed matched gate |

## Recurrent CUBA

Direct GeNN here uses the disclosed Brian-Euler variant. Matched f32 qualification does not imply f64 equivalence.

| Host | Worker | Median (ms) | Min–max (ms) |
|---|---|---:|---|
| L4 | rust-f64 | 58.92 | 57.13–64.96 |
| L4 | cpu-f32 | 562.43 | 557.13–591.68 |
| L4 | cuda | 116.79 | 115.80–129.42 |
| L4 | brian2cuda | 604.28 | 597.13–710.01 |
| L4 | brian2genn | 411.15 | 358.74–426.22 |
| L4 | Direct GeNN Brian-Euler variant | 52.79 | 51.36–61.05 |
| A100 | rust-f64 | 48.46 | 47.20–49.80 |
| A100 | cpu-f32 | 368.51 | 368.34–372.89 |
| A100 | cuda | 137.13 | 135.24–167.77 |
| A100 | brian2cuda | 682.62 | 660.81–763.23 |
| A100 | brian2genn | 447.80 | 401.51–469.89 |
| A100 | Direct GeNN Brian-Euler variant | 52.11 | 50.52–59.63 |
