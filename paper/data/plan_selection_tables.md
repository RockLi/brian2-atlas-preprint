# GPU policy-selection and exact-input cache records

Derived from six hash-verified archived reports. Profiling ranges are observed minima and maxima, not confidence intervals.

## All calibration candidates

| Host / case | Candidate | Status | Median (ms) | Min–max (ms) | Selected |
|---|---|---|---:|---:|---|
| M3 quiet | baseline | eligible | 79.634 | 73.337–91.118 | yes |
| M3 quiet | prefix | eligible | 98.412 | 96.730–108.345 | no |
| M3 quiet | bitset | eligible | 68.456 | 68.242–81.432 | no |
| M3 quiet | prefix-bitset | eligible | 71.078 | 64.129–97.370 | no |
| M3 wide | baseline | eligible | 1313.675 | 358.886–1845.094 | yes |
| M3 wide | prefix | eligible | 519.800 | 440.055–912.583 | no |
| M3 wide | bitset | eligible | 293.621 | 292.943–542.793 | no |
| M3 wide | prefix-bitset | eligible | 381.499 | 361.671–519.501 | no |
| L4 quiet | baseline | eligible | 24.077 | 23.927–24.724 | no |
| L4 quiet | prefix | eligible | 26.773 | 26.382–27.017 | no |
| L4 quiet | bitset | eligible | 17.291 | 17.267–17.569 | yes |
| L4 quiet | prefix-bitset | eligible | 19.731 | 19.591–19.782 | no |
| L4 wide | baseline | eligible | 122.258 | 122.052–122.926 | no |
| L4 wide | prefix | eligible | 108.081 | 106.376–108.561 | no |
| L4 wide | bitset | eligible | 82.955 | 82.196–84.181 | no |
| L4 wide | prefix-bitset | eligible | 79.859 | 79.820–84.184 | yes |
| A100 quiet | baseline | eligible | 30.896 | 29.337–33.420 | no |
| A100 quiet | prefix | eligible | 32.863 | 32.346–39.665 | no |
| A100 quiet | bitset | eligible | 20.839 | 20.555–25.267 | yes |
| A100 quiet | prefix-bitset | eligible | 24.828 | 23.493–28.827 | no |
| A100 wide | baseline | eligible | 154.335 | 153.641–155.010 | no |
| A100 wide | prefix | eligible | 124.458 | 124.318–125.206 | no |
| A100 wide | bitset | eligible | 97.873 | 97.208–110.398 | no |
| A100 wide | prefix-bitset | eligible | 91.750 | 91.593–101.296 | yes |

## Later exact-input cache cohort

Activation costs include transport and enabled allocation/compilation reuse, excluding Brian frontend lowering. These are a separate source cohort.

| Host / case | Selected | Calibration + transport (s) | Three hit activations (s) | Hit median (s) | f64 diagnostic |
|---|---|---:|---|---:|---|
| M3 quiet | baseline | 3.707 | 0.571, 0.533, 0.560 | 0.560 | passed |
| M3 wide | bitset | 8.768 | 1.249, 1.213, 1.212 | 1.213 | failed |
| L4 quiet | bitset | 13.759 | 1.727, 1.730, 1.722 | 1.727 | passed |
| L4 wide | prefix-bitset | 23.239 | 3.943, 3.834, 3.832 | 3.834 | failed |
| A100 quiet | baseline | 19.589 | 1.984, 1.967, 1.962 | 1.967 | passed |
| A100 wide | prefix-bitset | 26.255 | 4.203, 4.156, 4.247 | 4.203 | failed |
