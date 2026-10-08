# Full MNIST comparison

Current handoff: [final experiment and delivery report](../../FLYWIRE_MNIST_FINAL_REPORT.md). The optimized CPU full-test result is 87.14%; historical weak-input results below remain unchanged.

The Mac Studio campaign completed all 60,000 training and 10,000 official test
images in 13,102.65 seconds (3 h 38 min). The fixed FlyWire network plus ridge
readout scored **23.64%** on the official test (validation: 23.99%), versus
84.67% for the actual projected-spike control and 86.06% for pixel ridge.
These results do not show a classification benefit from this network/configuration.
The [2026-09-10 review](../../review-evidence/flywire-mnist-20260910/REVIEW.md)
documents the skipped input-calibration gate, feature/activity diagnostics,
independent Brian2 check and follow-up comparisons. The original test result
is preserved; post-hoc diagnostics use only original fit/validation data.

The traditional CNN baseline has completed: **98.80% (9,880 / 10,000)** on the
official test set, with a Wilson 95% interval of 98.57–99.00%. It selected epoch
7 out of a fixed ten-epoch budget using 50,000 fit / 10,000 validation examples,
then reinitialized and trained on all 60,000 official training examples for
seven epochs. The test set was opened only after model selection and refitting
were locked. The experiment took 97.61 seconds on the local CPU with two PyTorch
threads. Artifacts and the exact architecture are in [CNN evidence](evidence/cnn-full/report.json).
The [learning curve](evidence/cnn-full/training-curve.png) and
[epoch history](evidence/cnn-full/training-history.json) record the selection run.

This is an 80,202-parameter conventional CNN: two 5×5 convolutions (16 and 32
channels), ReLU and 2×2 max pooling, then fully connected layers of 128 and 10
units. Input scaling is `/255`; there is no data augmentation. The conventional
training/evaluation approach is also illustrated by the
[official PyTorch MNIST example](https://github.com/pytorch/examples/blob/main/mnist/main.py).
Our architecture, optimizer and split are explicitly recorded in the saved
protocol rather than inferred from that example.

CNN trains all of its layers. The FlyWire experiment preserves all internal
conductance-LIF weights and trains a linear activity readout. Comparing their
test accuracies measures performance under these specified training protocols;
it does not isolate the scientific value of the biological topology. Random
rewiring, no-recurrence ablation and multiple seeds remain separate research
controls. The full-dataset result evaluates this uncalibrated configuration,
not the achievable accuracy of a calibrated or internally trained FlyWire model.

## Shared data contract

- The official training set is deterministically split into 50,000 fit and
  10,000 validation examples using seed 783, shared with the CNN experiment.
- FlyWire extracts fixed features for all 60,000 once. Four regularization
  strengths are selected using fit/validation only. The selected readout is
  refitted on all 60,000 before evaluation on all 10,000 official test examples.
- Pixel-only and actual projected-spike input controls use the same split and
  regularization budget. Projected count archives store integer counts before
  fanout scaling; per-feature standardization makes that constant scaling
  irrelevant to the ridge fit.
- The full runner accepts only the CPU-f64 path in this milestone. Existing
  Metal/CUDA and MPI development adapters remain available separately. They
  cannot silently replace the numeric profile of cached CPU features.

## Full-run commands

Build the repo extensions and AOT runner as in [README](README.md), and install
`threadpoolctl` for bounded BLAS fitting. CNN additionally requires PyTorch.

```sh
export PYTHONPATH="$PWD/brian2-rust/python:$PWD/brian2-rust/experiments:$PWD"
python -m flywire_mnist.cnn --data /path/to/mnist --output /path/to/cnn-run \
  --epochs 10 --threads 2
python -m flywire_mnist.campaign \
  --data /path/to/mnist --graph /path/to/flywire-v783-ei-sensory \
  --output /path/to/flywire-full --workers 12 --fit-threads 8 --shard-size 32
```

The campaign executes `init`, `extract --phase train`, `fit`,
`extract --phase test`, then `evaluate`. For a finite throughput probe, run those
stages separately and pass `--max-shards N` to `extract`; completed probe shards
are reused by the full campaign. Use a new directory for a new protocol, and
the same campaign command to resume interrupted work under unchanged sources.

Each CPU worker starts from the same frozen initial state for every image.
The native AOT `--spike-input POPULATION SCHEDULE.bin` sidecar changes only one
SpikeGenerator schedule. Both Python and Rust check event bounds, ordering and
duplicates; the native artifact rejects targeting a non-generator population.
The snapshot and executable hashes are checked, and a diagnostic full-instance
writer proves byte-level equivalence to the ordinary validated instance API.
This avoids copying roughly 190 MB of graph/initial-state data to a new file
for every image. The simulator still reloads its immutable state and graph;
there is no unverified checkpoint or state carry-over between samples.

Workers write independent atomic compressed shards. Resume checks protocol,
sample order, dtype, dimensions and payload checksums. Counts must fit uint16
or extraction fails. Output readouts are hashed into a test lock only after all
60,000 training features are present. Test extraction refuses an absent,
incomplete or mismatched lock. A final report needs all 10,000 test features.
Only one coordinator can own an output directory; OS locks release on a crash.

Inspect `progress.json` for completed images and measured throughput,
`*-campaign.json` for process/stage status, and `*-campaign.log` for execution
logs. Worker initial snapshots are temporary and removed between completed
stages. An interrupted shard is recomputed; completed shards are retained.
The pipeline does not install services or recurring scheduled tasks.

## Deployment target

User-selected host: `rock@100.90.28.27` (Mac Studio, Mac13,2, 20 CPUs, 128 GiB
RAM). Dedicated root: `/atlas-home/0004/workspace/flywire-mnist-20260909`.
Use the native ARM64 `/opt/homebrew/bin/python3.12`, not the host's default
x86-64 Python. Source, environment, public data, and runs are kept inside that
root. Select final worker count from measured throughput with CPU headroom;
do not extrapolate the local 10-image run into a guaranteed completion time.

### Deployed full campaign — 2026-09-09

The full CPU-f64 campaign completed using frozen source commit `f8d7b23a6`.
It uses 16 independent CPU workers (sample parallelism, not MPI), with BLAS
limited to one thread during extraction and eight during readout fitting.
The source archive SHA-256 is
`37de777c22a8a6f011d8468a78a29a4c18ce773c2edc3dc89747876a35359d65`.
Python 3.12 and all compiled extensions run natively on ARM64. Dependency
wheels and locked Cargo sources are staged under the dedicated root for
offline rebuilding. The remote regression subset passed 9 tests with 2 skips;
10 full-graph feature vectors exactly matched the original CPU run.

| CPU workers | Newly extracted images | Wall time | Images/second |
| --- | ---: | ---: | ---: |
| 4 | 128 | 84.88 s | 1.51 |
| 8 | 256 | 75.85 s | 3.38 |
| 16 | 512 | 87.20 s | 5.87 |

All 896 probe images are reused. The full campaign was launched independently
of the SSH session with `nohup`, reduced scheduling priority (`nice -n 5`), and
job-scoped idle-sleep prevention (`caffeinate -i`). Coordinator PID at launch:
`94240`. The remaining extraction is estimated at roughly 3–4 hours from the
probe, subject to sustained throughput and machine load; the actual completed
campaign took 3 h 38 min. Its final FlyWire test accuracy is 23.64%. Deployment evidence
is a dated snapshot in [mac-studio-deployment.json](evidence/mac-studio-deployment.json).

On the Mac Studio, run output is
`/atlas-home/0004/workspace/flywire-mnist-20260909/runs/full-cpu`.
Inspect it remotely with:

```sh
ssh rock@100.90.28.27 'cat /atlas-home/0004/workspace/flywire-mnist-20260909/runs/full-cpu/progress.json'
ssh rock@100.90.28.27 'cat /atlas-home/0004/workspace/flywire-mnist-20260909/runs/full-cpu-campaign.json'
```

The coordinator automatically proceeds through all 60,000 training images,
readout selection/refitting, and all 10,000 official test images. A successful
run produces `runs/full-cpu/report.json` with FlyWire, projected-input and
pixel-control test results. To resume after an interruption, use the same
environment and command (the coordinator lock rejects overlapping launches):

```sh
cd /atlas-home/0004/workspace/flywire-mnist-20260909/source
export LC_ALL=C LANG=C
export PYTHONPATH=brian2-rust/python:brian2-rust/experiments:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
nohup nice -n 5 /usr/bin/caffeinate -i ../.venv/bin/python \
  -m flywire_mnist.campaign --data ../data/mnist --graph ../data/flywire \
  --output ../runs/full-cpu --workers 16 --fit-threads 8 --shard-size 32 \
  >> ../runs/full-cpu-launch.log 2>&1 < /dev/null &
```

Keep the deployed Python sources and model artifacts unchanged while resuming:
their hashes are part of the experiment protocol.
