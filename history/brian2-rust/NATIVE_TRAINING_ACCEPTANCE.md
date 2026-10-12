# Native training acceptance — 2026-10-12

The original stochastic-equation and dynamic-synapse goal is complete within
its documented Brian frontend contract: all 21 requirements and 11 acceptance
gates passed. Arbitrary Python execution and every possible Brian semantic are
not implied. Cross-host / multiple physical GPU acceptance remains deferred by
the user.

Implemented coverage includes Euler / stochastic Heun / Milstein, multiple
state and parameter banks, simulation time and refractory guards, reproducible
addressed random streams, continuous and event-driven synaptic state, ordered
pre/post pathways, STDP/STP, delays/history/summed variables, complete and
truncated BPTT, explicit surrogate and detach policies, optimizer/checkpoint
continuation, and atomic read-only/failure behavior. CPU, Metal, CUDA and local
MPI have independent reference/VJP and restore coverage.

## Accepted results

| Validation | Passed | Additional scope |
| --- | ---: | --- |
| CPU full module scope | 12,940 | 21,814 authenticated hardware skips |
| Metal full module scope | 23,877 | 10,877 authenticated CUDA skips |
| CUDA physical L4 | 236 distinct cases | all 175 original named CUDA parameters |
| Compiler / CUDA ABI checks | 21 | real compilation and exported capabilities |
| Rust library | 51 | actual test execution |

CPU and Metal each collected the same 34,754 ordered identities across 272
modules, including six empty helper modules. Hardware skips are not counted as
passes. These are recorded executions, not a new full test run at commit time.

Final records:

- [Goal and historical implementation notes](NATIVE_TRAINING_STOCHASTIC_SYNAPSE_GOAL.md)
- [All completion gates and requirement mapping](mpi-evidence/training-goal-audit-20261010/completion-gates-v9-modules.json)
- [CPU receipt](mpi-evidence/training-module-scope-v9-20261012/cpu-accepted.json)
- [Metal receipt](mpi-evidence/training-module-scope-v9-20261012/metal-accepted.json)
- [Named CUDA acceptance](mpi-evidence/training-requirement-cuda-v5-20261010/named-CUDA-accepted.json)
- [Closure receipt](mpi-evidence/training-module-scope-v9-20261012/closure/receipt.json)

The original outer controller exit is unknown. Every original module result
has its recorded actual child exit; a fresh controller re-audit and completion
auditor both exited 0, and all 4,352 original test records remained unchanged.
This distinction is retained in the closure receipt.

The commit contains final reports, source/hash manifests and acceptance
controls. Historical intermediate evidence, raw per-module logs, cloud build
copies, local native binaries and caches remain in the local evidence archive;
the reports contain their recorded paths and hashes. Historical audit scripts
therefore require that archive and its original environment, rather than being
portable test entrypoints. The final snapshot also records shared workspace
files from separate experiments; their unrelated resource-budget/cache changes
are excluded from this training commit.

For commit hygiene, trailing whitespace in two Python files was removed after
acceptance, with identical parsed Python ASTs. [The before/after SHA record](mpi-evidence/training-commit-20261012/whitespace-equivalence.json)
preserves this difference; the original acceptance hashes are not rewritten.

## Build and run

See [NATIVE_TRAINING.md](NATIVE_TRAINING.md) for the native build, examples,
backend prerequisites and checkpoint format. From the repository root, with
Python dependencies installed (including a local build of the Brian Cython
extensions, e.g. `python -m pip install -e .`) and Rust/Cargo available:

```sh
CARGO_TARGET_DIR="$PWD/brian2-rust/target-native-training" \
  cargo build --release --locked --manifest-path brian2-rust/Cargo.toml --bin b2-train
export B2_TRAIN_RUNNER="$PWD/brian2-rust/target-native-training/release/b2-train"
export PYTHONPATH="$PWD/brian2-rust/python:$PWD"
python -m pytest -q brian2-rust/tests/test_native_training.py
python -m pytest -q brian2-rust/tests/test_training_stochastic.py
python -m pytest -q brian2-rust/tests/test_training_dynamic.py
```

Hardware tests require their explicit backend switches and genuine devices;
missing hardware must not be reported as successful GPU validation. Cloud
runners are explicit entrypoints and are never started by this build procedure.

## Isolated staged commit check

After excluding unrelated workspace changes, the staged checkout compiled its
Brian Cython extensions and native Rust binaries independently. All 51 Rust
library tests, 137 Python regression cases and two genuine local two-rank MPI
cases passed (39 hardware-dependent regression cases skipped). The MPI checks
include a fresh-process checkpoint restore.

See [precommit verification](mpi-evidence/training-commit-20261012/precommit-verification.json)
and its adjacent actual process/XML receipts. The initial large debug build ran
out of local disk space; removing only this check's temporary copies and using
a compact build resolved it. No new cloud run was required.
