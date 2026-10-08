# Full-scale PD14 on a 16 GB Apple Silicon laptop

Date: 2026-09-05

This report tests whether the unified Brian2 model can construct and execute the
full Potjans–Diesmann 2014 DC-input network on a memory-constrained developer
machine. It is a feasibility and resource test, not a speed comparison: the
Brian2 C++ standalone path could not reach compilation within the physical-memory
budget, whereas the Rust Device completed the full workload.

## Environment and workload

| Item | Value |
| --- | --- |
| Host | MacBook Air (Mac15,12), Apple M3 |
| CPU | 8 cores: 4 performance + 4 efficiency |
| Memory | 16 GB unified memory |
| OS | macOS 26.6.2, native arm64 |
| Brian2 | 2.10.1.post199 |
| Rust | rustc 1.98.1, LLVM 22.1.8, `aarch64-apple-darwin` |
| Model | 77,169 neurons, 55 non-empty projections, 298,880,968 synapses |
| Time step | 0.1 ms |
| Recording | disabled, to isolate simulation and topology resources |
| Rust execution | 4 workers |

Both attempts were launched through `examples/pd14_device.py`. The Rust path
used the procedural fixed-total topology and procedural weight/delay initializers
carried by B2IR. The C++ path used the same logical fixed-total network, which
requires endpoint and parameter arrays to be materialised by the Brian2 Python
frontend before the standalone project can be compiled.

## Rust result: completed

The Rust Device completed both a 100 ms smoke run and the full 10 s simulation.

| Measure | 100 ms smoke | 10 s full run |
| --- | ---: | ---: |
| Simulation body | 1.813 s | 104.009 s |
| End-to-end frontend/build/run/load | 23.248 s | 125.586 s |
| Python/frontend peak RSS | 0.206 GB | 0.319 GB |
| Simulation child peak RSS | 3.791 GB | 6.314 GB |
| Delivered synaptic events | 91,084,720 | 9,468,254,412 |
| Delivered-event throughput | 50.23 million/s | 91.03 million/s |
| Generated output directory | 14 MB | 51 MB |

The full run therefore fits with substantial physical-memory headroom and does
not depend on swap for its topology representation. Its simulation body runs at
about 10.40 seconds of wall time per second of model time on this fanless laptop.

## Brian2 C++ standalone result: memory-budget failure

The matching 100 ms C++ attempt did not reach C++ compilation or simulation.
During Python-side connectivity and parameter materialisation, `vmmap -summary`
reported the following state for the frontend process:

| `vmmap` measure | Observed value |
| --- | ---: |
| Total virtual size | 15.8 GB |
| Total resident | 1.4 GB |
| Total swapped | 13.5 GB |
| Default malloc zone size | 14.7 GB |
| Default malloc zone allocated | 14.6 GB |
| Default malloc zone swapped | 13.4 GB |
| Partial standalone output before compilation | 4.6 GB |

The attempt was stopped after it exceeded the planned 12 GiB process-memory
safety budget and the host entered severe swapping. The incomplete generated
project was then removed; no benchmark source data was deleted.

macOS did **not** deliver an observable kernel OOM kill in this run. Its virtual
memory system swapped most of the allocation instead. The precise conclusion is
that the C++ frontend does not fit in this machine's physical memory and is not
operationally viable here; on a comparable 16 GB environment with disabled or
strictly limited swap, this allocation profile would be expected to terminate as
OOM. It would be inaccurate to claim that this particular macOS process was
literally killed by the OOM mechanism.

## Parallel-routing follow-up

After adding heterogeneous-delay target ownership, source-route fusion and
parallel procedural materialisation, the full topology was rerun for 100 ms on
the same machine with four workers. The before/after `results.bin` files are
byte-identical and both report 91,084,720 delivered synaptic events.

| Measure | Before | After | Change |
| --- | ---: | ---: | ---: |
| Simulation body | 1.813 s | 1.276 s | 1.42x faster |
| End-to-end Device call | 23.248 s | 14.627 s | 1.59x faster |
| Peak child RSS | 3.791 GB | 5.140 GB | +1.349 GB |

The RSS increase is expected from parallel materialisation making the large
arrays resident earlier instead of relying as heavily on lazily mapped zero
pages. It remains comfortably inside the 16 GB host for this run, but event
capacity pooling and a formal memory estimator remain required before treating
the larger residency as final.

## Conclusion

The full-scale PD14 model is viable on the 16 GB M3 machine through the Rust
Device and completed 10 seconds of model time. The original C++ standalone path
failed the controlled memory budget before compilation even for a 100 ms run.
This difference comes from the execution architecture, not from shortening the
Rust model: Rust constructs fixed-total endpoints, weights, and delay ticks from
compact procedural descriptors inside the native process instead of first
creating roughly 300 million-edge arrays in Python and emitting a large static
standalone project.

Because the C++ attempt never reached simulation, this local result establishes
memory feasibility but cannot provide a local speed ratio. The same unified
model has completed on 128 GB Apple Silicon and a 1.1 TiB Linux server, where
both backends fit and Rust was respectively 1.98x and 2.16x faster in the
simulation body; see [PD14_COMPARISON.md](PD14_COMPARISON.md) and
[PD14_LINUX_COMPARISON.md](PD14_LINUX_COMPARISON.md).

## Reproduction

```sh
# Full Rust run used for the result above
python examples/pd14_device.py --backend rust \
  --rustc "$HOME/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc" \
  --output output/pd14-local-rust-full-10s \
  --neuron-scale 1 --indegree-scale 1 --duration-ms 10000 \
  --no-record-spikes --threads 4

# The C++ command is intentionally shown for reproducibility, but should only be
# run when at least about 20 GB of physical memory is available.
python examples/pd14_device.py --backend cpp \
  --output output/pd14-local-cpp-full-100ms \
  --neuron-scale 1 --indegree-scale 1 --duration-ms 100 \
  --no-record-spikes --threads 1
```

The Rust report JSON and binary result dump are generated artifacts and are not
committed. The C++ partial directory was incomplete and was removed immediately
after the controlled stop.
