# 20,480-neuron fixed-rank multi-node MPI validation

This campaign executes the largest published explicit/general NMDA workload
without changing the scientific model: 20,480 neurons, 754,974,720 synapses,
335,544,320 per-edge NMDA synapses, float64 RK4 at 0.1 ms, the published
0.5 ms recurrent delays, both original `PoissonInput` objects, and one second
of biological time. The same frozen 40-rank plan and executable are placed as
1×40, 2×20, 4×10, and 5×8 ranks. No equation, state, connection, delay,
input, monitor, precision, or numerical method changes between placements.

## Artifact and setup audit

The frozen model SHA-256 is
`089ca58343b429b689ef07634b7272dd6e0f26770e7b9bb302a8dde0a85b5a09` and
the 40-rank plan SHA-256 is
`55c5274ed3801d4300a8180c3017b744679f7934c4cfe037fef3c8caebd5575e`.
The native x86_64 executable SHA-256 is
`be62136169c7ba400c63f3c25c218fe8b1bf500ce7fff1e32c8837830e2c445c`.
It was compiled on node 25 with Rust 1.98.1/LLVM 22.1.8, GCC 13.3.0, and
MPICH 5.0.1 `ch3:sock`; bridge and Rust compilation took 0.160 and 3.940 s.

Planning produced an 18,077,816,021-byte project. Model hashing took 88.080 s,
JSON loading 254.644 s, validation/generation/sharding 5,885.826 s, project
hashing 11.118 s, and total preparation 6,239.668 s. Peak planner RSS was
232,684,096 KiB, about 221.9 GiB. This one-time cost is separately reported
and is materially larger than one precompiled simulation.

## Controlled protocol and result

All machines are dual-socket AMD EPYC 9454 nodes. Every rank is
single-threaded and pinned to a distinct physical CPU. Systemd cgroups disable
swap and constrain each node to the recorded CPU set and 128 GiB memory.
Hydra control uses Teleport; MPI data uses private `bond0`. Each topology has
one excluded warm-up and five measured runs. Simulation time is the maximum
rank-local simulation stage. Launcher wall includes launch, simulation,
distributed collection, the 10,203,679,408-byte result write, and teardown;
it excludes planning and compilation.

| Placement | Simulation median (min–max) | Speedup vs 1 node | Launcher wall median (min–max) | Median rank exchange | Sum rank peak RSS | Sum proxy cgroup peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 node × 40 ranks | 588.812 s (587.709–604.263) | 1.000× | 627.612 s (624.337–641.272) | 36.532 s | 28.410 GiB | 21.200 GiB |
| 2 nodes × 20 ranks | 592.506 s (562.709–629.639) | 0.994× | 632.551 s (603.616–670.158) | 46.640 s | 28.424 GiB | 29.239 GiB |
| 4 nodes × 10 ranks | 586.537 s (566.299–591.104) | 1.004× | 627.555 s (607.119–632.339) | 38.304 s | 28.440 GiB | 33.867 GiB |
| 5 nodes × 8 ranks | 586.646 s (584.054–588.178) | 1.004× | 627.369 s (624.836–629.581) | 40.459 s | 28.443 GiB | 34.796 GiB |

All 24 formal warm-up and measured outputs match the frozen serial Rust
reference byte for byte. `results.bin` always has SHA-256
`492577d26519c93e828be15eee002e031bf74d907cd440d1fc0562d8528bee38`;
`events.bin` always has SHA-256
`99c9a97cd232fa38a40f150d77dd5a92f2ce3b6b1786f02bb7bb639d5bdaffa5`.
Every run records 57,080 spikes, 1,803,694,080 delivered synaptic events, and
a final time of 1.0 s. CPU-placement audits pass for every rank, and all
cgroup records report zero OOM events.

Four nodes have the lowest simulation median, but the gain over one node is
only 0.39%; five nodes are effectively identical. Two nodes are 0.62% slower.
These differences are smaller than the observed run ranges, so the fixed
40-rank placement is saturated rather than usefully scaling across machines.
The engine still performs 10,000 collective spike exchanges per rank, one per
simulation tick. Distributing the unchanged rank count cannot remove that
global synchronization and increases aggregate cgroup memory from 21.2 GiB
on one node to 34.8 GiB on five.

An additional five one-node measured replays were retained as a replication
set. They are also byte exact and give simulation/wall medians of
582.851/621.012 s, consistent with the declared formal set. They are not
pooled into the primary table because they were scheduled as a follow-up set.

The earlier matched eight-worker Rust CPU result on node 23 has a
2,271.710 s wall median; the best 40-rank MPI launcher wall is 627.369 s.
That descriptive ratio is 3.62×, but it uses 40 single-threaded ranks on up to
five machines versus eight workers on one machine and is therefore not a
same-resource engine speedup. At 10,240 neurons, the strict same-host,
same-40-core control already established that MPI is 1.677× faster than the
shared-memory route. A corresponding same-host 40-core CPU control was not
run at 20,480.

The [machine-readable summary](../results/processed/mpi_multinode_rank40_20480_20260920.json),
[scaling figure](../results/processed/mpi_multinode_rank40_20480_20260920.png),
and [compact raw records](../results/raw/mpi_multinode_rank40_20480_20260919/)
retain every formal repetition, artifact identity, stage timing, CPU binding,
memory counter, and resource-guard record. The complete project, compact
records, source snapshot, reports, and one representative full output are
archived on the external T7 under
`brian2-local-artifacts-20260917/nmda2025-multinode-20480-20260920`.
The project archive SHA-256 is
`4395a93e605fadeb54a000d3bcdf0a5a15b162692dcac36f9afb718276c16fe7`;
the representative-output archive SHA-256 is
`bbee6317f0dba410ce49198681689bb27bad161c8f35e3e19ddb99719d11f0ba`.
Its `SHA256SUMS` verifies all 749 retained files with zero failures.
