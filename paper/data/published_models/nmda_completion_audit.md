# NMDA 2025 external-validation completion audit

This audit maps the original 20-section task to evidence in the current
package. A row is complete only when the named artifact or machine record was
inspected. All requirements are now backed by final artifacts and the strict
machine-readable completion audit.

| Requirement | Status | Authoritative evidence |
| --- | --- | --- |
| 1. Preserve the published workload | **Complete** | External upstream checkout commit and access date are fixed in [SOURCE.md](../SOURCE.md); the checkout is not vendored or edited. The exact/general Brian2 declarations are exported rather than rewritten in Rust. |
| 2. Record upstream sources | **Complete** | [SOURCE.md](../SOURCE.md) contains the mandatory title, authors, journal, year, volume, pages, DOI, publication date, Springer/PubMed/PMC/PDF/code URLs, access date and actual commit `68e6dd970cfc6bab26459fcb8c34ee0f16560d9e`. |
| 3. Read and understand the paper/model | **Complete** | [model_mapping.md](../model_mapping.md) maps paper sections to code and distinguishes exact/general, restricted aggregate, and approximate NEST dynamics. |
| 4. Identify exact implementations A–J | **Complete** | [model_mapping.md](../model_mapping.md) records Brian2 and NEST model identity, numerical methods, connectivity, delays, states, synapse representation, seeds, scaling and timing boundaries from code inspection. |
| 5. Run original Brian2 first | **Complete** | [reproduction.md](reproduction.md) records unchanged upstream execution. The full CPU scale sweep through 20,480 and raw environment/timing artifacts are summarized in [performance.md](performance.md). |
| 6. Establish scientific behavior | **Complete for the primary Brian2 benchmark** | [correctness.md](correctness.md) contains deterministic float64 same-event validation through 10,240, seeded statistical validation, public monitor scope and the separate float32 CUDA contract. The unseeded 20,480 result is explicitly structural/statistical rather than falsely declared bitwise cross-engine equality. |
| 7. Use Brian2 frontend → B2IR → plan → Rust | **Complete** | [unsupported_features.md](unsupported_features.md), B2IR manifests and the final report document the unchanged Brian2 route and generic fixes. No model detector or NMDA-specific replacement kernel was added. |
| 8. Fair CPU comparison | **Complete** | Same-machine, same-workload, same-duration and matched eight-thread CPU comparisons use at least five formal repetitions at 2,560/5,120/10,240/20,480 where reported. The primary largest result is Linux 23 Brian2/Rust 4,443.328/2,271.710 s. See [performance.md](performance.md). |
| 9. GPU evaluation | **Complete for CUDA and Metal** | L4/A100 cover the fixed-input float32 gate at 640 and full 2,560/5,120 capacity workloads. Native Apple M3 Metal additionally passes 32 backend regressions, the published explicit model at 640 and 2,560, and a same-IR CPU-f32 numerical gate at 640. A correctness-gated generic bitset event path improves warm runtime by 1.989×/2.361× at 640/2,560 with exact checked outputs. See [correctness.md](correctness.md), [gpu_scale5120.md](gpu_scale5120.md), [the Metal/NEST extension](metal_nest_extension.md), and machine JSON. CPU, CUDA and Metal timing/precision scopes remain separate. |
| 10. MPI evaluation | **Complete through the largest published size** | Formal native runs on Mac Studio 27 and Linux 23 cover 1/2/4/8/16 and, on 23, 32 ranks. Fixed 40-rank 1/2/4/5-node placement covers 2,560, 10,240 and 20,480, including communication, synchronization, traffic, memory and exact output hashes. See [performance.md](performance.md) and [mpi_multinode_20480.md](mpi_multinode_20480.md). |
| 11. Do not confuse scientific implementations | **Complete** | The explicit/general Brian2 model is the primary target; restricted/aggregated Brian2 is retained as a control. NEST `iaf_bw_2001_exact` and `iaf_bw_2001` are labeled exact and approximate scientific models, never engine variants. |
| 12. Decision-making functional validation | **Complete** | The [full report](decision_psychometric_400.md) records 400 matched exact/approximate trials at each of five coherences: 2,000 pairs, 4,000 simulations, 2,000 unique seeds, 8,000 hash-validated artifacts, sustained-activity trajectories and zero failures. |
| 13. Repeated performance methodology | **Complete for formal performance claims** | Formal CPU, MPI and thread-scaling comparisons use an excluded warm-up and at least five measured repetitions; exceptionally expensive GPU/capacity observations are labeled by their actual repetition count. No ratios combine machines. |
| 14. Self-contained package layout | **Complete** | The package has source/provenance, scripts, raw/processed results, analysis and reports. The complete large campaign and package snapshot are on T7 with recorded hashes rather than silently vendored. |
| 15. Mandatory SOURCE.md | **Complete** | [SOURCE.md](../SOURCE.md) satisfies every required field and records the license-file audit. |
| 16. Acceptance gates | **Gates 1–5 and requested extensions complete** | Gate status and evidence are in [final_nmda2025_validation.md](final_nmda2025_validation.md). Performance was not promoted before correctness; Metal has its own float32 gate, and both the NEST benchmark comparison and NEST decision extension are labeled separately from engine execution. |
| 17. Final report | **Complete** | [final_nmda2025_validation.md](final_nmda2025_validation.md) covers reproduction, feature coverage, CPU/GPU/MPI performance, scaling, the complete decision psychometric result and scientific integrity. |
| 18. Explicit prohibitions | **Complete** | Reports retain failures and slower cases, label thread/precision differences, preserve topology/delays/duration/monitors, and never divide current timings by the paper's different hardware. |
| 19. First execution milestone | **Complete, then extended by user request** | The smallest published original run, structural manifest, engine CPU execution, dynamics validation and matched CPU comparison were completed before CUDA/MPI/full-scale continuation and generic optimization. |
| 20. Scientific integrity principle | **Complete** | [final_nmda2025_validation.md](final_nmda2025_validation.md) contains the explicit integrity ledger. Every validation-only input or precision change is isolated and declared. |

## Completion evidence

The final validator confirms 2,000 decision pairs, 4,000 simulations, 400
trials per model and coherence, 2,000 unique seeds, identical pair inputs,
expected source and image identity, 8,000 valid JSON/NPZ artifact hashes, five
finished worker statuses and zero failures. The processed summary, two rendered
figures, report and finalization record are present. The T7 archive contains the
complete campaign and package snapshot with 2,000 pair receipts, 8,000
scientific files and a fully verified `SHA256SUMS`. The final result of
`analysis/audit_completion.py` is recorded in
[completion_audit_20260921.json](../results/processed/completion_audit_20260921.json).
The two final PNGs were visually inspected and their PDF counterparts checked
for page geometry and readability; the evidence is in the
[figure review](../results/processed/decision_psychometric_figure_review_20260921.json).
