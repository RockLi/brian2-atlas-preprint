# Multi-Area Model confirmation acceptance ledger

State checked on 2026-10-06. This ledger separates recorded output from
scientific and performance claims; it does not change any frozen protocol.

| Claim or gate | Current evidence | Decision |
| --- | --- | --- |
| Four-node, 32-rank, 100.5 s seed1751 neural output and terminal collection | Existing terminal receipts and accepted `raw-audit-completion-v8.json`, SHA-256 `b52633c963aff20545b8c749a50694834f5f1c7cf48468f337b9c2d74153097d` | Engineering raw-output gate passed |
| Seed1751 six-estimator descriptive analysis | The single v8 attempt reused four hash-verified v7 stages, completed FC and lags, and finalized with `science-completion-v8.json`, SHA-256 `cecddd315f1570a40163c4b984407906c01fa89bca75182e70db3b27602a975c` | Six descriptive estimators complete; no equivalence acceptance |
| Seed1751 evidence preservation | 555 files totaling 501,593,150 bytes were rehashed on T7. A receipt-only recovery published the previously missing node23 receipt; both receipts match SHA-256 `3fdc89b9d79e3ddf60f2e14fdee31e85b23d0fcb7710852dca5974ca4e3d8d31` | Full six-stage evidence collected and verified |
| Three Rust realizations | The descriptive 1729/1750/1751 cohort was built once and independently verified on T7 and node23; receipt SHA-256 `2b59079743eddcbc0ee44d6e85226702103df73173c80581811b9c0493f84a1b` | Descriptive cohort complete; formal scientific gate unadmitted |
| Three full-duration native NEST references | [Native full-reference evidence](../native-full-reference-v1/README.md) records 1729, 1730 and 1731 as completed exploratory references | Descriptive reference only; no confirmation gate |
| Seed1751 modern V1 140-cell spectrum extraction | One bounded raw scan finalized with `v1-140-completion-v1.json`, SHA-256 `ceb98c2c5b9671374df14a9007a3338aa752ec7af8240d58dc92db8a84a0a783`; 19 evidence members were rehashed on T7 and node23 with collection receipt SHA-256 `a285fba067b6c4e6f38031aa51d141faac11cbc0107c9ae394aedf40c4ccf6c7`. An [independent NumPy FFT recomputation](v1-140-four-view-recompute-v1.json) checked both rates and all four Welch powers against the archive, with maximum absolute error `5.56e-15`. A [hash-pinned descriptive comparison](v1-140-published-psd-diagnostic-v1.json) includes all four views against the published 100 s V1 spectrum | Four modern deterministic derived views preserved and numerically verified; historical paper sample and paper equivalence unresolved |
| Native NEST and paper statistical equivalence | [Native/Rust comparison](../native-full-reference-v1/cohort-comparison-v1/README.md) records seed variation, systematic LvR mismatch with unresolved historical sampling, and a different V1 spectrum view. [Feasibility audit](../confirmation-feasibility-v1/README.md) shows the three-reference sample is too small for a credible prospective margin/power claim | Unadmitted; no outcome-independent margins, final test, multiplicity rule for separate claims, or adequate sample-size rationale |
| Comparable speed and cost | The [native/Rust diagnostic comparison](../native-full-reference-v1/cohort-comparison-v1/README.md) used different parallel layouts, output policies and timing scopes. No matched, tuned performance packet is admitted | Unproven; no performance trace or advantage claim |

The next scientific gate requires a prospective, outcome-independent design for
equivalence margins, a final test, multiplicity handling, and power/sample-size
rationale. The observed Rust cohort and exploratory NEST range cannot be used
to choose those criteria. No performance packet trace has begun, and no native
NEST, paper, or speed/cost advantage claim is admitted.

The [post-cohort scientific-gate review](SCIENTIFIC_GATE_REVIEW.md) separates
engineering, native-simulator and paper claims and identifies estimator
alignment work that must precede a formal test.
The [correlation source-history check](CORRELATION_SOURCE_HISTORY_20261006.md)
establishes when the all-area wrapper existed and why its fixed 3001-ID
selection remains distinct from the seed1751 uniform diagnostic; it does not
identify the paper's selected cells or dependency version.
The [V1 140-cell spectrum preparation](V1_140_SPECTRUM_PREPARATION.md) adds a
tested four-view calculation component and bounded two-pass deterministic
selector. The [archived-count identity probe](V1_140_IDENTITY_PROBE_20261006.md)
resolved seed1751's 140 modern-wrapper IDs independently of the terminal
tick. The [bounded one-attempt raw extraction](V1_140_LIVE_START_20261006.md)
has now preserved their per-cell 1-ms trains and four derived views on T7,
with matching node23 and T7 receipts. The historical random sample and
scientific decisions remain open.
The [raw extraction budget](V1_140_RAW_EXTRACTION_BUDGET_20261006.md) binds
the existing 200 GB source sizes and hashes to the completed guarded attempt.
The [prospective acceptance design draft](PROSPECTIVE_ACCEPTANCE_DRAFT.md)
lists the still-open scientific and resource decisions; it is not a frozen
test or authorization to launch another campaign.
The [2026-10-06 capacity snapshot](CAPACITY_SNAPSHOT_20261006.md) covers all
six production hosts and T7. Node23 is 307.02 GiB below the old native
two-seed disk admission threshold. The actual seed1751 Rust raw output was
187.94 GiB under its 256 GiB accepted ceiling, invalidating the earlier
hypothetical 128 GiB Rust budget. The snapshot also records identity-checked
time and CPU scopes; it is neither a new launch admission nor a chosen
whole-campaign budget.
The same snapshot identifies node23's separate brick1 NVMe as a potential
future output volume, but its capacity and path contracts have not passed a
new launch admission. Existing brick2 evidence remains untouched.
