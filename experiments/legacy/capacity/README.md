# Connected capacity and weak-scaling experiment programs

This collection preserves the experiment-specific preparation, admission, launch, audit and reporting programs for Supplement S19–S20, manuscript §4.8 / §5.10, Figure 10 and Table 6. The sibling directory layout is retained because the later studies reuse earlier controller helpers.

The exact model generator for the five-point study is [prepare-scaling.py](frozen-source/experiment/prepare-scaling.py), SHA-256 `ab11e08e536a187fb61d5e3d4ee0a6cb48725f71cad5fee4af7453f42a95805c`. It was recovered from the frozen source archive identified by the paper, not regenerated with Atlas dev. The older fixed-size model generators are in the same folder.

Study controllers retain their historical environment and fleet assumptions. `selection.json`, machine credentials, large model/output arrays and live directories must be supplied explicitly; no default runnable cluster deployment is provided. Do not execute a fleet controller merely to inspect this collection. This import performs syntax/hash checks only and starts no simulations or remote jobs.

Reported numerical gates, layouts and results remain in [capacity evidence](../../../paper/data/capacity/weak-evidence.json). The conditional full-scale analysis is [analyze_full_scale_resources.py](../../../paper/scripts/analyze_full_scale_resources.py); it requires no new simulation. File-level import identities are in [the manifest](../../../migration/paper-experiment-code-import-20261009.json).
