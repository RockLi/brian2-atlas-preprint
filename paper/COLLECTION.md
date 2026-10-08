# Collecting historical evidence

The normal manuscript build reads the retained files already in `paper/data/`:

```sh
python paper/scripts/build_manuscript.py
```

Collectors are separate maintenance tools. They copy and verify old experiment records and can replace retained data. Run them in an isolated checkout using the experiment's actual captured inputs. The current Atlas `dev` revision must not be substituted for those inputs. Version identities already recorded by a collector remain part of its output; choosing a directory alone does not prove that it is the right scientific source snapshot.

The original machine paths are preserved in the import manifest and historical evidence. Executable collectors now require explicit source locations:

| Collector | Required input selection |
| --- | --- |
| `collect_evidence.py` | `--cpu-root`, `--gpu-root`, `--mpi-root`, each pointing to the corresponding retained `brian2-rust/` directory |
| `collect_plan_evidence.py` | `ATLAS_PREPRINT_PLAN_SOURCE` (retained `brian2-rust/`), `ATLAS_PREPRINT_PLAN_ARCHIVE` (GPU archive root), `ATLAS_PREPRINT_LEGACY_REPO` |
| `collect_published_models.py` | `ATLAS_PREPRINT_LEGACY_REPO`, `ATLAS_PREPRINT_NMDA_ROOT` (the retained `nmda_skaar_2025` directory), `ATLAS_PREPRINT_DENDRITIC_ARCHIVE` |
| `collect_v3_evidence.py` | `ATLAS_PREPRINT_LEGACY_REPO`, `ATLAS_PREPRINT_DENDRITIC_ARCHIVE`; the legacy checkout must retain Git metadata for the review snapshot |
| `collect_capacity_evidence.py`, `collect_scaling_evidence.py`, `collect_mpi_heterogeneous.py`, `analyze_full_scale_resources.py` | `ATLAS_PREPRINT_LEGACY_REPO` |

`ATLAS_PREPRINT_LEGACY_REPO` names the retained legacy repository root containing `brian2-rust/`. It is never inferred from this repository's parent directory. The dendritic archive is the original contextual-dendritic-gating archive tree. Set `TMPDIR` and `MPLCONFIGDIR` to your external scratch disk when collecting or plotting large histories.

Missing source selections stop before evidence is read or replaced. Outputs continue to be written under this repository's `paper/data/` and `paper/validation/`. Migration checks cover path selection and syntax. Historical collectors have not been rerun as new experiments, and the supplied path must still be checked against the relevant retained source manifests and archive checksums.
