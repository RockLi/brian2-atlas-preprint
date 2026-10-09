"""Sample final per-edge NMDA rise/gating states from either simulator."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust" / "python"))
from brian2_rust.results import load_results  # noqa: E402


def cpp_array(directory, projection, state):
    stem = f"_dynamic_array_C_{projection}_NMDA_{state}"
    paths = [path for path in directory.glob(stem + "_*")
             if path.name[len(stem) + 1:].isdigit()]
    if len(paths) != 1:
        raise RuntimeError(f"expected one file for {stem}, found {paths}")
    return np.memmap(paths[0], dtype="<f8", mode="r")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--backend", choices=("cpp", "rust"), required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--results-dir", type=Path,
                        help="existing Rust compiled-replay dump; defaults to ARTIFACT/rust")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scale <= 0 or not float(args.scale).is_integer():
        parser.error("published benchmark scale must be a positive integer")
    artifact = args.artifact.resolve()
    if args.backend == "cpp":
        if args.results_dir is not None:
            parser.error("--results-dir is only for Rust compiled-replay dumps")
        directory = artifact / "results"
        arrays = {(projection, state): cpp_array(directory, projection, state)
                  for projection in ("EE", "EI")
                  for state in ("x", "s_NMDA")}
    else:
        model = json.loads((artifact / "model.json").read_text())
        result = load_results(model, args.results_dir or artifact / "rust")
        names = {synapse["name"]: synapse["states"]
                 for synapse in result["synapses"]}
        arrays = {(projection, state):
                  names[f"C_{projection}_NMDA"][state]
                  for projection in ("EE", "EI")
                  for state in ("x", "s_NMDA")}
    outputs = {}
    summary = {"backend": args.backend, "scale": args.scale,
               "observation": "final per-edge state, 4096 evenly spaced edges per projection"}
    for projection in ("EE", "EI"):
        expected = int((4_194_304 if projection == "EE" else 1_048_576)
                       * args.scale**2)
        x, gate = arrays[(projection, "x")], arrays[(projection, "s_NMDA")]
        if len(x) != expected or len(gate) != expected:
            raise RuntimeError(f"{projection} NMDA edge-state count differs")
        indices = np.linspace(0, expected - 1, 4096, dtype=np.int64)
        for state, values in (("x", x), ("s_NMDA", gate)):
            sample = np.asarray(values[indices]).copy()
            outputs[f"{projection}_{state}"] = sample
            summary[f"{projection}_{state}"] = {
                "sample_mean": float(sample.mean()),
                "sample_std": float(sample.std()),
                "sample_min": float(sample.min()),
                "sample_max": float(sample.max()),
                "sample_quantiles_10_50_90": [
                    float(q) for q in np.quantile(sample, [0.1, 0.5, 0.9])],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **outputs)
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
