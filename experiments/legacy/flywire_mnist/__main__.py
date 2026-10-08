"""Reproducible development runs; official test evaluation remains a separate gate."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import numpy as np
from . import dataset
from .config import Config, digest
from .encoding import encode, projection
from .graph import load_graph
from .model import ROOT, build, features, instance
from .readout import metrics, predict, select


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="download and verify original MNIST IDX")
    prepare.add_argument("--data", type=Path, required=True)
    run = commands.add_parser("run", help="fit/validation experiment; never opens official test")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--graph", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--backend", choices=("cpu", "metal", "cuda", "mpi"), default="cpu")
    run.add_argument("--config", type=Path)
    run.add_argument("--train-count", type=int, default=100)
    run.add_argument("--validation-count", type=int, default=50)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--no-recurrence", action="store_true")
    run.add_argument("--max-gpu-mib", type=int, default=512)
    run.add_argument("--mpi-source", type=Path)
    run.add_argument("--ranks", type=int, default=2)
    run.add_argument("--keep-runtime", action="store_true", help="retain large per-sample execution files")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(dataset.prepare(args.data), indent=2))
        return
    if not 1 <= args.train_count <= 50000 or not 1 <= args.validation_count <= 10000:
        parser.error("counts must fit the fixed 50,000/10,000 training/validation split")
    if args.backend == "mpi" and not args.mpi_source:
        parser.error("MPI requires an explicit isolated --mpi-source checkout")
    config = Config(**(json.loads(args.config.read_text()) if args.config else {}))
    graph = load_graph(args.graph)
    images, labels, ids = dataset.load(args.data)
    fit_ids, val_ids = dataset.split(labels, seed=config.seed)
    chosen = np.r_[fit_ids[:args.train_count], val_ids[:args.validation_count]]
    readouts = graph.readouts(config.kc_count, config.seed)
    template = build(graph, config, recurrent=not args.no_recurrence)
    args.output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in (ROOT/"experiments/flywire_mnist", ROOT/"python/brian2_rust", ROOT/"src")
                     for p in sorted(folder.rglob("*")) if p.suffix in {".py", ".rs", ".h", ".m", ".c", ".cu"}}
    manifest = {"schema": "flywire-mnist-development-v1", "status": "incomplete",
                "commit": commit, "source_hashes": source_hashes,
                "graph": graph.identity, "scope": graph.scope, "neurons": len(graph.root_ids),
                "edges": graph.edge_count, "config": asdict(config), "backend": args.backend,
                "recurrent": not args.no_recurrence, "definition": template["protocol"]["layers"]["definition"],
                "readout_root_ids": [str(x) for x in graph.root_ids[readouts]],
                "fit_ids": chosen[:args.train_count].tolist(),
                "validation_ids": chosen[args.train_count:].tolist(),
                "official_test_used": False,
                "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in args.data.glob("train-*.gz")}}
    write_json(args.output/"manifest.json", manifest)
    from .backends import CPU, GPU
    if args.backend == "cpu":
        executor = CPU(template, args.output/"execution", threads=args.threads)
    elif args.backend == "mpi":
        from .backends.mpi import MPI
        executor = MPI(template, args.output/"execution", source=args.mpi_source, ranks=args.ranks)
    else:
        executor = GPU(template, args.output/"execution", backend=args.backend,
                       max_bytes=args.max_gpu_mib*1024**2)
    manifest["numeric_profile"] = executor.numeric_profile
    manifest["backend_options"] = getattr(executor, "options", {"threads": args.threads})
    if args.backend == "mpi":
        manifest["backend_options"] = json.loads((args.output/"execution/mpi-source.json").read_text())
    write_json(args.output/"manifest.json", manifest)
    x, projected, seconds = [], [], []
    src, dst = projection(graph.inputs, config.fanout, config.seed)
    target_slots = np.searchsorted(np.sort(graph.inputs), dst)
    try:
        for position, sample_id in enumerate(chosen):
            model = instance(template, images[sample_id], int(sample_id), config)
            result = executor.run(model, f"sample-{sample_id}")
            row = features(model, result, readouts, config.bins)
            x.append(row); seconds.append(result["wall_seconds"])
            event_i, event_t = encode(images[sample_id], int(sample_id), config)
            control = []
            for left, right in zip(config.edges, config.edges[1:]):
                counts = np.bincount(event_i[(event_t >= left) & (event_t < right)], minlength=784)
                control.append(np.bincount(target_slots, weights=counts[src]/config.fanout,
                                           minlength=len(graph.inputs)))
            projected.append(np.concatenate(control))
            np.savez_compressed(args.output/f"features-{sample_id}.npz", features=row,
                                sample_id=sample_id, instance=model["protocol"]["layers"]["instance"])
            if not args.keep_runtime:
                # All paths below were exclusively created by this run. Keep
                # features and provenance, not O(samples × full-state) files.
                work = args.output/"execution"/f"sample-{sample_id}"
                summaries = {p.name: json.loads(p.read_text()) for p in work.glob("*.json")
                             if p.name in {"runtime-binding.json", "compilation.json", "metadata.json", "runtime.json"}}
                write_json(args.output/f"runtime-{sample_id}.json", {"seconds": seconds[-1], **summaries})
                shutil.rmtree(work)
                (args.output/"execution"/f"sample-{sample_id}.bin").unlink(missing_ok=True)
            print(json.dumps({"sample": int(sample_id), "done": position+1, "total": len(chosen),
                              "readout_spikes": int(row.sum()), "seconds": seconds[-1]}), flush=True)
    finally:
        executor.close()
    x = np.asarray(x)
    n = args.train_count
    targets = labels[chosen]
    report = {"scope": graph.scope, "official_test_used": False, "results": {},
              "sample_seconds": {"median": float(np.median(seconds)), "total": sum(seconds)},
              "zero_feature_fraction": float(np.mean(x == 0)),
              "feature_dimension": x.shape[1],
              "note": "Development validation after readout selection, not a locked test result."}
    for name, values in (("flywire", x), ("pixels", images[chosen].reshape(len(chosen), -1)/255.),
                         ("projected_input", np.asarray(projected))):
        classifier, trials = select(values[:n], targets[:n], values[n:], targets[n:])
        np.savez_compressed(args.output/f"readout-{name}.npz", **classifier)
        report["results"][name] = {"validation": metrics(targets[n:], predict(classifier, values[n:])),
                                   "selected_alpha": float(classifier["alpha"]), "trials": trials}
    shuffled = np.random.default_rng(config.seed+10).permutation(targets[:n])
    classifier, trials = select(x[:n], shuffled, x[n:], targets[n:])
    report["results"]["shuffled_fit_labels"] = {"validation": metrics(targets[n:], predict(classifier, x[n:])),
                                                "trials": trials}
    np.savez_compressed(args.output/"features.npz", features=x, projected=np.asarray(projected),
                        labels=targets, sample_ids=chosen)
    manifest["status"] = "complete-development-run"
    manifest["experiment_id"] = digest({k: v for k, v in manifest.items() if k != "status"})
    write_json(args.output/"manifest.json", manifest)
    write_json(args.output/"report.json", report)
    print(json.dumps({"report": str(args.output/"report.json"),
                      "validation_accuracy": {k: v["validation"]["accuracy"]
                                              for k,v in report["results"].items()}}))


if __name__ == "__main__":
    main()
