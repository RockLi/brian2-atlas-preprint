"""Run only in the explicit MPI checkout's Python environment, never imported."""
import json
from pathlib import Path
import sys
import numpy as np


def main():
    from brian2_rust.distributed import write_mpi_project, compile_mpi_project, run_mpi_project
    from brian2_rust.results import load_results
    job, source = map(Path, sys.argv[1:3])
    model = json.loads((job/"model.json").read_text())
    write_mpi_project(model, job/"native", ranks=int(sys.argv[3]),
                      runner=source/"brian2-rust/target/release/b2-runner")
    compile_mpi_project(job/"native")
    run_mpi_project(job/"native", job/"results", timeout=300)
    results = load_results(model, job/"results")
    np.savez_compressed(job/"states.npz", **{f"{i}/{key}": value
                        for i, p in enumerate(results["populations"]) for key, value in p["states"].items()})


if __name__ == "__main__":
    main()
