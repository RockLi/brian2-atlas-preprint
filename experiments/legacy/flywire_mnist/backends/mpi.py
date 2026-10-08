"""Explicit subprocess boundary for the independently developed MPI branch.

Correctness-first: compile each immutable MPI sample. This is deliberately not
advertised as an efficient batch adapter until MPI supports instance replacement.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def source_identity(root):
    paths = sorted((root/"brian2-rust/python/brian2_rust").rglob("*.py"))
    paths += sorted((root/"brian2-rust/src").rglob("*.rs"))
    return hashlib.sha256(b"".join(str(p.relative_to(root)).encode()+p.read_bytes()
                                   for p in paths)).hexdigest()


class MPI:
    numeric_profile = "mpi-cpu-f64"

    def __init__(self, template, directory, *, source, ranks=2):
        self.source = Path(source).resolve()
        if not (self.source/"brian2-rust/python/brian2_rust/distributed.py").is_file():
            raise ValueError("--mpi-source must contain the MPI implementation")
        if type(ranks) is not int or ranks < 2:
            raise ValueError("MPI requires at least two ranks")
        self.ranks = ranks
        self.identity = source_identity(self.source)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.source, text=True).strip()
        (self.directory/"mpi-source.json").write_text(json.dumps({"source": str(self.source),
             "commit": commit, "source_sha256": self.identity, "ranks": ranks}, indent=2)+"\n")

    def run(self, model, sample_key):
        if source_identity(self.source) != self.identity:
            raise RuntimeError("MPI source changed during experiment; restart with a fixed checkout")
        start = time.perf_counter()
        job = self.directory/sample_key
        job.mkdir()
        (job/"model.json").write_text(json.dumps(model))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join([str(self.source/"brian2-rust/python"), str(self.source)])
        proc = subprocess.run([sys.executable, str(Path(__file__).with_name("mpi_worker.py")),
                               str(job), str(self.source), str(self.ranks)], cwd=job,
                              env=environment, capture_output=True, text=True, timeout=600)
        (job/"worker.log").write_text(proc.stdout+proc.stderr)
        if proc.returncode:
            raise RuntimeError(f"MPI worker failed; inspect {job/'worker.log'}: {proc.stderr[-2000:]}")
        if source_identity(self.source) != self.identity:
            raise RuntimeError("MPI source changed during execution; discard this result")
        with np.load(job/"states.npz", allow_pickle=False) as data:
            populations = [{"states": {key.split('/', 1)[1]: data[key].copy()
                          for key in data.files if key.startswith(f"{i}/")}}
                           for i in range(len(model["definition"]["populations"]))]
        return {"populations": populations, "wall_seconds": time.perf_counter()-start}

    def close(self):
        pass
