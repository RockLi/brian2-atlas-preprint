#!/usr/bin/env python3
"""Create a new read-only-path view of the split node23 v8 science sources.

The symlinks reference immutable existing inputs. This script refuses a second
invocation and never writes to the source files or launches an analysis.
"""
import hashlib
import json
import os
from pathlib import Path


BASE = Path("/data/brick2/brian2-mpi-region-20260907")
ROOT = BASE / "primary-host-v1/confirmation-retry-v2-seed1751-debug"
DEPLOY = ROOT / "v1-140-extractor-v1"
VIEW = DEPLOY / "science-input"
EVIDENCE = ROOT / "evidence"
SOURCE = BASE / "confirmation-analysis-source-v1-seed1750"
ANALYSIS = BASE / "confirmation-analysis/confirmation-retry-v2-seed1751-debug-v4-journal-stdio-100500ms-v8-identity-repair"
EXPECTED = {
    EVIDENCE / "science-completion-v8.json": "cecddd315f1570a40163c4b984407906c01fa89bca75182e70db3b27602a975c",
    EVIDENCE / "raw-audit-completion-v8.json": "b52633c963aff20545b8c749a50694834f5f1c7cf48468f337b9c2d74153097d",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if os.uname().sysname != "Linux" or os.uname().nodename != "hk-prod-model-ae02-23":
        raise RuntimeError("node23 Linux required")
    if VIEW.exists() or VIEW.is_symlink():
        raise FileExistsError("science input view already exists")
    if not DEPLOY.is_dir() or DEPLOY.is_symlink():
        raise RuntimeError("new deployment directory missing")
    for path, digest in EXPECTED.items():
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("accepted input differs: " + str(path))
    if not SOURCE.is_dir() or not (SOURCE / "python/brian2_rust/results.py").is_file():
        raise ValueError("accepted source tree missing")
    if not all((ANALYSIS / stage / "catalog.json").is_file()
               for stage in ("cell", "series")):
        raise ValueError("accepted cell or series stage missing")
    links = {
        "control/science-completion-v8.json": EVIDENCE / "science-completion-v8.json",
        "control/raw-audit-completion-v8.json": EVIDENCE / "raw-audit-completion-v8.json",
        "control/analysis-source-v1": SOURCE,
        "analysis/cell": ANALYSIS / "cell",
        "analysis/series": ANALYSIS / "series",
    }
    VIEW.mkdir(mode=0o700)
    (VIEW / "control").mkdir(mode=0o700)
    (VIEW / "analysis").mkdir(mode=0o700)
    for name, target in links.items():
        (VIEW / name).symlink_to(target)
    print(json.dumps({"view": str(VIEW), "links": {name: str(target)
                                                  for name, target in links.items()},
                      "accepted_receipts_sha256": {str(path): digest
                                                   for path, digest in EXPECTED.items()}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
