"""Explicit source selection for historical evidence collection.

The manuscript build uses retained data and does not require these settings.
Collectors must never silently read the current product checkout as history.
"""
import os
from pathlib import Path


def external_directory(name):
    variable = "ATLAS_PREPRINT_" + name
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(
            f"Set {variable} to the retained input directory for this experiment. "
            "See paper/COLLECTION.md; the normal manuscript build uses retained data."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"{variable} is not a directory: {path}")
    return path


def legacy_repo():
    path = external_directory("LEGACY_REPO")
    if not (path / "brian2-rust").is_dir():
        raise SystemExit("ATLAS_PREPRINT_LEGACY_REPO must contain brian2-rust/")
    return path
