"""Hash-attested native rank stream for a bounded V1 140-cell derivation.

Each selector pass reads and hashes every one of the 48 immutable rank files.
This module does not attest the supplied manifest or parameter geometry; a
production caller must bind both to accepted receipts before using it.
"""

import hashlib
import os
from pathlib import Path
import stat

import numpy as np

from mam_native_v1_140_selector import select_v1_populations
from mam_v1_140_selector import END_TICK, MAX_BLOCK


DTYPE = np.dtype([("tick", "<u4"), ("cell", "<u4")])
MAX_RANK_BYTES = 3 * 2**30


def inspect_records(records):
    """Validate a caller-pinned 48-rank manifest without reading large files."""
    if len(records) != 48 or [row.get("rank") for row in records] != list(range(48)):
        raise ValueError("native rank manifest must cover 0..47 exactly")
    total = 0
    for row in records:
        rank, path, size, digest = (row["rank"], Path(row["path"]),
                                   row["bytes"], row["sha256"])
        if (path.name != f"rank{rank}.events.bin" or not path.is_absolute()
                or type(size) is not int or not 0 < size <= MAX_RANK_BYTES
                or size % DTYPE.itemsize or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid native rank member contract")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise ValueError("native rank file missing, linked or changed size")
        total += size
    return total // DTYPE.itemsize


def native_blocks(records, *, total_neurons, release_cache):
    """Yield physical ticks/global cells and verify each full-file SHA-256."""
    if release_cache and not hasattr(os, "posix_fadvise"):
        raise ValueError("Linux file-cache release unavailable")
    for row in records:
        rank, path, expected_size = row["rank"], Path(row["path"]), row["bytes"]
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise ValueError("native source changed before pass")
        digest = hashlib.sha256()
        consumed = 0
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (not stat.S_ISREG(opened.st_mode) or opened.st_size != expected_size
                    or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)):
                raise ValueError("native source replaced before open")
            while True:
                chunk = stream.read(MAX_BLOCK * DTYPE.itemsize)
                if not chunk:
                    break
                if len(chunk) % DTYPE.itemsize:
                    raise ValueError("partial native event record")
                digest.update(chunk)
                values = np.frombuffer(chunk, dtype=DTYPE)
                ticks, cells = values["tick"], values["cell"]
                if (np.any(ticks > END_TICK) or np.any(cells >= total_neurons)
                        or np.any((cells.astype(np.uint64) + 1) % 48 != rank)):
                    raise ValueError("native tick, cell or rank ownership differs")
                consumed += len(chunk)
                yield ticks, cells
                if release_cache:
                    os.posix_fadvise(stream.fileno(), consumed - len(chunk),
                                     len(chunk), os.POSIX_FADV_DONTNEED)
            end_fd = os.fstat(stream.fileno())
        end_path = path.lstat()
        if (consumed != expected_size or end_fd.st_size != expected_size
                or (end_path.st_dev, end_path.st_ino) != (before.st_dev, before.st_ino)
                or end_path.st_size != expected_size
                or digest.hexdigest() != row["sha256"]):
            raise ValueError("native rank bytes or checksum differ")


def select_attested(records, *, groups, total_neurons, release_cache=False):
    """Run both complete, hash-verified native passes before returning data."""
    raw = inspect_records(records)
    result = select_v1_populations(
        lambda: native_blocks(records, total_neurons=total_neurons,
                              release_cache=release_cache),
        groups=groups, total_neurons=total_neurons, expected_raw=raw)
    # Matching a caller-supplied manifest is not the same as binding that
    # manifest to the accepted native audit and its T7 collection receipts.
    result["rank_hashes_verified"] = True
    result["manifest_bound_to_accepted_audit"] = False
    result["raw_events"] = raw
    result["rank_sha256"] = tuple(row["sha256"] for row in records)
    return result
