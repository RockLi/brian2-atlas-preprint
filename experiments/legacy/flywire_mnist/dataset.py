"""Original IDX data, explicit official split, label-independent sample IDs."""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def prepare(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, expected in FILES.items():
        path = folder / name
        if not path.exists():
            temporary = path.with_suffix(".partial")
            with urllib.request.urlopen(BASE + name, timeout=60) as response:
                data = response.read(32 * 1024**2)
            if hashlib.md5(data).hexdigest() != expected:
                raise ValueError(f"download checksum mismatch: {name}")
            temporary.write_bytes(data)
            temporary.replace(path)
        data = path.read_bytes()
        if hashlib.md5(data).hexdigest() != expected:
            raise ValueError(f"MNIST checksum mismatch: {name}")
        manifest[name] = {"md5": expected, "sha256": hashlib.sha256(data).hexdigest(),
                          "url": BASE + name, "bytes": len(data)}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_idx(path):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes())
    if len(raw) < 4 or raw[:3] != b"\x00\x00\x08" or raw[3] not in (1, 3):
        raise ValueError("invalid uint8 IDX header")
    ndim = raw[3]
    if len(raw) < 4 + 4 * ndim:
        raise ValueError("truncated IDX dimensions")
    shape = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    if not all(shape) or len(raw) != 4 + 4 * ndim + int(np.prod(shape)):
        raise ValueError("invalid IDX payload size")
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(shape).copy()


def load(folder, *, test=False):
    folder = Path(folder)
    prefix = "t10k" if test else "train"
    names = [f"{prefix}-images-idx3-ubyte.gz", f"{prefix}-labels-idx1-ubyte.gz"]
    for name in names:
        if hashlib.md5((folder / name).read_bytes()).hexdigest() != FILES[name]:
            raise ValueError(f"MNIST checksum mismatch: {name}")
    images, labels = (read_idx(folder / name) for name in names)
    expected = 10000 if test else 60000
    if images.shape != (expected, 28, 28) or labels.shape != (expected,) or np.any(labels > 9):
        raise ValueError("unexpected original MNIST dimensions/labels")
    ids = np.arange(expected, dtype=np.int64) + (60000 if test else 0)
    return images, labels, ids


def split(labels, *, validation=10000, seed=783):
    """Stable stratification with proportional allocation and disjoint indices."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, np.arange(10)).all():
        raise ValueError("expected one-dimensional 0–9 labels")
    if not 0 < validation < len(labels):
        raise ValueError("invalid validation size")
    counts = np.bincount(labels.astype(int), minlength=10)
    quotas = validation * counts / len(labels)
    sizes = np.floor(quotas).astype(int)
    sizes[np.argsort(-(quotas - sizes), kind="stable")[:validation - sizes.sum()]] += 1
    rng = np.random.default_rng(seed)
    fit, val = [], []
    for label, n in enumerate(sizes):
        indices = rng.permutation(np.flatnonzero(labels == label))
        val.extend(indices[:n]); fit.extend(indices[n:])
    return rng.permutation(fit).astype(np.int64), rng.permutation(val).astype(np.int64)
