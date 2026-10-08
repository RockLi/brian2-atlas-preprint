"""Shared full-MNIST split, provenance and atomic experiment artifacts."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from . import dataset
from .config import digest


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+f".{os.getpid()}.partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def training_data(folder, seed=783):
    images, labels, ids = dataset.load(folder)
    fit, validation = dataset.split(labels, seed=seed)
    split = {"seed": seed, "fit_ids": fit.tolist(), "validation_ids": validation.tolist(),
             "fit_count": len(fit), "validation_count": len(validation),
             "official_test_count": 10000,
             "training_files": {name: file_sha(Path(folder)/name)
                                for name in dataset.FILES if name.startswith('train-')}}
    split["sha256"] = digest(split)
    return images, labels, ids, fit, validation, split


def source_identity():
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root)): file_sha(p) for p in sorted(root.rglob('*.py'))}


def test_report(labels, predicted):
    from .readout import metrics
    report = metrics(labels, predicted)
    p, n, z = report['accuracy'], report['count'], 1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    report['accuracy_wilson_95'] = [float(center-radius), float(center+radius)]
    matrix = np.asarray(report['confusion_matrix'])
    report['recall_per_class'] = np.divide(matrix.diagonal(), matrix.sum(axis=1),
        out=np.zeros(10, float), where=matrix.sum(axis=1)!=0).tolist()
    return report
