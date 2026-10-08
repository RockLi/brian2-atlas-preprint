#!/usr/bin/env python3
"""Remote-only, no-network Fig. 6 EMNIST/Gabor preprocessing version probe.

Uses the unchanged tagged Fig_6 filtering functions but avoids Torch's broken
Tensor.numpy() bridge under NumPy 2 by converting selected tensors via list.
It never constructs NetworkTask, invokes Brian2.run, or measures speed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import sys


EXPECTED_HOST = "hk-prod-model-ae09-94"
EXPECTED_SOURCE_SHA256 = "d51e7fb110b8a1de909d9422b32958c821c0422aff2894fbac09219ab26f4048"
LABELS = {"0": 0, "1": 1, "l": 47, "O": 24}
N_SAMPLES_PER_LABEL = 19  # 10 training, 4 visual recall, 5 extra training


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def array_sha256(array) -> str:
    value = hashlib.sha256()
    value.update(str(array.dtype).encode())
    value.update(str(tuple(array.shape)).encode())
    value.update(array.tobytes())
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays-output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != EXPECTED_HOST or platform.system() != "Linux":
        parser.error("preprocessing probe is restricted to the approved remote host")
    repo = args.paper_repository.resolve(strict=True)
    source = repo / "scripts/Fig_6.py"
    if sha256(source) != EXPECTED_SOURCE_SHA256:
        parser.error("tagged Fig_6.py source hash mismatch")
    raw = repo / "results/datasets/ProjectEMNIST/raw/emnist-byclass-train-images-idx3-ubyte"
    if not raw.is_file() or raw.stat().st_size != 547178704:
        parser.error("existing EMNIST byclass input missing or size-mismatched")
    if args.output.exists() or args.arrays_output.exists() or args.output == args.arrays_output:
        parser.error("refusing to overwrite outputs")
    if args.preflight_only:
        print(json.dumps({"source_sha256": EXPECTED_SOURCE_SHA256,
                          "raw_emnist_bytes": raw.stat().st_size,
                          "simulation_executed": False,
                          "performance_measurement": False}, sort_keys=True))
        return

    import numpy as np
    import scipy
    import torch
    import torchvision

    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("Fig_6_preprocessing_probe", source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load frozen Fig_6 source")
    fig6 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fig6)

    dataset = fig6.ProjectEMNIST(root=fig6.EMNIST_DATA_ROOT, split="byclass",
                                  train=True, download=False, transform=None)
    np.random.seed(927)
    class_positions = {}
    selected_positions = {}
    samples = {}
    for label, target in LABELS.items():
        class_positions[label] = torch.nonzero(dataset.targets == target).flatten().tolist()
        selected = np.random.choice(len(class_positions[label]), 50, replace=False)
        selected_positions[label] = [class_positions[label][int(i)] for i in selected]
        # Equivalent to the source's ToTensor()+squeeze() for 8-bit grayscale
        # PIL input, but avoids torch.from_numpy/Tensor.numpy in NumPy 2.
        samples[label] = [np.array(dataset.data[i].float().div(255).tolist(),
                                   dtype=np.float32)
                          for i in selected_positions[label][:N_SAMPLES_PER_LABEL]]
    if any(len(value) != N_SAMPLES_PER_LABEL for value in samples.values()):
        raise ValueError("incomplete selected EMNIST samples")

    patches = [(np.zeros((2, 2)), (0, 0)) for _ in range(400)]
    running_number = 0
    for ii in range(12):
        for sigma in (0.2, 0.4):
            for orientation in np.linspace(0, 180, 5)[:4]:
                for phase in (0, 180):
                    for frequency in (1, 1.5):
                        patch = fig6.gabor_patch(28, frequency, sigma, orientation, phase)
                        start_x, end_x = 5, 23
                        start_y, end_y = 5, 23
                        if ii < 9:
                            size = 6
                            n_x, n_y = ii % 3, ii // 3
                            start_x, end_x = 5 + n_x * size, 5 + n_x * size + size
                            start_y, end_y = 5 + n_y * size, 5 + n_y * size + size
                        x = np.random.randint(start_x, end_x)
                        y = np.random.randint(start_y, end_y)
                        patches[running_number] = (patch, (x, y))
                        running_number += 1
    if running_number != 384:
        raise ValueError("unexpected Gabor patch count")

    all_inputs = np.zeros((len(LABELS), N_SAMPLES_PER_LABEL, 400))
    sorted_indices = None
    for label_index, label in enumerate(LABELS):
        for sample_index in range(N_SAMPLES_PER_LABEL):
            gabor = fig6.filter_image_with_patches(
                samples[label][sample_index], patches=patches,
                create_random_patch_positions=False)
            response = fig6.process_gabor_filtered_array_for_input(
                gabor, n=20, max_response=12).flatten()
            if label_index == 0 and sample_index == 0:
                sorted_indices = np.argsort(response)
            all_inputs[label_index, sample_index, :] = response[sorted_indices]
    if sorted_indices is None:
        raise ValueError("no first-sample response")
    positions = np.array([pair[1] for pair in patches], dtype=np.int64)
    first_sample_pixels = np.array(samples["0"][0], dtype=np.float32)
    args.arrays_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays_output, all_inputs=all_inputs,
                        sorted_indices=sorted_indices,
                        patch_positions=positions,
                        first_sample_pixels=first_sample_pixels)
    report = {
        "schema": "contextual-dendritic-fig6-numpy-preprocessing-probe-v1",
        "purpose": "remote_only_emnist_gabor_preprocessing_no_network_simulation_no_timing",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "raw_emnist_train_images_bytes": raw.stat().st_size,
        "seed": 927,
        "source_arguments": {"n_train": 10, "n_recall": 4,
                             "n_train_additional": 5,
                             "create_random_patch_positions": False},
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "class_counts": {label: len(class_positions[label]) for label in LABELS},
        "selected_dataset_positions": selected_positions,
        "first_sample_pixels_sha256": array_sha256(first_sample_pixels),
        "patch_positions_sha256": array_sha256(positions),
        "sorted_indices_sha256": array_sha256(sorted_indices),
        "all_inputs_sha256": array_sha256(all_inputs),
        "all_inputs_per_label_sha256": {label: array_sha256(all_inputs[i])
                                        for i, label in enumerate(LABELS)},
        "arrays_npz_sha256": sha256(args.arrays_output),
        "network_constructed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "scientific_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"numpy": np.__version__,
                      "first_sample_pixels_sha256": report["first_sample_pixels_sha256"],
                      "patch_positions_sha256": report["patch_positions_sha256"],
                      "sorted_indices_sha256": report["sorted_indices_sha256"],
                      "all_inputs_sha256": report["all_inputs_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
