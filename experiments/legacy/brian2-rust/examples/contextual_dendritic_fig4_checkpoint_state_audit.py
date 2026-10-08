#!/usr/bin/env python3
"""Hash saved Fig. 4 checkpoint state at 32 s, excluding process pointers.

Pure-data audit only: this loads the completed pickle but never restores or
runs the Brian2 network. Cython buffer pointer addresses are deliberately
omitted because they are process-specific and do not contain buffer values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


EXPECTED_SHA256 = {
    "reference": "9fcea6aed259aa117eb254e7d54b1fa613327333be1048997a452eec1bc65804",
    "candidate": "4cea2df731f83b5fbc5a7ccb61b75d66c01c93f5c9bdbfa0f7b736d96ac9ddd4",
    "compiled_candidate": "aa1a7a23345785b591e193693c409000fce3c4007daeb169c270110837948add",
}
EXPECTED_BYTES = {"reference": 46323499, "candidate": 46323819,
                  "compiled_candidate": 46323484}
FILENAME = "stored_imprint_5ca82125_0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_digest(digest: object, value: object) -> None:
    """Encode only deterministic value semantics, including dtype and shape."""
    if isinstance(value, np.ndarray):
        digest.update(b"array:")
        digest.update(str(value.dtype).encode())
        digest.update(repr(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    elif isinstance(value, np.generic):
        digest.update(b"scalar:")
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict:")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            update_digest(digest, key)
            update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"tuple:" if isinstance(value, tuple) else b"list:")
        digest.update(str(len(value)).encode())
        for item in value:
            update_digest(digest, item)
    elif isinstance(value, (str, bytes, int, float, bool)) or value is None:
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())
    else:
        raise TypeError(f"unsupported saved-state item {type(value).__name__}")


def value_sha256(value: object) -> str:
    digest = hashlib.sha256()
    update_digest(digest, value)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--role", choices=tuple(EXPECTED_SHA256), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing checkpoint audit")
    if (args.checkpoint.name != ("compiled-" + FILENAME if args.role == "compiled_candidate" else FILENAME)
            or args.checkpoint.stat().st_size != EXPECTED_BYTES[args.role]
            or sha256(args.checkpoint) != EXPECTED_SHA256[args.role]):
        parser.error("checkpoint identity mismatch")
    with args.checkpoint.open("rb") as handle:
        saved = pickle.load(handle)
    if set(saved) != {"default"} or saved["default"].get("0_t") != 32.0:
        parser.error("unexpected checkpoint format or simulation time")
    state = saved["default"]
    rng = state["_random_generator_state"]
    if set(rng) != {"numpy_state", "rand_buffer_index", "rand_buffer",
                    "randn_buffer_index", "randn_buffer"}:
        parser.error("unexpected RNG state structure")
    numpy_state = rng["numpy_state"]
    if numpy_state[0] != "MT19937":
        parser.error("unexpected NumPy generator")
    rng_report = {
        "numpy_state_semantic_sha256": value_sha256(numpy_state),
        "numpy_key_array_sha256": hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
        "numpy_position": int(numpy_state[2]),
        "rand_buffer_index": [int(item) for item in rng["rand_buffer_index"]],
        "randn_buffer_index": [int(item) for item in rng["randn_buffer_index"]],
        "rand_buffer_and_randn_buffer_contain_process_pointers_not_values": True,
        "cython_buffer_pointer_values_omitted": True,
    }
    components = {}
    for name, item in sorted(state.items()):
        if name in {"0_t", "_random_generator_state"}:
            continue
        if not isinstance(item, dict):
            raise TypeError(f"unexpected component state {name}: {type(item).__name__}")
        components[name] = {key: value_sha256(value) for key, value in sorted(item.items())}
    report = {
        "schema": "contextual-dendritic-fig4-checkpoint-state-audit-v1",
        "purpose": "compare_completed_32s_checkpoint_values_no_restore_simulation_or_timing",
        "role": args.role,
        "checkpoint_basename": FILENAME,
        "checkpoint_bytes": EXPECTED_BYTES[args.role],
        "checkpoint_sha256": EXPECTED_SHA256[args.role],
        "checkpoint_time_seconds": 32.0,
        "rng": rng_report,
        "components": components,
        "scientific_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"role": args.role, "components": len(components),
                      "fields": sum(map(len, components.values())),
                      "numpy_position": rng_report["numpy_position"]}, sort_keys=True))


if __name__ == "__main__":
    main()
