"""Describe one seed's same-backend output consistency across CPU hosts."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--subdir", default="seeded",
                        help="trial directory within each host's raw results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "network_size": 5120,
        "seed": args.seed,
        "contract": "same backend and explicit seed across native ARM/x86_64; this is not Brian2-versus-Rust pointwise validation",
        "backend_consistency": {},
    }
    for backend in ("cpp", "rust"):
        paths = {host: args.raw / host / args.subdir /
                 f"seeded_{backend}_5120_seed{args.seed}.npz"
                 for host in ("host27", "host23")}
        with np.load(paths["host27"]) as mac, np.load(paths["host23"]) as linux:
            if sorted(mac.files) != sorted(linux.files):
                raise RuntimeError(f"{backend} cross-host monitor fields differ")
            fields = {}
            for name in sorted(mac.files):
                a, b = np.asarray(mac[name]), np.asarray(linux[name])
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise RuntimeError(f"{backend} {name} shape/dtype differs")
                equal = bool(np.array_equal(a, b, equal_nan=True))
                difference = np.abs(a.astype(float) - b.astype(float))
                finite = difference[np.isfinite(difference)]
                fields[name] = {
                    "exact": equal,
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                    "max_absolute_difference": (
                        float(finite.max()) if finite.size else None),
                }
            report["backend_consistency"][backend] = {
                "exact_field_count": sum(item["exact"] for item in fields.values()),
                "total_field_count": len(fields),
                "fields": fields,
                "mac_source": str(paths["host27"]),
                "linux_source": str(paths["host23"]),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({backend: (data["exact_field_count"], data["total_field_count"])
           for backend, data in report["backend_consistency"].items()})


if __name__ == "__main__":
    main()
