"""Record same-backend final NMDA edge-state consistency across CPU hosts."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--subdir", default="seeded_cpu8_31_35")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[31, 32, 33, 34, 35])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "protocol": "same scientific backend, explicit seed and 4096 final NMDA edge indices on native ARM Mac27 vs x86_64 Linux23; no Brian2-versus-Rust pointwise comparison",
        "network_size": 5120,
        "seeds": args.seeds,
        "trials": [],
    }
    for seed in args.seeds:
        for backend in ("cpp", "rust"):
            paths = {host: args.raw / host / args.subdir / "nmda_samples" /
                     f"seeded_{backend}_5120_seed{seed}_nmda_final.npz"
                     for host in ("host27", "host23")}
            with np.load(paths["host27"]) as mac, np.load(paths["host23"]) as linux:
                if sorted(mac.files) != sorted(linux.files):
                    raise RuntimeError("cross-host NMDA sample fields differ")
                fields = {}
                for name in sorted(mac.files):
                    left, right = np.asarray(mac[name]), np.asarray(linux[name])
                    if left.shape != right.shape or left.dtype != right.dtype:
                        raise RuntimeError("cross-host NMDA sample shape/dtype differs")
                    fields[name] = {
                        "byte_equal": bool(left.tobytes() == right.tobytes()),
                        "max_absolute_difference": float(np.max(np.abs(left-right))),
                        "sample_count": int(left.size),
                    }
            report["trials"].append({
                "seed": seed, "backend": backend,
                "mac_path": str(paths["host27"]),
                "linux_path": str(paths["host23"]),
                "fields": fields,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({backend: all(all(field["byte_equal"] for field in row["fields"].values())
                        for row in report["trials"] if row["backend"] == backend)
           for backend in ("cpp", "rust")})


if __name__ == "__main__":
    main()
