#!/usr/bin/env python3
"""Pin a no-network Fig. 6 filtered-input redraw to the original PDF raster."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


EXPECTED = {
    "reference_pdf": "054db75150afd393ad813dbe4e6cf0f0f11eef2ff07ca88a3550c96725940a90",
    "candidate_pdf": "add58d0532822e660e6aadb9c2a00896f18f618b6fa635db843b1b906487617f",
    "redraw_pdf": "beeda83342d3e168abf8aeddad3859368e9c641729a7d4803a52494657c5216b",
    "reference_png": "dc289eb16fcd78d986f7745982fb66dd106060817a83b00c2bb7225999183e3f",
    "candidate_png": "2001831047e603aa299b8fad4dae5616a36d17111dc7e7b041ab3a18e380f0ed",
    "redraw_png": "2001831047e603aa299b8fad4dae5616a36d17111dc7e7b041ab3a18e380f0ed",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for role in EXPECTED:
        parser.add_argument(f"--{role.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite report")
    paths = {role: getattr(args, role) for role in EXPECTED}
    for role, path in paths.items():
        if sha256(path) != EXPECTED[role]:
            parser.error(f"frozen {role} hash mismatch")
    if paths["candidate_png"].read_bytes() != paths["redraw_png"].read_bytes():
        parser.error("candidate and reconstructed input rasters differ")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "info", "-i",
         str(paths["reference_png"]), "-i", str(paths["redraw_png"]),
         "-lavfi", "psnr", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    match = re.search(r"PSNR .*?average:([0-9.]+)", proc.stderr)
    if not match:
        parser.error("no finite reference/redraw PSNR")
    result = {
        "schema": "contextual-dendritic-fig6-preprocessing-redraw-check-v1",
        "purpose": "verify_numpy126_pure_data_reconstruction_of_existing_candidate_pdf",
        "input_sha256": EXPECTED,
        "render_command": "pdftoppm -f 1 -l 1 -singlefile -scale-to 1200 -png PDF PREFIX",
        "candidate_vs_numpy126_redraw_raster_byte_exact": True,
        "reference_vs_numpy126_redraw_psnr_db": float(match.group(1)),
        "raw_published_processed_input_arrays_compared": False,
        "scientific_gate_changed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidate_exact": True,
                      "reference_psnr_db": result["reference_vs_numpy126_redraw_psnr_db"]}))


if __name__ == "__main__":
    main()
