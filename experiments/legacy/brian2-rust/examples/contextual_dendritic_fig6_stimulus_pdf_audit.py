#!/usr/bin/env python3
"""Read-only Fig. 6 rendered-stimulus diagnostic; no model or timing work.

The PNG inputs are Poppler renders made with:
  pdftoppm -f 1 -l 1 -singlefile -scale-to 1200 -png PDF PREFIX
This script pins both PDF sources and checks corresponding rendered PNGs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


EXPECTED_PDF_SHA256 = {
    "images": (
        "2f64abadf31e2aa9d4567c15631f6f8bd61b8efb3e9bc23615253a0553eb3c51",
        "463196913ab9a929b542573caa7c03131d113f7ba6225680d7bf83f0f44bfbe6",
    ),
    "gabor_patch_positions": (
        "80a83b4317e6cf5e9e5cba69bd74bdffbd56a90375373c9ac68c745a833236e8",
        "ddbeb01a3d5fbffadfbc3b5325b443ee5d3d888c886be4b7c38f3ffa4b2d99ef",
    ),
    "gabor_patches": (
        "614050729de69903655b17f0d964d423690415f1777a8a62cbdcef417a1c7e66",
        "26081fbb85eba573a3d9f8bd2947c2b1e09a233607709010b2ccd43df65fd6f0",
    ),
    "images_filtered": (
        "054db75150afd393ad813dbe4e6cf0f0f11eef2ff07ca88a3550c96725940a90",
        "add58d0532822e660e6aadb9c2a00896f18f618b6fa635db843b1b906487617f",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def psnr(reference: Path, candidate: Path) -> float | None:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "info", "-i", str(reference),
         "-i", str(candidate), "-lavfi", "psnr", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    match = re.search(r"PSNR .*?average:(inf|[0-9.]+)", proc.stderr)
    if not match:
        raise ValueError("ffmpeg did not report PSNR")
    return None if match.group(1) == "inf" else float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-pdf-dir", type=Path, required=True)
    parser.add_argument("--candidate-pdf-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite diagnostic")
    rows = {}
    for name, (reference_sha, candidate_sha) in EXPECTED_PDF_SHA256.items():
        reference_pdf = args.reference_pdf_dir / f"{name}.pdf"
        candidate_pdf = args.candidate_pdf_dir / f"{name}.pdf"
        if sha256(reference_pdf) != reference_sha or sha256(candidate_pdf) != candidate_sha:
            parser.error(f"frozen PDF hash mismatch for {name}")
        reference_png = args.render_dir / f"reference-{name}.png"
        candidate_png = args.render_dir / f"candidate-{name}.png"
        dims = [png_dimensions(path) for path in (reference_png, candidate_png)]
        if dims != [(1200, 1200), (1200, 1200)]:
            parser.error(f"unexpected PNG dimensions for {name}: {dims}")
        render_hashes = [sha256(path) for path in (reference_png, candidate_png)]
        rows[name] = {
            "reference_pdf_sha256": reference_sha,
            "candidate_pdf_sha256": candidate_sha,
            "reference_render_sha256": render_hashes[0],
            "candidate_render_sha256": render_hashes[1],
            "render_pixel_dimensions": [1200, 1200],
            "render_byte_exact": render_hashes[0] == render_hashes[1],
            "render_psnr_db": psnr(reference_png, candidate_png),
        }
    result = {
        "schema": "contextual-dendritic-fig6-stimulus-pdf-raster-audit-v1",
        "purpose": "descriptive_pre_model_stimulus_artifact_comparison_no_simulation_no_timing",
        "render_command": "pdftoppm -f 1 -l 1 -singlefile -scale-to 1200 -png PDF PREFIX",
        "figures": rows,
        "raw_emnist_or_gabor_arrays_compared": False,
        "scientific_gate_changed": False,
        "performance_authorized": False,
        "simulation_executed": False,
        "performance_measurement": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {"raster_exact": row["render_byte_exact"],
                            "psnr_db": row["render_psnr_db"]}
                      for name, row in rows.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
