#!/usr/bin/env python3
"""Low-load, data-only Fig. 8 bar-panel redraw from the accepted 20-seed gate.

This redraws only the fixed-20, 10-Hz, after-imprint single-cue panel. It
does not import Brian2, construct a model, simulate, or measure performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GATE_SHA256 = "39e3af7c5d91e3ea3a711190e4e88076913982f00188c716c62916cec9e5ab5f"
ARCHIVE_AUDIT_SHA256 = "ee834cf8fb33dbdeafbba1824bb1b88f629cb1526c0acafe63fafc217b327540"
BAR_ORDER = ("first", "last", "same")  # Original Fig_8.py plotted order.
AREAS = ("Y", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--archive-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("refusing to overwrite an existing redraw")
    if (sha256(args.gate_report) != GATE_SHA256
            or sha256(args.archive_audit) != ARCHIVE_AUDIT_SHA256):
        parser.error("accepted 20-seed metric report or T7 archive audit differs")
    gate = json.loads(args.gate_report.read_text())
    archive = json.loads(args.archive_audit.read_text())
    if (gate["metrics_gate_passed"] is not True
            or gate["coverage_passed"] is not True
            or gate["seed_count"] != 20
            or gate["area_record_count"] != 240
            or gate["new_recall_group_count"] != 68
            or gate["whole_figure8_s7_science_gate_passed"] is not False
            or gate["performance_authorized"] is not False
            or archive["plotted_fixed20_after_imprint_single_cue_archive_gate_passed"] is not True
            or archive["all_12_predeclared_raw_hdf_gates_passed"] is not True
            or archive["all_12_t7_full_raw_hdf_sha256_matched_closed_source_reports"] is not True
            or archive["new_group_count"] != 68):
        parser.error("limited-subset science gate not accepted")
    data = []
    for area in AREAS:
        for category in BAR_ORDER:
            bar = gate["bars"][area][category]
            values = [float(value) for value in bar["values"]]
            if (len(values) != 40 or bar["finite_count"] != 40
                    or not all(math.isfinite(value) for value in values)
                    or not math.isclose(sum(values) / 40, bar["mean"],
                                        rel_tol=0, abs_tol=1e-12)):
                parser.error(f"bad finite observations for {area}/{category}")
            data.append((area, category, float(bar["mean"]), values))

    fig, ax = plt.subplots(figsize=(8.7, 5.1), layout="constrained")
    positions = [0, 1, 2, 4, 5, 6]
    for position, (area, category, mean, values) in zip(positions, data):
        color = "#5682a3" if area == "Y" else "#d47d54"
        ax.bar(position, mean, width=0.74, color=color, alpha=0.82, zorder=2)
        ax.scatter([position] * len(values), values, s=8, color="#252525",
                   alpha=0.52, linewidths=0, zorder=3)
    ax.set_xticks(positions, [f"{category}\narea {area}" for area, category, _, _ in data])
    ax.set_ylabel("Mean firing rate / last-imprint rate")
    ax.set_xlabel(
        "Fixed size 20 · 10 Hz · after-imprint · single-cue only; not the full Fig. 8/S7 experiment",
        fontsize=8, labelpad=12)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Fig. 8 source-defined bar panel — complete 20-seed limited subset")
    args.output_dir.mkdir(parents=True)
    png = args.output_dir / "fig8-fixed20-after-imprint-single-cue-20seed.png"
    fig.savefig(png, dpi=180, metadata={"Software": "matplotlib; data-only redraw"})
    plt.close(fig)
    proof = {
        "schema": "contextual-fig8-complete-subset-replot-v1",
        "mode": "mac_data_only_low_load_cache_redraw_no_simulation_no_performance",
        "scope": "fixed20_10hz_after_imprint_single_cue_source_plotted_bar_panel_only",
        "input_metrics_gate_sha256": GATE_SHA256,
        "input_t7_archive_audit_sha256": ARCHIVE_AUDIT_SHA256,
        "source_script_sha256": sha256(Path(__file__)),
        "bar_order_matches_source_Fig_8_py": list(BAR_ORDER),
        "areas": list(AREAS),
        "seed_count": 20,
        "points_per_bar": 40,
        "png_file": png.name,
        "png_sha256": sha256(png),
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    (args.output_dir / "provenance-v1.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"png_sha256": proof["png_sha256"],
                      "provenance_sha256": sha256(args.output_dir / "provenance-v1.json")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
