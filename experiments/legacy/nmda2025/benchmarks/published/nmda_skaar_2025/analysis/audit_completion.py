#!/usr/bin/env python3
"""Strict final-deliverable audit for the NMDA 2025 validation package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


UPSTREAM_COMMIT = "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e"
UPSTREAM_FIGURE4_SHA256 = (
    "442419267393d5ea17dd85282e0341d236b4d2a203e0e71bc8e0ad55c95689fc"
)
REQUIRED_FILES = (
    "README.md",
    "SOURCE.md",
    "model_mapping.md",
    "reports/reproduction.md",
    "reports/correctness.md",
    "reports/performance.md",
    "reports/unsupported_features.md",
    "reports/final_nmda2025_validation.md",
    "reports/completion_audit.md",
    "reports/decision_psychometric_400.md",
    "results/raw/environment.json",
    "results/raw/original_brian2.csv",
    "results/processed/regression_tests_20260920.json",
    "results/processed/decision_psychometric_pipeline_selftest_20260920.json",
    "results/processed/decision_psychometric_400_20260920.json",
    "results/processed/decision_psychometric_400_artifacts_20260920.jsonl",
    "results/processed/decision_psychometric_finalization_20260920.json",
    "results/raw/decision_making/psychometric_400_20260920/manifest_audit.json",
    "results/raw/decision_making/psychometric_400_20260920/real_pair_crosscheck_20260920.json",
    "reports/figures/decision_psychometric_400_20260920.png",
    "reports/figures/decision_psychometric_400_20260920.pdf",
    "reports/figures/decision_psychometric_trajectories_20260920.png",
    "reports/figures/decision_psychometric_trajectories_20260920.pdf",
    "results/processed/mpi_multinode_rank40_20480_20260920.json",
)
SOURCE_FIELDS = (
    "Paper title:",
    "Authors:",
    "Journal:",
    "Year:",
    "Volume:",
    "Pages:",
    "DOI:",
    "Publication date:",
    "Springer:",
    "PubMed:",
    "PMC:",
    "PDF:",
    "Upstream code:",
    "Upstream git commit:",
    "Access date:",
)
MAPPING_TOPICS = (
    "restricted/aggregated",
    "unrestricted/explicit",
    "iaf_bw_2001",
    "iaf_bw_2001_exact",
    "Integrator",
    "Network",
    "Delays",
    "State and synapse representation",
    "Inputs/seeds",
    "Scale, initial conditions, and timing",
)
STALE_DECISION_PHRASES = (
    "The full task remains open while the paper-scale 400-trial decision campaign is running",
    "A larger controlled cluster sweep remains needed",
    "pending final decision artifacts",
    "decision extension in progress",
    "pending final decision update",
    "psychometric reproduction therefore remains open",
    "full psychometric curve and an engine decision route remain separate findings",
    "400-trial-per-level psychometric result and any engine execution remain open",
    "full 400-trial-per-level psychometric sweep was not",
    "The next functional extension is a cluster-scale psychometric sweep",
    "Repeated full-size and multi-node ranks remain open",
    "independently seeded eight-thread statistical validation remains pending",
    "Mac 27 controlled reruns remain pending",
)


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--t7-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.package.resolve()
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        check((root / relative).is_file(), f"missing required file: {relative}", errors)

    source = (root / "SOURCE.md").read_text()
    for field in SOURCE_FIELDS:
        check(field in source, f"SOURCE.md missing field {field}", errors)
    check(UPSTREAM_COMMIT in source, "SOURCE.md missing exact upstream commit", errors)

    mapping_path = root.parents[2] / "docs/nmda2025/model_mapping.md"
    check(mapping_path.is_file(), f"full model mapping missing: {mapping_path}", errors)
    mapping = mapping_path.read_text() if mapping_path.is_file() else ""
    for topic in MAPPING_TOPICS:
        check(topic.lower() in mapping.lower(), f"model mapping missing topic {topic}", errors)

    environment = load_json(root / "results/raw/environment.json")
    primary = environment.get("primary_toolchain_policy", {})
    check(primary.get("rust_release") == "1.98.1", "primary Rust is not 1.98.1", errors)
    check(
        "1.98.1" in json.dumps(environment.get("formal_environment_records", {})),
        "formal environment records do not identify Rust 1.98.1",
        errors,
    )
    superseded = environment.get("superseded_initial_probe", {})
    if "1.86" in json.dumps(superseded):
        check(
            "excluded" in superseded.get("status", "").lower(),
            "historical Rust 1.86 record is not explicitly excluded",
            errors,
        )

    repository_root = root.parents[2]
    toolchain_path = repository_root / "brian2-rust/rust-toolchain.toml"
    device_path = repository_root / "brian2-rust/python/brian2_rust/device.py"
    device_test_path = repository_root / "brian2-rust/tests/test_device.py"
    check(toolchain_path.is_file(), "Rust toolchain pin is missing from the codebase", errors)
    if toolchain_path.is_file():
        toolchain = toolchain_path.read_text()
        check(
            'channel = "1.98.1"' in toolchain,
            "Rust toolchain pin is not 1.98.1",
            errors,
        )
    check(device_path.is_file(), "Rust Device compiler gate is missing", errors)
    if device_path.is_file():
        device_source = device_path.read_text()
        check(
            "Rust runner build requires cargo 1.98.1" in device_source
            and "Rust build requires rustc 1.98.1" in device_source
            and "Rust build requires native" in device_source,
            "Rust Device does not enforce Cargo/rustc 1.98.1 and native architecture",
            errors,
        )
    check(device_test_path.is_file(), "Rust compiler-gate regression test is missing", errors)
    if device_test_path.is_file():
        device_test = device_test_path.read_text()
        check(
            "test_pinned_rustc_rejects_older_release" in device_test
            and "test_pinned_rustc_rejects_non_native_architecture" in device_test,
            "Rust compiler-gate regression does not reject old/non-native toolchains",
            errors,
        )

    if (root / "results/processed/decision_psychometric_400_20260920.json").is_file():
        decision = load_json(
            root / "results/processed/decision_psychometric_400_20260920.json"
        )
        validation = decision.get("validation", {})
        check(decision.get("schema") == "nmda-skaar-2025-decision-psychometric-v1", "wrong decision schema", errors)
        check(validation.get("complete") is True, "decision validation is not complete", errors)
        check(validation.get("pairs") == 2000, "decision pair count is not 2000", errors)
        check(validation.get("simulations") == 4000, "decision simulation count is not 4000", errors)
        check(validation.get("unique_seeds") == 2000, "decision unique seed count is not 2000", errors)
        check(
            validation.get("artifact_files_hashed") == 8000,
            "decision scientific artifact hash count is not 8000",
            errors,
        )
        points = decision.get("points", [])
        check([point.get("coherence_percent") for point in points] == [1, 5, 10, 20, 40], "decision coherence axis mismatch", errors)
        check(all(point.get("trial_count") == 400 for point in points), "decision trial count is not 400 at every coherence", errors)
        method_audit = decision.get("paper_method_audit", {})
        check(
            method_audit.get("upstream_figure4_sha256")
            == UPSTREAM_FIGURE4_SHA256,
            "decision summary does not pin the audited upstream figure4.py",
            errors,
        )
        check(
            "full 0-4000 ms" in method_audit.get(
                "primary_rule_from_released_figure4_code", ""
            ),
            "decision primary choice rule is not explicitly code-faithful",
            errors,
        )
        wang = decision.get("wang_2002_fit_comparison", {})
        check(
            wang.get("alpha") == 9.2 and wang.get("beta") == 1.5,
            "decision summary does not preserve the paper's Wang fit parameters",
            errors,
        )
        check(
            set(wang.get("point_values", {})) == {"1", "5", "10", "20", "40"},
            "decision summary Wang fit coherence points are incomplete",
            errors,
        )
        for point in points:
            post_choice = point.get("post_stimulus_choice", {})
            check(
                "3000-4000 ms >" in post_choice.get("rule", ""),
                f"coherence {point.get('coherence_percent')}: missing post-stimulus choice rule",
                errors,
            )
            for model in ("exact", "approximate"):
                primary = point.get("accuracy", {}).get(model, {})
                check(
                    primary.get("paper_plot_estimate")
                    == primary.get("bootstrap_mean"),
                    f"coherence {point.get('coherence_percent')}/{model}: paper plot estimate is not the bootstrap mean",
                    errors,
                )
                bootstrap = post_choice.get("accuracy", {}).get(model, {})
                check(
                    bootstrap.get("bootstrap_repetitions") == 5000,
                    f"coherence {point.get('coherence_percent')}/{model}: post bootstrap is not 5000",
                    errors,
                )
                rss = point.get("peak_rss", {}).get(model, {})
                check(
                    rss.get("count") == 400
                    and rss.get("unit") == "KiB"
                    and rss.get("median_kib", 0) > 0,
                    f"coherence {point.get('coherence_percent')}/{model}: peak RSS summary is incomplete",
                    errors,
                )
        runtime_by_host = decision.get("runtime_by_host", {})
        check(
            set(runtime_by_host) == set(decision.get("protocol", {}).get("nodes", []))
            and len(runtime_by_host) == 5,
            "decision per-host runtime coverage is incomplete",
            errors,
        )
        for host, by_model in runtime_by_host.items():
            for model in ("exact", "approximate"):
                check(
                    by_model.get(model, {}).get("count") == 400,
                    f"{host}/{model}: runtime sample count is not 400",
                    errors,
                )
        campaign_wall = decision.get("campaign_wall", {})
        check(
            campaign_wall.get("pair_count") == 2000
            and campaign_wall.get("elapsed_seconds", 0) > 0
            and bool(campaign_wall.get("first_pair_started_utc"))
            and bool(campaign_wall.get("last_pair_completed_utc")),
            "decision distributed campaign wall interval is incomplete",
            errors,
        )
        campaign_wall_by_host = decision.get("campaign_wall_by_host", {})
        check(
            set(campaign_wall_by_host)
            == set(decision.get("protocol", {}).get("nodes", []))
            and all(
                value.get("pair_count") == 400
                and value.get("elapsed_seconds", 0) > 0
                for value in campaign_wall_by_host.values()
            ),
            "decision per-host campaign wall intervals are incomplete",
            errors,
        )
        peak_rss_by_host = decision.get("peak_rss_by_host", {})
        check(
            set(peak_rss_by_host) == set(decision.get("protocol", {}).get("nodes", []))
            and len(peak_rss_by_host) == 5,
            "decision per-host peak RSS coverage is incomplete",
            errors,
        )
        for host, by_model in peak_rss_by_host.items():
            for model in ("exact", "approximate"):
                rss = by_model.get(model, {})
                check(
                    rss.get("count") == 400
                    and rss.get("unit") == "KiB"
                    and rss.get("median_kib", 0) > 0,
                    f"{host}/{model}: peak RSS sample summary is incomplete",
                    errors,
                )
        check(
            "not a simultaneous whole-host" in decision.get("peak_rss_scope", ""),
            "decision peak RSS scope is ambiguous",
            errors,
        )

        catalog_path = (
            root / "results/processed/decision_psychometric_400_artifacts_20260920.jsonl"
        )
        if catalog_path.is_file():
            catalog_lines = [line for line in catalog_path.read_text().splitlines() if line]
            check(
                len(catalog_lines) == 8000,
                f"decision artifact catalog has {len(catalog_lines)} records, expected 8000",
                errors,
            )

        manifest_audit_path = (
            root / "results/raw/decision_making/psychometric_400_20260920/manifest_audit.json"
        )
        if manifest_audit_path.is_file():
            manifest_audit = load_json(manifest_audit_path)
            check(manifest_audit.get("complete") is True, "manifest audit is not complete", errors)
            check(
                manifest_audit.get("manifest_sha256") == validation.get("manifest_sha256"),
                "manifest audit hash differs from final decision summary",
                errors,
            )

    selftest_path = (
        root / "results/processed/decision_psychometric_pipeline_selftest_20260920.json"
    )
    if selftest_path.is_file():
        selftest = load_json(selftest_path)
        check(
            selftest.get("schema")
            == "nmda-skaar-2025-decision-psychometric-pipeline-selftest-v1",
            "wrong psychometric pipeline self-test schema",
            errors,
        )
        checks = selftest.get("checks", {})
        check(
            checks.get("completion_audit_complete") is True
            and checks.get("completion_audit_errors") == []
            and checks.get("artifact_catalog_records") == 8000
            and checks.get("archive_pairs") == 2000
            and checks.get("archive_scientific_files") == 8000,
            "psychometric pipeline self-test is incomplete",
            errors,
        )
        for relative, expected in selftest.get("script_sha256", {}).items():
            path = root / relative
            check(
                path.is_file() and sha256(path) == expected,
                f"psychometric self-test script changed after validation: {relative}",
                errors,
            )

    real_pair_path = (
        root
        / "results/raw/decision_making/psychometric_400_20260920/real_pair_crosscheck_20260920.json"
    )
    if real_pair_path.is_file():
        real_pair = load_json(real_pair_path)
        check(
            real_pair.get("schema")
            == "nmda-skaar-2025-decision-real-pair-crosscheck-v1"
            and real_pair.get("complete") is True,
            "real campaign pair JSON/NPZ cross-check is incomplete",
            errors,
        )

    finalization_path = (
        root / "results/processed/decision_psychometric_finalization_20260920.json"
    )
    if finalization_path.is_file():
        finalization = load_json(finalization_path)
        check(
            finalization.get("schema")
            == "nmda-skaar-2025-decision-psychometric-finalization-v1",
            "wrong decision finalization schema",
            errors,
        )
        check(
            finalization.get("validation", {}).get("complete") is True,
            "decision finalization is not complete",
            errors,
        )
        for item in finalization.get("outputs", []):
            path = (root / item.get("path", "")).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                errors.append(f"decision finalization path escapes package: {path}")
                continue
            if not path.is_file():
                errors.append(f"decision finalization output missing: {path}")
            elif sha256(path) != item.get("sha256"):
                errors.append(f"decision finalization hash mismatch: {path}")
        check(
            len(finalization.get("outputs", [])) == 8,
            "decision finalization does not contain exactly eight audited outputs",
            errors,
        )

    for relative in (
        "README.md",
        "reports/decision_making.md",
        "reports/decision_coherence_screen.md",
        "reports/final_nmda2025_validation.md",
        "reports/completion_audit.md",
        "reports/unsupported_features.md",
        "reports/cross_host_20260915.md",
        "reports/scale5120_progress.md",
        "reports/performance.md",
    ):
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text()
        for phrase in STALE_DECISION_PHRASES:
            check(phrase not in text, f"stale decision status in {relative}: {phrase}", errors)

    markdown_links = 0
    missing_links = []
    for path in root.rglob("*.md"):
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(errors="replace")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            markdown_links += 1
            if not (path.parent / target).resolve().exists():
                missing_links.append(f"{path.relative_to(root)} -> {target}")
    check(not missing_links, f"missing Markdown links: {missing_links}", errors)

    check(args.t7_archive.is_dir(), f"T7 archive missing: {args.t7_archive}", errors)
    archived_pair_count = 0
    archived_scientific_file_count = 0
    if args.t7_archive.is_dir():
        checksum = args.t7_archive / "SHA256SUMS"
        check(checksum.is_file(), "T7 archive SHA256SUMS missing", errors)
        checksum_entries = 0
        if checksum.is_file():
            for line_number, line in enumerate(checksum.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
                if match is None:
                    errors.append(f"invalid SHA256SUMS line {line_number}")
                    continue
                expected, relative = match.groups()
                path = (args.t7_archive / relative).resolve()
                try:
                    path.relative_to(args.t7_archive.resolve())
                except ValueError:
                    errors.append(f"SHA256SUMS path escapes archive: {relative}")
                    continue
                if not path.is_file():
                    errors.append(f"SHA256SUMS file missing: {relative}")
                    continue
                checksum_entries += 1
                if sha256(path) != expected:
                    errors.append(f"SHA256SUMS mismatch: {relative}")
        check(checksum_entries > 0, "T7 archive SHA256SUMS has no valid entries", errors)
        check(
            any(args.t7_archive.glob("*psychometric*"))
            or any(args.t7_archive.rglob("decision_psychometric_400_20260920.json")),
            "T7 archive has no psychometric campaign artifact",
            errors,
        )
        pair_files = list(args.t7_archive.rglob("results/*/pair.json"))
        scientific_files = []
        for name in ("exact.json", "exact.npz", "approximate.json", "approximate.npz"):
            model_files = list(args.t7_archive.rglob(f"results/*/{name}"))
            check(
                len(model_files) == 2000,
                f"T7 archive {name} count is {len(model_files)}, expected 2000",
                errors,
            )
            scientific_files.extend(model_files)
        archived_pair_count = len(pair_files)
        archived_scientific_file_count = len(scientific_files)
        check(
            archived_pair_count == 2000,
            f"T7 archive pair.json count is {archived_pair_count}, expected 2000",
            errors,
        )
        check(
            len(list(args.t7_archive.rglob("decision_psychometric_400_20260920.json")))
            >= 1,
            "T7 archive is missing the processed psychometric summary",
            errors,
        )

    result = {
        "schema": "nmda-skaar-2025-completion-audit-v1",
        "package": str(root),
        "t7_archive": str(args.t7_archive),
        "required_files": len(REQUIRED_FILES),
        "markdown_local_links_checked": markdown_links,
        "t7_checksum_entries_verified": checksum_entries if args.t7_archive.is_dir() else 0,
        "t7_archived_pair_count": archived_pair_count,
        "t7_archived_scientific_file_count": archived_scientific_file_count,
        "complete": not errors,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
