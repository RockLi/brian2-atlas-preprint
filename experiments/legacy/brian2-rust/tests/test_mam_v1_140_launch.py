from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import mam_v1_140_launch as launch


def test_science_view_binds_split_node23_sources(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    source = tmp_path / "source"
    analysis = tmp_path / "analysis"
    view = tmp_path / "view"
    for directory in (evidence, source, analysis / "cell", analysis / "series",
                      view / "control", view / "analysis"):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("science-completion-v8.json", "raw-audit-completion-v8.json"):
        (evidence / name).write_text("{}")
    monkeypatch.setattr(launch, "EVIDENCE", evidence)
    monkeypatch.setattr(launch, "SOURCE", source)
    monkeypatch.setattr(launch, "ANALYSIS", analysis)
    monkeypatch.setattr(launch, "SCIENCE", view)
    (view / "control/science-completion-v8.json").symlink_to(
        evidence / "science-completion-v8.json")
    (view / "control/raw-audit-completion-v8.json").symlink_to(
        evidence / "raw-audit-completion-v8.json")
    (view / "control/analysis-source-v1").symlink_to(source)
    (view / "analysis/cell").symlink_to(analysis / "cell")
    (view / "analysis/series").symlink_to(analysis / "series")
    assert launch.input_mirror_valid()
    (view / "analysis/series").unlink()
    (view / "analysis/series").symlink_to(analysis / "cell")
    assert not launch.input_mirror_valid()


def test_service_has_prospective_hard_caps():
    arguments = launch.command()
    for required in ("--property=MemoryMax=24576M", "--property=MemorySwapMax=0",
                     "--property=CPUQuota=200%", "--property=RuntimeMaxSec=10800",
                     "--min-free-gib", "1280", "--file-mib", "512",
                     "--timeout", "10795", "--mode", "collect"):
        assert required in arguments
