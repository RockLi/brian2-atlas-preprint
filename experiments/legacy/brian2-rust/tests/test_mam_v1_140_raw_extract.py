from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pytest

from mam_v1_140_raw_extract import same_archived_identity, systemd_duration_seconds


def test_mirrored_archive_paths_preserve_content_binding():
    frozen = dict(groups=[dict(name=str(i)) for i in range(8)],
                  source_sha256={"/atlas-storage/0002/a": "aa", "/atlas-storage/0002/b": "bb"},
                  selected_ids_certain_for_terminal_0_or_1=True)
    mirror = dict(groups=[dict(name=str(i)) for i in range(8)],
                  source_sha256={"/data/brick2/b": "bb", "/data/brick2/a": "aa"},
                  selected_ids_certain_for_terminal_0_or_1=True)
    assert same_archived_identity(frozen, mirror)
    mirror["source_sha256"]["/data/brick2/a"] = "changed"
    assert not same_archived_identity(frozen, mirror)
    mirror["source_sha256"]["/data/brick2/a"] = "aa"
    mirror["groups"][0]["name"] = "changed"
    assert not same_archived_identity(frozen, mirror)


@pytest.mark.parametrize("display,seconds", [
    ("3h", 10800), ("18h 5s", 64805), ("1h 30min", 5400),
    ("500ms", 0.5), ("10800000000", 10800),
])
def test_systemd_runtime_limit_display(display, seconds):
    assert systemd_duration_seconds(display) == seconds


@pytest.mark.parametrize("display", ["infinity", "0", "3h 3h", "3h garbage"])
def test_systemd_runtime_limit_rejects_unbounded_or_malformed(display):
    if display == "0":
        assert systemd_duration_seconds(display) == 0
    else:
        with pytest.raises(ValueError):
            systemd_duration_seconds(display)
