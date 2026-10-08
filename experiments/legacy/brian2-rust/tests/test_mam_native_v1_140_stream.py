import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mam_native_v1_140_stream import DTYPE, native_blocks, select_attested


def fixture(tmp_path):
    groups = [(100 + 3 * index, 3, 1) for index in range(8)]
    events = {rank: [(4000, (rank - 1) % 48)] for rank in range(48)}
    for start, _, _ in groups:
        for tick, cell in ([(5001, start)] * 56
                           + [(5001, start + 1)] * 55
                           + [(5005, start + 1), (1005000, start + 1)]):
            events[(cell + 1) % 48].append((tick, cell))
    records = []
    for rank in range(48):
        path = tmp_path / f"rank{rank}.events.bin"
        raw = np.asarray(events[rank], dtype=DTYPE).tobytes()
        path.write_bytes(raw)
        records.append(dict(rank=rank, path=str(path), bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest()))
    return records, groups


def test_two_hash_verified_passes_and_global_rank_ownership(tmp_path):
    records, groups = fixture(tmp_path)
    result = select_attested(records, groups=groups, total_neurons=140)
    assert result["source_bound"] is False
    assert result["rank_hashes_verified"] is True
    assert result["manifest_bound_to_accepted_audit"] is False
    assert result["raw_events"] == sum(row["bytes"] for row in records) // 8
    assert [ids.tolist() for ids in result["global_ids"]] == [
        [start + 1] for start, _, _ in groups]
    assert [counts.tolist() for counts in result["eligibility_spikes"]] == [[57]] * 8
    assert int(result["population_counts"].sum()) == 16
    np.testing.assert_array_equal(result["population_counts"][:, 0], [1] * 8)
    np.testing.assert_array_equal(result["population_counts"][:, -1], [1] * 8)


def test_same_size_source_change_and_symlink_rejected(tmp_path):
    records, groups = fixture(tmp_path)
    path = Path(records[0]["path"])
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1  # A valid tick changes without changing file size or ownership.
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum differ"):
        select_attested(records, groups=groups, total_neurons=140)

    records, groups = fixture(tmp_path)
    path = Path(records[0]["path"])
    actual = tmp_path / "actual.bin"
    path.rename(actual)
    path.symlink_to(actual)
    with pytest.raises(ValueError, match="missing, linked"):
        select_attested(records, groups=groups, total_neurons=140)


def test_same_bytes_path_replacement_during_open_pass_rejected(tmp_path):
    records, _ = fixture(tmp_path)
    path = Path(records[0]["path"])
    blocks = native_blocks(records[:1], total_neurons=140, release_cache=False)
    next(blocks)
    original = tmp_path / "original-rank0.events.bin"
    raw = path.read_bytes()
    path.rename(original)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="bytes or checksum differ"):
        next(blocks)
