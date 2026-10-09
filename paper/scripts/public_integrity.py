"""Verify unchanged originals or explicitly mapped public path-redacted exports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = json.loads((ROOT / "migration/public-path-redaction.json").read_text())
EXPORTS = {r["original_sha256"]: r for r in RECORD["changes"]}

def verify_retained(raw, original_sha256, original_bytes=None):
    actual = hashlib.sha256(raw).hexdigest()
    if actual == original_sha256:
        assert original_bytes is None or len(raw) == original_bytes
        return
    record = EXPORTS[original_sha256]
    assert actual == record["public_sha256"], "Unrecognized evidence change"
    assert len(raw) == record["public_bytes"]
    assert original_bytes is None or original_bytes == record["original_bytes"]
