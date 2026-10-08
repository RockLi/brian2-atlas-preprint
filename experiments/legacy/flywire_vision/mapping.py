"""Audit published hexel coordinates without changing the frozen graph labels."""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def root_id(value):
    # Never pass FlyWire IDs through float, including in CSV parsing.
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"root ID must be an unsigned decimal string: {value!r}")
    number = int(value)
    if not 0 < number < 2**64:
        raise ValueError("root ID outside uint64 range")
    return str(number)


def parse_hexels(text, cell_type):
    """Read explicit p/q labels; do not infer axes from CSV display orientation."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or rows[0][0] != "p/q":
        raise ValueError("expected published p/q grid")
    qs = [int(v) for v in rows[0][1:]]
    if not qs or len(qs) != len(set(qs)) or min(qs) < 1:
        raise ValueError("invalid q labels")
    result, ps = [], set()
    for row in rows[1:]:
        if len(row) != len(qs) + 1:
            raise ValueError("ragged coordinate grid")
        p = int(row[0])
        if p < 1 or p in ps:
            raise ValueError("invalid p labels")
        ps.add(p)
        for q, value in zip(qs, row[1:]):
            if value.strip():
                result.append(dict(root_id=root_id(value.strip()), cell_type=cell_type, p=p, q=q))
    return result


def _table(path, key, delimiter=","):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=delimiter))
    result = {}
    for row in rows:
        rid = root_id(row[key])
        if rid in result:
            raise ValueError("duplicate root ID in annotation table")
        result[rid] = row
    return result


def audit(source_dir):
    source_dir = Path(source_dir)
    lock = json.loads((HERE / "sources.json").read_text())
    for item in lock["files"]:
        path = source_dir / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"source checksum mismatch: {item['path']}")
    old = _table(source_dir / "annotations-v2.1.0.tsv", "root_id", "\t")
    visual = _table(source_dir / "visual-system-parts-list/data/neuron_table.csv", "cell ID")
    if old.keys() != visual.keys() or len(old) != 139255:
        raise ValueError("visual and frozen v783 annotation root sets differ")
    records = []
    with zipfile.ZipFile(source_dir / "hexel-supplement.zip") as archive:
        names = sorted(n for n in archive.namelist()
                       if n.startswith("DataS2/") and n.endswith(".csv"))
        expected = {"L1", "L2", "L3", "L4", "L5", "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9"}
        if {Path(n).stem for n in names} != expected:
            raise ValueError("unexpected published coordinate tables")
        for name in names:
            records.extend(parse_hexels(archive.read(name).decode(), Path(name).stem))
    multiplicity = Counter(r["root_id"] for r in records)
    issues, eligible = [], []
    for r in records:
        rid = r["root_id"]
        reasons = []
        if rid not in old or rid not in visual:
            reasons.append("missing_v783_root")
        else:
            a, b = old[rid], visual[rid]
            if a["cell_type"] != r["cell_type"]:
                reasons.append("frozen_type_mismatch")
            if b["resolved type"] != r["cell_type"]:
                reasons.append("visual_type_mismatch")
            if a["side"] != "right" or b["side (most synapses)"] != "right":
                reasons.append("hemisphere_mismatch")
        if multiplicity[rid] != 1:
            reasons.append("root_has_multiple_assignments")
        if reasons:
            issues.append({**r, "reasons": reasons,
                           "frozen_type": old.get(rid, {}).get("cell_type"),
                           "visual_type": visual.get(rid, {}).get("resolved type")})
        else:
            eligible.append(r)
    counts = lambda rows: dict(sorted(Counter(r["cell_type"] for r in rows).items()))
    report = dict(schema="flywire-vision-mapping-audit-v1", root_sets_equal=True,
                  neurons=len(old), source_lock=lock, published_assignments=counts(records),
                  eligible_assignments=counts(eligible), issue_records=len(issues),
                  issue_reasons=dict(Counter(reason for r in issues for reason in r["reasons"])),
                  issues=issues,
                  status="annotation audit only; graph CSR and visual dynamics not validated")
    return report, eligible
