"""Audit cross-host B2IR definition differences without changing either IR."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def differences(left, right, path="definition"):
    if type(left) is not type(right):
        return [path + "/type"]
    if isinstance(left, dict):
        result = ([path + "/keys"] if set(left) != set(right) else [])
        for key in sorted(left.keys() & right.keys()):
            result.extend(differences(left[key], right[key], path + "/" + key))
        return result
    if isinstance(left, list):
        result = ([path + "/length"] if len(left) != len(right) else [])
        for index, (a, b) in enumerate(zip(left, right)):
            result.extend(differences(a, b, path + "/" + str(index)))
        return result
    return [] if left == right else [path]


def normalize_monitor_order(definition):
    # Audit-only deep copy; the original B2IR documents remain immutable.
    copied = json.loads(json.dumps(definition))
    for population in copied["populations"]:
        population["monitor"]["variables"].sort()
        for monitor in population["state_monitors"]:
            monitor["variables"].sort()
            monitor["output_variables"].sort()
    return copied


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mac", type=Path, required=True)
    parser.add_argument("--linux", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mac = json.loads(args.mac.read_text())
    linux = json.loads(args.linux.read_text())
    paths = differences(mac, linux)
    grouped = dict(Counter("/".join(path.split("/")[:-1]) for path in paths))
    non_monitor = [path for path in paths
                   if "/monitor/" not in path and "/state_monitors/" not in path]
    normalized_mac, normalized_linux = (normalize_monitor_order(mac),
                                         normalize_monitor_order(linux))
    result = {
        "scope": "cross-host B2IR definition layer; audit-only normalization",
        "mac_source": str(args.mac),
        "linux_source": str(args.linux),
        "raw_definition_hash_equal": mac == linux,
        "difference_count": len(paths),
        "difference_groups": grouped,
        "non_monitor_difference_paths": non_monitor,
        "all_monitor_names_and_lengths_equal": all(
            set(a["monitor"]["variables"]) == set(b["monitor"]["variables"])
            and len(a["monitor"]["variables"]) == len(b["monitor"]["variables"])
            and all(set(x["variables"]) == set(y["variables"])
                    and len(x["variables"]) == len(y["variables"])
                    and set(x["output_variables"]) == set(y["output_variables"])
                    and len(x["output_variables"]) == len(y["output_variables"])
                    for x, y in zip(a["state_monitors"], b["state_monitors"]))
            for a, b in zip(mac["populations"], linux["populations"])),
        "normalized_mac_sha256": digest(normalized_mac),
        "normalized_linux_sha256": digest(normalized_linux),
        "normalized_definitions_equal": normalized_mac == normalized_linux,
        "interpretation": "only variable-list permutations in monitor plans differ; no original IR was rewritten",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({k: result[k] for k in ("difference_count",
        "non_monitor_difference_paths", "normalized_definitions_equal")})
    if non_monitor or not result["all_monitor_names_and_lengths_equal"] or not result["normalized_definitions_equal"]:
        raise SystemExit("cross-host scientific B2IR definition differs")


if __name__ == "__main__":
    main()
