"""Record hashes and schema of large external B2IR artifacts without vendoring."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = args.artifact / "model.json"
    instance = args.artifact / "native" / "instance.bin"
    with model.open("rb") as stream:
        stream.seek(-4096, 2)
        ending = stream.read().decode()
    matched = re.search(r'"protocol":(\{.*?\}),"run":', ending)
    if not matched or '"schema":"b2ir-v1"' not in ending:
        raise RuntimeError("unexpected B2IR envelope")
    protocol = json.loads(matched.group(1))
    output = {
        "artifact": str(args.artifact.resolve()),
        "schema": "b2ir-v1",
        "protocol": protocol,
        "model_json_bytes": model.stat().st_size,
        "model_json_sha256": checksum(model),
        "instance_bin_bytes": instance.stat().st_size,
        "instance_bin_sha256": checksum(instance),
        "storage_policy": "large exact IR/instance remain external; hashes, scientific source and structural manifest are retained in package",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
