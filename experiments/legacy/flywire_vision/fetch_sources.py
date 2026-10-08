"""Fetch only the public files in the checked-in source lock; verify hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name("sources.json").read_text())
    for item in lock["files"]:
        target = args.output/item["path"]
        if target.exists():
            data = target.read_bytes()
        else:
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                data = response.read(item["bytes"]+1)
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"source differs from pinned data: {item['path']}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
        print(item["path"], item["sha256"])


if __name__ == "__main__":
    main()
