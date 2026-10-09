"""Extract only the small B2IR definition prefix from a huge external fixture."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefix = b'{"definition":'
    with args.model.open("rb") as stream:
        buffer = stream.read(len(prefix))
        if buffer != prefix:
            raise RuntimeError("unexpected B2IR layer order")
        buffer = bytearray()
        for _ in range(32):
            buffer.extend(stream.read(1024 * 1024))
            try:
                definition, _ = json.JSONDecoder().raw_decode(buffer.decode())
                break
            except json.JSONDecodeError:
                continue
        else:
            raise RuntimeError("definition layer exceeds 32 MiB or is invalid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(definition, sort_keys=True,
                                      indent=2) + "\n")
    print({"definition_bytes": args.output.stat().st_size,
           "source": str(args.model.resolve())})


if __name__ == "__main__":
    main()
