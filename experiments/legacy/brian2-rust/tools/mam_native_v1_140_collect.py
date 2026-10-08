#!/usr/bin/env python3
"""Exact-once T7 preservation of one accepted native V1 140-cell derivation.

The source run and every output member are rehashed before transfer. A failed
transfer keeps its pending directory for inspection; it is never retried by
this collector. Collection does not decide scientific equivalence.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


PROXY = "jbf.goldenhen.com.hk:55443"
NODE = "hk-prod-model-ae02-23"
BASE = Path("/data/brick2/brian2-mpi-region-20260907")
CONTROL = BASE / "native-v1-140-derived-control-v1"
LAUNCHER = CONTROL / "mam_native_v1_140_launch.py"
LAUNCHER_SHA = "de0d6d150497b12abfd9b15c9c667509ce5898b334b35b292d60aea01f37a614"
T7 = Path("/Volumes/T7")
T7_PARENT = T7 / "brian2-mpi-20260908/artifacts"
MIN_FREE = 128 * 2**30
MAX_MEMBER = 512 * 2**20
MAX_TOTAL = 2 * 2**30


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def remote_python(code, timeout=240):
    encoded = base64.b64encode(code.encode()).decode()
    payload = "import base64;exec(base64.b64decode(" + repr(encoded) + "))"
    command = ["tsh", "--proxy=" + PROXY, "ssh", "rock@" + NODE,
               "/usr/bin/python3 -c " + shlex.quote(payload)]
    result = subprocess.run(command, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=timeout)
    need(result.returncode == 0, "remote collection probe failed: "
         + result.stdout + result.stderr)
    return result.stdout


def remote_probe(seed):
    code = """
import hashlib,importlib.util,json,pathlib,stat
seed=__SEED__
launcher=pathlib.Path(__LAUNCHER__)
spec=importlib.util.spec_from_file_location('native_v1_140_launch',launcher)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
p=module.paths(seed)
receipt=module.CONTROL/f'seed{seed}-collection.json'
valid=module.verify_completion(seed)
files={
 'control/launcher.py':launcher,
 'control/launch.json':p['launch'],
 'control/guard.json':p['guard'],
 'control/controller.json':p['controller'],
 'control/completion.json':p['completion'],
 'control/journal.log':p['journal'],
 'control/expected-source.json':p['manifest'],
 'control/rust-identity.json':module.RUST_IDENTITY,
 'control/normalization.json':module.NORMALIZATION,
 'control/parameters.json':p['parameters'],
 'control/accepted-series.json':p['series'],
 'control/accepted-series-catalog.json':p['series'].with_name('catalog.json'),
}
files.update({f'source/{name}':module.SOURCE/name for name in module.SOURCE_SHA})
files.update({f'output/{path.name}':path for path in p['output'].iterdir()}
             if p['output'].is_dir() and not p['output'].is_symlink() else {})
rows=[];total=0
if valid:
 for relative,path in sorted(files.items()):
  info=path.lstat()
  if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=__MAX_MEMBER__:
   raise ValueError('missing, linked or oversized collection member: '+str(path))
  digest=module.sha(path)
  rows.append(dict(relative=relative,source=str(path),bytes=info.st_size,sha256=digest))
  total+=info.st_size
 if total>__MAX_TOTAL__: raise ValueError('collection exceeds size cap')
 if module.sha(launcher)!=__LAUNCHER_SHA__ or module.source_state(seed)!=module.expected_sources(seed):
  raise ValueError('collection source bundle differs')
existing=module.sha(receipt) if receipt.is_file() and not receipt.is_symlink() else None
print(json.dumps(dict(seed=seed,source_valid=valid,files=rows,total_bytes=total,
 completion_sha256=module.sha(p['completion']) if valid else None,
 remote_receipt_sha256=existing),sort_keys=True))
"""
    for old, new in {"__SEED__": str(seed), "__LAUNCHER__": repr(str(LAUNCHER)),
                     "__MAX_MEMBER__": str(MAX_MEMBER),
                     "__MAX_TOTAL__": str(MAX_TOTAL),
                     "__LAUNCHER_SHA__": repr(LAUNCHER_SHA)}.items():
        code = code.replace(old, new)
    return json.loads(remote_python(code))


def destinations(seed):
    output = T7_PARENT / f"native-v1-140-derived-seed{seed}-v1"
    return output, output / "collection.json"


def verify_existing(remote, output, receipt_path):
    need(output.is_dir() and not output.is_symlink()
         and receipt_path.is_file() and not receipt_path.is_symlink()
         and sha(receipt_path) == remote["remote_receipt_sha256"],
         "existing collection receipt differs")
    receipt = json.loads(receipt_path.read_text())
    need(receipt.get("schema") == "b2-mam-native-v1-140-collection-v1"
         and receipt.get("seed") == remote["seed"]
         and receipt.get("source_completion_sha256") == remote["completion_sha256"]
         and receipt.get("files") == remote["files"]
         and receipt.get("scientific_acceptance") is False,
         "existing collection identity differs")
    expected = {row["relative"] for row in receipt["files"]} | {"collection.json"}
    actual = {path.relative_to(output).as_posix(): path
              for path in output.rglob("*") if path.is_file()}
    need(expected <= actual.keys(), "existing collection member missing")
    # macOS creates AppleDouble companions for exFAT metadata on T7. They
    # are not evidence members, but require an exact companion and header.
    for relative in actual.keys() - expected:
        sidecar = actual[relative]
        companion = sidecar.with_name(sidecar.name[2:])
        need(sidecar.name.startswith("._") and not sidecar.is_symlink()
             and 0 < sidecar.stat().st_size <= 64 * 1024
             and companion.exists()
             and (companion.is_dir() or companion.relative_to(output).as_posix() in expected)
             and sidecar.open("rb").read(8) == bytes.fromhex("0005160700020000"),
             "unexpected T7 member: " + relative)
    for row in receipt["files"]:
        path = output / row["relative"]
        need(path.is_file() and not path.is_symlink()
             and path.stat().st_size == row["bytes"]
             and sha(path) == row["sha256"],
             "existing collection member differs: " + row["relative"])
    return sha(receipt_path)


def probe(seed):
    remote = remote_probe(seed)
    output, receipt = destinations(seed)
    mounted = T7.is_dir() and T7.stat().st_dev != T7.parent.stat().st_dev
    free = None
    if mounted:
        fs = os.statvfs(T7)
        free = fs.f_bavail * fs.f_frsize
    local_exists = output.exists() or output.is_symlink()
    remote_exists = remote["remote_receipt_sha256"] is not None
    pending = list(T7_PARENT.glob(f".native-v1-140-derived-seed{seed}-v1.pending-*")) if mounted else []
    inconsistent = local_exists != remote_exists or bool(pending)
    already = local_exists and remote_exists and not pending
    existing_sha = verify_existing(remote, output, receipt) if already else None
    ready = (remote["source_valid"] is True and not inconsistent
             and not local_exists and not remote_exists and mounted
             and free is not None and free >= MIN_FREE + remote["total_bytes"]
             and remote["total_bytes"] <= MAX_TOTAL)
    return dict(schema="b2-mam-native-v1-140-collection-probe-v1", seed=seed,
                ready=ready, already_collected=already,
                inconsistent_destinations=inconsistent,
                t7_mounted=mounted, t7_free_bytes=free,
                t7_output_exists=local_exists,
                remote_receipt_exists=remote_exists,
                pending_destinations=[str(path) for path in pending],
                existing_receipt_sha256=existing_sha, remote=remote)


def fetch(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["tsh", "--proxy=" + PROXY, "scp",
                             "rock@" + NODE + ":" + source, str(destination)],
                            stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=3600)
    need(result.returncode == 0, "source transfer failed: " + result.stdout + result.stderr)


def collect(seed):
    state = probe(seed)
    need(state["ready"] is True and state["already_collected"] is False
         and state["inconsistent_destinations"] is False,
         "native V1 collection not ready")
    remote = state["remote"]
    output, _ = destinations(seed)
    pending = output.with_name("." + output.name + ".pending-" + str(os.getpid()))
    pending.mkdir(mode=0o700)
    for row in remote["files"]:
        destination = pending / row["relative"]
        fetch(row["source"], destination)
        need(destination.stat().st_size == row["bytes"]
             and sha(destination) == row["sha256"],
             "transferred source differs: " + row["relative"])
    receipt_data = dict(schema="b2-mam-native-v1-140-collection-v1", seed=seed,
                        source_completion_sha256=remote["completion_sha256"],
                        files=remote["files"], total_bytes=remote["total_bytes"],
                        all_members_hash_verified=True, scientific_acceptance=False,
                        rust_native_equivalence=False, paper_equivalence=False,
                        performance_cost_acceptance=False,
                        collected_unix_seconds=time.time())
    receipt_path = pending / "collection.json"
    with receipt_path.open("x") as stream:
        json.dump(receipt_data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(pending, output)
    encoded = base64.b64encode((output / "collection.json").read_bytes()).decode()
    remote_python("from pathlib import Path;import base64,os;"
                  "p=Path(" + repr(str(CONTROL / f"seed{seed}-collection.json")) + ");"
                  "f=p.open('xb');f.write(base64.b64decode(" + repr(encoded) + "));"
                  "f.flush();os.fsync(f.fileno());f.close()")
    verified = probe(seed)
    need(verified["already_collected"] is True,
         "native V1 collection receipt publication incomplete")
    return dict(collected=True, seed=seed,
                t7_output=str(output),
                receipt_sha256=verified["existing_receipt_sha256"],
                total_bytes=remote["total_bytes"],
                scientific_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1729, 1730, 1731), required=True)
    parser.add_argument("--mode", choices=("probe", "collect"), required=True)
    args = parser.parse_args()
    result = probe(args.seed) if args.mode == "probe" else collect(args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
