"""Read-only Mach-O architecture gates and private dylib capture; stdlib only.

CPU architecture attestation is necessary but does not prove loadability, Metal
availability, numerical correctness, or successful dispatch. No executable is run.
"""
from pathlib import Path
import hashlib
import json
import os
import struct

ARM64 = 0x0100000C
EXECUTE = 2
DYLIB = 6


class ArchitectureGateError(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _thin(data, offset, size):
    if size < 32 or offset < 0 or offset + size > len(data):
        raise ValueError("truncated or out-of-range Mach-O slice")
    magic = data[offset:offset + 4]
    orders = {b"\xcf\xfa\xed\xfe": "<", b"\xfe\xed\xfa\xcf": ">"}
    if magic not in orders:
        raise ValueError("expected a 64-bit Mach-O slice")
    order = orders[magic]
    _, cpu, subtype, filetype, ncmds, commands_size, flags, reserved = struct.unpack_from(order + "8I", data, offset)
    if commands_size > size - 32 or ncmds > commands_size // 8:
        raise ValueError("Mach-O load commands exceed slice bounds")
    cursor, commands_end = offset + 32, offset + 32 + commands_size
    for _ in range(ncmds):
        if cursor + 8 > commands_end:
            raise ValueError("truncated Mach-O load command")
        _, command_size = struct.unpack_from(order + "2I", data, cursor)
        if command_size < 8 or command_size % 8 or cursor + command_size > commands_end:
            raise ValueError("invalid Mach-O load command size")
        cursor += command_size
    if cursor != commands_end:
        raise ValueError("Mach-O command count/size mismatch")
    return dict(offset=offset, bytes=size, cpu_type=cpu, cpu_subtype=subtype,
                architecture={ARM64: "arm64", 0x01000007: "x86_64"}.get(cpu, hex(cpu)),
                filetype=filetype, byte_order="little" if order == "<" else "big",
                command_count=ncmds, commands_bytes=commands_size)


def parse_macho(data):
    data = bytes(data)
    if len(data) < 4:
        raise ValueError("truncated Mach-O magic")
    fat = {b"\xca\xfe\xba\xbe": (">", False), b"\xbe\xba\xfe\xca": ("<", False),
           b"\xca\xfe\xba\xbf": (">", True), b"\xbf\xba\xfe\xca": ("<", True)}
    if data[:4] not in fat:
        return dict(format="mach-o-64", slices=[_thin(data, 0, len(data))])
    if len(data) < 8:
        raise ValueError("truncated fat Mach-O header")
    order, wide = fat[data[:4]]
    count = struct.unpack_from(order + "I", data, 4)[0]
    entry_size = 32 if wide else 20
    header_end = 8 + count * entry_size
    if not 1 <= count <= 32 or header_end > len(data):
        raise ValueError("invalid fat Mach-O architecture table")
    slices, ranges = [], []
    for index in range(count):
        values = struct.unpack_from(order + ("IIQQII" if wide else "5I"), data, 8 + index * entry_size)
        cpu, subtype, offset, size, alignment = values[:5]
        if alignment > 63 or offset < header_end or offset % (1 << alignment):
            raise ValueError("invalid fat Mach-O slice alignment/offset")
        if any(offset < end and start < offset + size for start, end in ranges):
            raise ValueError("overlapping fat Mach-O slices")
        item = _thin(data, offset, size)
        if (cpu, subtype) != (item["cpu_type"], item["cpu_subtype"]):
            raise ValueError("fat architecture table differs from inner Mach-O header")
        ranges.append((offset, offset + size))
        slices.append(item)
    return dict(format="fat-mach-o-64" if wide else "fat-mach-o-32", slices=slices)


def describe_artifact(path, data=None):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("native artifact must be a regular file")
    if data is None:
        data = path.read_bytes()
    result = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                  executable=os.access(path, os.X_OK))
    try:
        result.update(parse_macho(data), parse_status="parsed")
    except (ValueError, struct.error) as error:
        result.update(parse_status="invalid", parse_error=str(error))
    return result


def require_arm64(identity, kind, expected_sha256=None):
    wanted = {"runner": EXECUTE, "dylib": DYLIB}.get(kind)
    if wanted is None:
        raise ValueError("kind must be runner or dylib")
    problems = []
    if identity.get("parse_status") != "parsed":
        problems.append("invalid Mach-O: " + identity.get("parse_error", "no parsed header"))
    else:
        slices = identity.get("slices", [])
        if not slices or any(s["cpu_type"] != ARM64 or s["byte_order"] != "little" for s in slices):
            problems.append("only little-endian ARM64 slices are allowed; mixed/universal x86 is rejected")
        if any(s["filetype"] != wanted for s in slices):
            problems.append("Mach-O filetype does not match " + kind)
    if kind == "runner" and not identity.get("executable"):
        problems.append("runner has no executable permission")
    if expected_sha256 is not None and identity.get("sha256") != expected_sha256:
        problems.append("artifact SHA differs from its frozen/producer identity")
    evidence = dict(identity, artifact_kind=kind, gate_status="rejected" if problems else "passed",
                    gate_reasons=problems, policy="all slices ARM64; runner MH_EXECUTE; library MH_DYLIB")
    if problems:
        raise ArchitectureGateError("; ".join(problems), evidence)
    return evidence


def gate_artifact(path, kind, expected_sha256=None):
    return require_arm64(describe_artifact(path), kind, expected_sha256)


def capture_and_gate_library(path, private_directory, expected_sha256):
    """Preserve actual build output before execute/dlopen, including rejected dylibs."""
    path = Path(path).resolve(strict=True)
    private_directory = Path(private_directory)
    private_directory.mkdir(parents=True, exist_ok=False)
    data = path.read_bytes()
    saved = private_directory / "libatlas-training-metal.dylib"
    with saved.open("xb") as stream:
        stream.write(data)
    original = describe_artifact(path, data)
    retained = describe_artifact(saved)
    result = dict(original=original, retained=retained, trainer_library_sha256=expected_sha256,
                  copy_sha256_matches=original["sha256"] == retained["sha256"],
                  purpose="Actual build artifact retained before first dlopen for architecture and loader diagnosis")
    try:
        if not result["copy_sha256_matches"]:
            raise ArchitectureGateError("retained library differs from actual build output", result)
        result["architecture_gate"] = require_arm64(original, "dylib", expected_sha256)
    except ArchitectureGateError as error:
        result["architecture_gate"] = error.evidence
        with (private_directory / "library-identity.json").open("x") as stream:
            json.dump(result, stream, indent=2); stream.write("\n")
        raise ArchitectureGateError(str(error), result) from error
    with (private_directory / "library-identity.json").open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    return result
