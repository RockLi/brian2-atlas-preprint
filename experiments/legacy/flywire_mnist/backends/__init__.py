"""Execution adapters share the same B2IR definition and sample instances."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time


class CPU:
    numeric_profile = "rustc-aot-f64"

    def __init__(self, template, directory, *, threads=1):
        from brian2_rust.native import write_project
        self.directory = Path(directory).resolve()
        self.threads = threads
        self.native = self.directory / "native"
        source, _, _ = write_project(template, self.native)
        rustc = shutil.which("rustc") or str(Path.home()/".cargo/bin/rustc")
        subprocess.run([rustc, "--edition=2021", "-C", "opt-level=3", "-C",
                        "codegen-units=1", "-C", "panic=abort", str(source), "-o",
                        str(self.native/"b2-native")], check=True, capture_output=True, text=True)
        (self.directory/"build-identity.json").write_text(json.dumps({
            "rustc": subprocess.check_output([rustc, "--version"], text=True).strip(),
            "flags": ["--edition=2021", "-C", "opt-level=3", "-C", "codegen-units=1", "-C", "panic=abort"],
            "binary_sha256": hashlib.sha256((self.native/"b2-native").read_bytes()).hexdigest(),
        }, indent=2)+"\n")

    def run(self, model, sample_key):
        from brian2_rust import run_compatible_instance
        start = time.perf_counter()
        result = run_compatible_instance(model, self.native, self.directory/f"{sample_key}.bin",
                                         self.directory/sample_key, threads=self.threads)
        result["results"]["wall_seconds"] = time.perf_counter()-start
        return result["results"]

    def close(self):
        pass


class GPU:
    def __init__(self, template, directory, *, backend, max_bytes=512*1024**2):
        from ..model import RUNNER
        if backend == "metal":
            from brian2_rust.metal import MetalExecutor as executor
        elif backend == "cuda":
            from brian2_rust.cuda import CudaExecutor as executor
        else:
            raise ValueError("expected metal or cuda")
        self.numeric_profile = backend+"-float32"
        self.executor = executor
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.previous = None
        self.max_bytes = max_bytes
        self.runner = RUNNER
        self.options = {"synapse_sparse": "bitset"}
        if backend == "metal":
            self.options["dag_execution"] = "resident"

    def run(self, model, sample_key):
        start = time.perf_counter()
        current = self.executor(model, self.directory/sample_key, numeric_mode="float32",
                                event_delivery="sparse", runner=self.runner,
                                compile_reuse=True, reuse_from=self.previous, **self.options)
        try:
            result = current.run(max_buffer_bytes=self.max_bytes)
        except BaseException:
            current.close()
            raise
        if self.previous is not None:
            self.previous.close()
        self.previous = current
        (self.directory/sample_key/"runtime.json").write_text(json.dumps(
            {"numeric_profile": result.get("numeric_profile"),
             "metal_runtime": result.get("metal_runtime"),
             "compilation": current.compilation_report}, indent=2)+"\n")
        result["wall_seconds"] = time.perf_counter()-start
        return result

    def close(self):
        if self.previous is not None:
            self.previous.close()
            self.previous = None
