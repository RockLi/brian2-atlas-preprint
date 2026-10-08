# Compilation reuse between GPU activations

`gpu_compile_reuse=True` enables bounded predecessor compilation reuse for
Metal and CUDA. The default is `False`.

```python
b.set_device("rust_standalone", engine="cuda", numeric_mode="float32",
             event_delivery="sparse", gpu_compile_reuse=True,
             gpu_buffer_reuse=True)
net.run(10 * b.ms)
net.run(10 * b.ms)
b.get_device().close_gpu()
```

Use `engine="metal"` on Apple hardware. The two reuse options are independent:
compilation-only reuse releases resident model buffers after each activation.
Every activation still exports and validates the current model, derives a fresh
plan and binds current host state and pending events. Generated source changes
cause compilation misses. This does not change f32 arithmetic or provide
GPU-resident continuation state.

Metal reuses the loaded Objective-C bridge when its source/options and compiler
context match. An identical kernel source and entry point can clone a retained
pipeline into an independent native handle on the same device. Data ownership
starts empty; optional buffer transfer is a separate step. Closing the predecessor
does not destroy the new handles' retained resources.

CUDA scans generated portable include dependencies with `nvcc -M`, hashes their
contents, and fingerprints architecture, compiler options, compiler binaries and
compilation environment. Matching source/entry/compilation identity can reuse
SHA-256-checked cubin bytes from the live predecessor. Each new executor writes
its sources and cubins to its own directory and loads fresh CuPy modules in its
current device context. Cached cubin bytes are bounded to 32 MiB per retained
executor. During construction both old and new executors can temporarily exist;
there is no process-global or persistent disk cache added by this option.

Opaque native-only scalar Functions bypass kernel/cubin reuse. Compilation
context changes invalidate reuse. Environment values are incorporated into the
fingerprint, but only the digest is reported. Dependency scan failures raise an
error rather than reusing an unverified CUDA binary.

`compilation.json` in the executor directory and `metal_runtime.compilation` or
`cuda_runtime.compilation` in Device summaries report actual kernel compilations
and predecessor hits. Metal also reports bridge reuse. A closed predecessor is
rejected. `close_gpu()`, Device reinitialization, reactivation and failed
transitions release retained executors.

The paired benchmark holds buffer reuse fixed and toggles only compilation reuse.
It checks final states, weights, traces and all spike indices/times for four
successive activations in two balanced orderings. Initial activations are excluded
from the reported continuation median. Its small recurrent workload measures
setup overhead, not general simulation throughput or a comparator ranking.
See [raw evidence](execution-plan-evidence/gpu-compilation/README.md).
