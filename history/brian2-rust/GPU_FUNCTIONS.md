# Native scalar GPU Functions

Metal and CUDA now accept a native-only Function with a matching
`b2ir-metal-v1` or `b2ir-cuda-device-v1` implementation. Use the existing Brian
`implementation` decorator; the exporter retains the source, entry point and
SHA-256 in B2IR. A missing backend implementation fails during planning.

GPU execution still requires explicit `numeric_mode="float32"`. Frozen v1
Function signatures admit `f64`, `i64` and `bool`. The GPU profile maps these to
`float`, signed 64-bit `long` and `bool`, including return values. This differs
from the CPU C ABI (`double`, `int64_t`, `uint8_t`). Each GPU compilation checks
the exact declared function type, so an implicit argument or return conversion
cannot silently change the interface. CUDA entries must be device-callable.

For example, a function with Python control flow can provide both adapters:

```python
import brian2 as b

@b.implementation("b2ir-metal-v1", """
float positive_square(float x) { return x > 0.0f ? x*x : 0.0f; }
""", name="positive_square")
@b.implementation("b2ir-cuda-device-v1", """
__device__ float positive_square(float x) {
    return x > 0.0f ? x*x : 0.0f;
}
""", name="positive_square")
@b.check_units(x=1, result=1)
def positive_square(x):
    if x > 0:
        return x*x
    return 0

b.set_device("rust_standalone", engine="metal", numeric_mode="float32")
group = b.NeuronGroup(32, "x : 1\ny : 1",
                      namespace={"positive_square": positive_square})
group.run_regularly("y = positive_square(x)")
```

Use `engine="cuda"` for the CUDA adapter. The Python body above is not executed
by the GPU; its control flow has no portable B2IR expression. An independent
Rust reference run requires a portable model, and CPU AOT requires a CPU adapter
for a native-only Function.

When a portable body exists, GPU backends retain their established portable
lowering, even if a native descriptor is also present. Native descriptors do not
override portable bodies. The CPU f32 mirror rejects native-only Functions:
arbitrary Metal/CUDA source has no guaranteed matching CPU implementation.

Native source is compiled verbatim inside a private namespace for each Function,
after generated CUDA lowering. It must be self-contained: helper declarations
belong in the same source, and cannot depend on another Function's namespace or
the generator's internal variables. Separate Functions may use identical helper
names. The generated source and the original descriptor hashes participate in
plan and binary identity. Portable-only generated kernels remain unchanged.

Arguments and calls respect whole-statement masks and eager B2IR evaluation.
Each floating return is checked before a later expression or assignment can hide
NaN/infinity. Native bodies themselves are opaque: internal arithmetic and
purity are the source author's responsibility. As in the existing B2IR contract,
explicit native source is trusted executable code; a hash or signature check
does not prove semantic equivalence. Stateful, RNG or non-thread-safe Function
contracts remain rejected.

See the [local verification evidence](execution-plan-evidence/gpu-native-functions/README.md).
NVIDIA compilation and execution are now verified on Modal L4 and A100-40GB:
each passed 121 selected tests, with 59 skips requiring Apple GPU, plus Rust and
CUDA baseline checks. See [CUDA verification](execution-plan-evidence/gpu-expression-cuda/README.md).

Explicit `dag_execution="workgroup"` now accepts native scalar Functions on
Metal and CUDA. Native declarations and source remain verbatim in each packed
stage, including text resembling generated-code markers. Floating results and
64-bit integer/boolean Functions are checked against independent references.
See [native workgroup validation](execution-plan-evidence/gpu-native-workgroup/README.md)
for the bounded experimental mode and its unchanged performance limitations.
