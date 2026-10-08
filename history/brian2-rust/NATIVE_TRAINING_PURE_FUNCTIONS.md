# Pure scalar functions in native training

`PureFunction(arguments, expression, parameters=(), statements=())` binds an explicit
scalar expression in the training compiler. Formal arguments are lexical and
each actual argument is compiled once, so a repeated use of a Gaussian or
Poisson argument does not introduce extra native draws. Closure parameters are
finite immutable scalar constants or nested `PureFunction` descriptors. The
expanded graph uses existing native forward operations and VJPs on CPU, Metal
and CUDA; it does not call Python during training.

```python
from brian2_rust import PureFunction, compile_training_equation
curve = PureFunction(('x',), '.2*sin(x)+x*x')
program = compile_training_equation('v+gain*curve(v)',
                                    parameters={'gain': (0, 0), 'curve': curve})
```

Brian neuron equations, thresholds/resets, synaptic updates and `run_regularly`
can use custom Brian `Function` objects with readable Python source. The
lowerer snapshots sequential scalar local assignments followed by one return expression, including finite scalar closure
constants/quantities in SI and nested pure helper functions. Brian's unit/signature checks run
before lowering. Standard `numpy`/`math` scalar function calls are identified
by object identity, and lowering never executes the callback. The callback's
recompiled source must match its loaded Python code; editing a module while old
function objects remain live cannot silently replace their semantics. A callback's
formal `v` shadows the surrounding state, and its body cannot implicitly read
the caller's state or namespace.

The current source adapter requires stateless, non-vectorised callbacks with
0–16 positional arguments, including Python positional-only arguments (`/`),
no positional defaults, and no explicit target overrides. Positional-only arguments count
toward the same arity limit and retain their source order.

Numeric decorators retain their executable bodies. Only an authenticated
canonical Brian unit-check wrapper is removed. Fixed positional call packets
can be forwarded or inspected using static indices/slices, builtin `len`, and
empty keyword packet checks. A positional-only prefix is bound before the
remaining packet. Ordinary tuple elements still execute eagerly; packet
specialization cannot erase an unrelated numeric expression or its error.
The numeric-wrapper CPU regression has689 passes and1166 hardware skips
across11 complete modules, recorded in
`mpi-evidence/training-wrapper-body-20261006`. The newer positional-only
extension has separate evidence in
`mpi-evidence/training-positional-only-callbacks-final-20261006`: the entire
new50-identity module passed CPU18/Metal34 tests, with32 GPU/16 CUDA skips.
Both modes have real exit0 receipts and strict source/identity audits. Full
12-module regression and latest CUDA acceptance remain pending.

Omitted keyword-only parameters can have finite builtin Python float/bool or
int32 defaults. Their values come from the loaded `__kwdefaults__` mapping,
not reevaluated source expressions; they become local scalar SSA bindings.
This also applies to numeric variadic wrappers and preserves local scalar
rebindings and generated packet argument names. Required keywords, mutable or
nonfinite defaults, and NumPy scalar defaults are refused. Explicit keyword
call syntax and positional defaults remain outside this extension.
Brian's unit-discarding NumPy implementation copies the loaded keyword-default
mapping along with its global namespace; unit-preserving implementations keep
calling the original function. Both warm modes have original Brian/native
physics evidence. The complete new124-identity regression passed CPU48 and
actual Metal86 tests, with76 GPU/38 CUDA skips and strict source/runtime/identity
and real exit audits. See
`mpi-evidence/training-keyword-defaults-physical-20261006`. Full13-module
execution and the whole stochastic/dynamic-synapse goal remain unaccepted.
Brian's automatically materialized default NumPy implementation is now
recognized after a network is prepared or run. The adapter checks its origin,
code/closure identity and absence of implementation hooks without invoking
those hooks. Unit-preserving defaults use the original callback; unit-discarding
defaults use NumPy's actual copied namespace, including scalar SI quantity
conversions. Changed original globals cannot replace that copied namespace.
Explicit registrations, modified default code/closures and opaque metadata
continue to be refused. The warm conversion evidence is tracked in
`mpi-evidence/training-warm-numpy-callback-20261006`.
Unused arguments and local assignments still execute in source order. Local
rebindings preserve SSA dependencies, while a discarded finite value has no
return-value VJP: `ignored = sqrt(gain)` at `gain=0` is finite and contributes
no singular derivative when unused. An invalid discarded value still fails.
The sequencing graph retains exact int32 and Boolean result kinds and remains
inside the call's surrounding lazy branch.

The source adapter also accepts augmented writes to private, unaliased
arithmetic intermediates. Brian NumPy callbacks receive arrays, so `saved=x;
x*=.8` changes `saved` and caller storage. Such writes to arguments, ambiguous
arithmetic aliases or locals passed through helper calls are refused. Rebind with `x=.8*x` to
create a private result. An augmented write cannot silently change an integer
array to floating point. Explicit descriptors can supply ordered `(name,
expression)` pairs in `statements`; their scalar assignment semantics are
functional. Python/NumPy remainder inside function bodies uses the native
remainder operation directly, avoiding Brian expression rendering's additional
int32 wrapping at large negative divisors.

Explicit `numpy.array(value, dtype=float)` or `dtype=numpy.float64` now creates
known private floating storage. `copy=True` is optional; other options and
`copy=False` are refused. The constructor is checked by identity and never
executed during conversion. Ordered in-place updates refresh every live alias
of that allocation; earlier derived values, independent copies and rebound
names retain their own values. This also preserves zero-dimensional array
behavior for scalar call arguments. The native expression evaluates the copied
operand once with a float coercion and identity VJP, using existing sequence
operations. `asarray`, unspecified/narrow dtypes, captured arrays and mutation
of caller storage remain unsupported. Evidence is tracked in
`mpi-evidence/training-private-array-aliases-20261006`.

Explicit scalar descriptors support comparisons, conditional expressions and
Python `and`/`or`. A conditional evaluates its predicate and only the selected
branch; `and`/`or` return the selected operand, retaining int32 or Boolean kinds.
Source-inspected array callbacks instead use identity-checked `numpy.where`,
`numpy.logical_and`, `numpy.logical_or` and `numpy.logical_not`. These operations
evaluate all their arguments. Invalid intermediate values fail even in an
unselected `where` value. A selection's VJP follows the selected value and does
not differentiate its predicate; Boolean threshold values retain the caller's
declared comparison surrogate. Eager Boolean AND/OR VJPs use both actual gate
values, including gates that a scalar short circuit would leave unevaluated.
Array-dependent Python conditional statements, scalar truth tests on arrays, and chained array
comparisons are refused; use explicit NumPy operations for array callbacks.

Boolean closures are finite immutable scalars. Brian Boolean function metadata
works with both explicit `return_unit=bool` and inferred `@check_units(result=bool)`;
the NumPy reference validates Boolean results without calling `bool` as a unit
function. Callable unit rules continue to receive argument dimensions.

The source adapter now expands statically bounded builtin `range` loops,
including nested loops, negative steps, empty ranges and `for ... else`.
Bounds may use immutable integer captures, previously assigned scalar locals
and enclosing loop indices. Python's whole-function local binding rules are
preserved: a later assignment shadows a same-named capture even at an earlier
loop. Expansion never executes the callback or a shadowed range callable.
Each iteration binds its index and retains scalar statement order, alias checks
and the existing graph/assignment bounds. Bounds and indices require int32,
each range has at most 64 iterations, and nesting has at most eight levels.
Straight-line `break` and `continue` now preserve their actual enclosing loop,
skip unreachable statements and suppress `for ... else` only after a break.
Control in an inner loop's else suite propagates to the enclosing loop.
Static integer conditions now select statement branches during conversion,
including integer comparisons, chained comparisons, not, and lazy and/or.
Finite immutable real constants and previously known scalar locals also select
branches, including ordinary arithmetic and true division. Bounded while loops
may advance a known real scalar; range bounds and indices still require int32.
The adapter never invokes scalar conversion, truth or comparison hooks on
opaque objects, arrays or quantities. Nonfinite reached controls are refused.
Static real arithmetic preserves NumPy scalar precision: promoting float32
controls to Python doubles can change a while loop's iteration count and
physical spikes. The original failing forward case and its correction are
tracked in `mpi-evidence/training-pure-real-control-precision-20261006`.
Bounded while loops use those same conditions and preserve break/continue/else.
Each while executes at most 64 iterations during conversion; an unknown
reached predicate or nonterminating loop is refused without running Python.
Executed early returns now terminate the entire callback, including returns
inside nested loops and else suites. Statements after a return are unreachable;
unbound Python locals and returns without values still refuse conversion.
Actual original Brian forward/noise VJP evidence is tracked in
`mpi-evidence/training-pure-returns-compat-20261006`.
Its complete CPU and actual Metal regression and strict audits passed:
282/475 tests respectively, with 384 GPU/191 CUDA hardware skips.
New real-control integration evidence is tracked separately in
`mpi-evidence/training-pure-real-control-20261006`.
Array-dependent conditions, dynamic ranges and mutation of argument arrays
are still refused. The loop-control extension has separate development and
freeze records in `mpi-evidence/training-pure-loop-control-20261006`.

Arbitrary attributes, mutable captured data, randomness inside callbacks,
and recursion remain explicitly refused. Dynamic loops, container mutation,
arbitrary Python/C++/CUDA callbacks,
array-valued captured quantities and target-specific implementations remain outside
this adapter's coverage. Use a call-site noise argument for stochastic inputs.
Inlining permits at most eight nested body expansions and retains the native
128-node program limit and at most 64 assignments. Typed integer paths remain detached under existing IR
rules.

Development tests cover physical native execution and independent finite
differences, lexical captures, repeated/nested calls, stochastic arguments,
neuron/synapse/regular positions and strict refusals. Frozen regression evidence
is tracked in `mpi-evidence/training-pure-functions-20261005` and the sequential
extension in `mpi-evidence/training-pure-statements-20261005`; conditional
development is tracked in `mpi-evidence/training-pure-branches-20261005`. New sequencing
opcodes require GPU capability `b2_train_sequence_v1()==1`; libraries lacking
it are refused before kernel dispatch. Eager Boolean operations additionally
require `b2_train_boolean_eager_v1()==1` before dispatch. NVIDIA runtime
and cross-host hardware acceptance are separate and remain unclaimed.

Loop development and original failures are preserved in
`mpi-evidence/training-pure-loops-20261006`; the lexical-local correction has a
separate frozen regression in `mpi-evidence/training-pure-loops-lexical-20261006`.
Metal/CUDA runtime acceptance for that freeze is pending; earlier callback
results do not establish its GPU coverage.

The lexical freeze's complete CPU run actually exited successfully and passed
its strict audit: 374 identities, 176 passed and 198 hardware-only skips.
The loop-exit and static-control extensions have their own newer frozen
sources; this result is not their runtime acceptance. Static-control evidence
is tracked in `mpi-evidence/training-pure-static-control-20261006`.

The separate bounded state-effect adapter now handles borrowed floating arrays at
canonical, unguarded full-neuron run_regularly call sites, including scalar-to-array
promotion and hidden input writeback. Its current CPU/Metal acceptance and explicit
linked/index/type limits are in `NATIVE_TRAINING_CALLBACK_EFFECTS.md`. This does not
relax the pure adapter’s mutation restrictions or admit arbitrary callback bodies.
