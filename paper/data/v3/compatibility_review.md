# Brian2 用户态兼容矩阵

本文以当前仓库的 Brian2 `2.10.1.post672` 用户 API 和语义为基线，对照
`rust_standalone` Gate 0 探针。它回答的是“现有 Brian2 模型迁移后哪些行为保持不变”，
而不是只列出 Rust 后端已经实现的内部能力。

状态：

- **原生**：reference 与 AOT 都执行，属于回归测试契约。
- **受限**：用户 API 可用，但只覆盖表中声明的子集或容量。
- **前端**：仅在 Python 建模/初始化阶段执行，不进入独立 runner。
- **未支持**：build/run 前 fail closed，不静默回退。

优先级：

- **P0 一致**：Beta 前必须保持 Brian2 的用户可见结果、顺序、单位与异常语义。
- **P1 核心**：高频建模能力，完成后显著扩大可迁移模型范围。
- **P2 扩展**：重要但不阻塞首批 LIF/CUBA/COBA/STDP 用户。
- **后续**：不进入当前单机 Rust Beta 边界。

“一致”不默认表示跨 backend 随机位流或浮点逐位相等。确定性模型按 schedule、事件顺序、
公开数组形状和约定数值容差验收；随机模型按已声明的 seed/worker 可复现契约和统计分布验收。

## Device、Network 与运行生命周期

| Brian2 用户能力 | Rust 状态 | 当前边界或差异 | 优先级 |
| --- | --- | --- | --- |
| `set_device/get_device/device` | 原生 | `rust_standalone` 正式注册；`engine=reference/aot` | P0 一致 |
| `set_device(..., threads=N)` | 受限 | AOT 单进程常驻 `std::thread` worker（无 Rayon），N 是线程池上限；并行独立 neuron/threshold、安全 target-owned `on_pre`、post-summed reduction 及无跨 target 写冲突的 pre/post plasticity；小 workload 与无法证明局部性的 pathway 自动串行回退 | P1 核心 |
| `set_device(..., engine="metal"/"cuda", numeric_mode="float32")` | 受限 | GPU 明确使用独立 float32 数值契约；`gpu_max_buffer_bytes` 是正整数、仅限 GPU 的每次执行总工作缓冲上限，默认 512 MiB，大模型必须按实际设备容量显式提高；同一上限传递到直接执行、buffer adoption、autotune 与 tuning-cache replay | P1 核心 |
| magic `run()` / `Network.run()` | 受限原生 | 非负 duration（`run(0*ms)` 完成验证但不执行 tick/callback）；同一初始化周期可依次运行完全由新对象组成的显式 Network 或 magic collection，并将 active clocks 从新模型的 `t` 重新起步；已经运行的对象仍保持单一 Network 所有权 | P0 一致 |
| 多次连续 `run()` | 原生 | 状态、clock、monitor、refractory、delay 事件延续 | P0 一致 |
| `build_on_run=False` + `device.build()` | 受限 | 可排队多个 run；排队期间禁止模型变更 | P0 一致 |
| run 之间修改状态/参数 | 受限 | 自动执行路径可读取上一段结果；完整 mutation 契约尚未冻结 | P0 一致 |
| `reinit(); activate()` | 原生 | 开始新的模型所有权周期；旧数组明确失效 | P0 一致 |
| `start_scope()` / magic collection | 受限 | 常规单 Network 用法可用；复杂对象生命周期未系统验收 | P1 核心 |
| `Network.add/remove` | 受限 | `run(0*ms)` 验证后、首次正时长执行前可添加 StateMonitor（官方 GSL 对比例程）；正时长执行后的任意对象增删仍未支持，physical model 在该边界后固定 | P2 扩展 |
| `store()/restore()` | 受限 | 自动执行模式支持内存/文件 checkpoint，恢复 neuron/synapse/refractory/monitor、绝对时间、counter RNG 及跨 run pending delay；显式排队 build 尚不支持 | P1 核心 |
| `report` / progress callback | 受限 | 自动执行支持 text/stdout/stderr/callback 的开始与结束通知；子进程内暂不周期回调 | P1 核心 |
| `profile=True` / `profiling_summary` | 受限 | 发布 simulation+recording 与 dump 粗粒度计时；尚非逐 CodeObject | P1 核心 |
| `scheduling_summary` | 前端 | Brian 对象可查看；另有 `brian2_rust.capability_report(network, duration)` 在 build 前汇总全部可静态识别的不兼容项 | P2 扩展 |
| `stop()` / keyboard interrupt / `maximum_run_time` | 受限 | 受支持的 `when='start'` NetworkOperation 可在 continuation 边界调用 `stop()`；keyboard interrupt 与 `maximum_run_time` 仍未支持 | P2 扩展 |
| whole-model C++ fallback | 未支持 | 不支持时明确报错；绝不混跑 CodeObject | P1 核心 |

## Clock、schedule 与事件顺序

| Brian2 用户能力 | Rust 状态 | 当前边界或差异 | 优先级 |
| --- | --- | --- | --- |
| `defaultclock`、显式 `dt` / `Clock` | 原生 | 任意非负 duration；每个 clock 执行 Brian 半开区间 `[Clock._calc_timestep(start), Clock._calc_timestep(end))`，因此慢 clock 在短 run 中可合法执行 0 步，并在连续 run 间保持下一调度 tick | P0 一致 |
| 多 clock / 不同 population dt | 原生 | 任意已导出 population 按绝对时间激活 | P0 一致 |
| 默认 schedule | 原生 | `start, groups, thresholds, synapses, resets, end` | P0 一致 |
| `when` / `order` 默认值 | 原生 | StateMonitor、threshold/reset、pathway 只接受已验证默认位置 | P0 一致 |
| 自定义 `Network.schedule` | reference 原生 / AOT 受限 | v24 保存 base slots 及 canonical `before_/after_` 展开；Rust 重建验证并按全局节点顺序执行；AOT 仅在 Effect Algebra 证明固定 phase 与 canonical 顺序观测等价时执行，否则 fail closed | P1 核心 |
| 任意 `when/order` 重排 | 部分基础 | reference 已按全局 `(slot, order, name, id)` 节点执行；当前对象 adapter 对 threshold/reset/pathway 仍限制 Brian 默认 event slots，AOT 对任何跨序移动逐对证明无 RAW/WAR/WAW 冲突 | P1 核心 |
| custom event | 原生 | v27 使用命名 EventStream；threshold、`run_on_event` reset、Synapses `on_event`、delay continuation、二进制事件 sidecar 与 effect dependency 均绑定具体事件名；AOT 对默认 event slots 原生生成 | P1 核心 |
| `run_regularly` / `run_at` | 原生 | 以一等 Clock table 支持独立 `dt`、任意合法 `when/order/name` 及 effect；reference 与 AOT canonical slot codegen 共用语义；连续 Subgroup runner 以父数组 guard 执行并保留局部 `i/N`；Subgroup 暂不写 `(unless refractory)` 状态 | P1 核心 |
| `NetworkOperation` / callback | 受限 | reference/AOT 自动执行模式支持至多 16 个 active、childless、`when='start'` 或 `when='end'` 的 Python callback；start callback 必须先于该 slot 的原生 effect，end callback 必须晚于该 slot 的原生 effect，run start/duration 对齐 callback/native clock，且各 callback `dt` 是所有 active native clock `dt` 的整数倍；调度器在 callback 时刻之间执行原生 continuation，同 tick 按 Brian schedule 顺序调用，支持读取最新 monitor、写回状态并调用 `stop()`；绝对 clock tick/final time 由 instance 提供，使相同长度的纯 Rust AOT continuation 复用已编译二进制；已成功但被后续 continuation 取代的 artifact 会及时清理，只保留最后成功分段；显式排队 build、带外部 C ABI Function 的编译复用与 tick 内部 slot 尚未支持 | P2 扩展 |
| 跨 run 的 delayed event | 原生 | 未来事件保留并在后续 run 投递 | P0 一致 |
| 同 tick 稳定事件顺序 | 原生 | source spike time、source index、连接创建顺序稳定 | P0 一致 |

## NeuronGroup、方程与数值方法

| Brian2 用户能力 | Rust 状态 | 当前边界或差异 | 优先级 |
| --- | --- | --- | --- |
| `NeuronGroup` | 受限 | 任意组数，合计 1–1,000,000 neurons；每组 0–32 个公开可变状态，允许纯 Synapses endpoint 及只有 threshold/refractory 的 stateless group；ODE 为 f32/f64，参数状态可为 f32/f64/i32/i64/u32/u64/bool | P0 一致 |
| 每组独立 schema/equations/dt | 原生 | 独立 CodeObjectSpec、参数、monitor、threshold/reset/refractory | P0 一致 |
| differential equations | 原生 | f32/f64、有限值；显式 storage conversion 定义 f32 写回舍入，单位在 Brian 前端检查 | P0 一致 |
| parameter equations | 受限 | 普通逐神经元可变参数、`(constant)` 或 `(constant, shared)`；支持 f32/f64/i32/i64/u32/u64/bool；可变 `(shared)` 可由唯一 `run_regularly` 做一次无条件标量赋值并在同一代码块广播读取（含 scalar `rand()`），也可由用户 Python/NetworkOperation 在 continuation 间更新并由 StateMonitor 记录；暂不支持 native writer 读取旧 shared 值、条件写入或多 writer | P0/P1 |
| deterministic subexpression | 受限 | 动态表达式（含合法的 `(shared)` 标记）由使用它的 CodeObject 展开；NeuronGroup `(constant over dt)` 作为 `before_start` 存储更新执行；reference 保持同 slot 全局名称顺序，AOT 固定 phase 只有在 Effect Algebra 证明等价时才启用 | P0 一致 |
| `exact` / `linear` / `independent` / `euler` / `rk2` / `rk4` / `exponential_euler` / `heun` / `milstein` | 原生 | deterministic ODE 的显式方法共用 Brian 抽象语句 lowering；`exact/linear/independent` 仅接受确定性方程，且 deprecated `independent` 仅开放无跨 ODE-state 依赖的原始适用域；Heun/Milstein 同样承接 Brian 展开的 SDE stages；stage snapshot 和语句提交顺序差分验收 | P0 一致 |
| 自动 method selection | 原生 | 非随机方程使用 Brian2 默认 `exact → euler → heun` 选择；选择后抽象语句进入同一通用 IR | P1 核心 |
| `gsl` / `gsl_rk2` / `gsl_rk4` / `gsl_rkf45` / `gsl_rkck` / `gsl_rk8pd` | CPU reference/AOT 原生 | NeuronGroup 的确定性 f64 ODE 使用 GSL 对应的 Euler–Cauchy 2(3)、经典 RK4 step-doubling、Fehlberg 4(5)、Cash–Karp 4(5) 或 Prince–Dormand 8(9) 方法和 scaled absolute-error 步长控制；支持 `adaptable_timestep`、`absolute_error`、`absolute_error_per_variable`、`max_steps`、`use_last_timestep`、`save_failed_steps`、`save_step_count` 及 fixed refractory 冻结。GSL 隐式方法、SpatialNeuron/Synapses GSL、linked variable、expression refractory 与 GPU/MPI 仍 fail closed | P1 核心 |
| stochastic differential equation / `xi` | 受限原生 | NeuronGroup 与 clock-driven Synapses 的 Euler additive noise 及 Heun/Milstein multiplicative noise 进入通用 IR；支持 `xi` 及 `xi_name` 命名噪声（包括经随机 subexpression 展开）；Brian 生成的单个 Wiener increment 保留为临时量供公式复用，由绝对 tick/index/draw-site 定位的 counter `randn` 执行；随机 subexpression 不可由 Monitor 或 Synapses endpoint 事后重算；reference/AOT 同 seed 逐位一致，不复制 NumPy/C++ 随机位流 | P1 核心 |
| threshold/reset 可选 | 原生 | 多语句 reset，按源顺序执行 | P0 一致 |
| 多 threshold/custom events | 原生 | 每个 population 使用独立 canonical event table，多个 threshold/reset 通过 event name 绑定；AOT 当前要求 Brian2 默认 threshold/reset event slots | P1 核心 |
| fixed scalar refractory | 原生 | Brian2 timestep 语义；支持公开 `lastspike/not_refractory` | P0 一致 |
| `(unless refractory)` | 原生 | 状态更新和 on_pre 条件写入均保持 | P0 一致 |
| expression/boolean/per-neuron refractory | 原生 | v28 reference、v30 AOT 直接执行 Brian 生成的 duration/boolean refractory 状态更新；`unless refractory`、threshold 和 on_pre 共用 `not_refractory`，不会误用 fixed-period 截止 tick 快路径 | P1 核心 |
| `Subgroup` 作为 Synapses endpoint | 原生 | 仅连续 slice；i/j 保持 subgroup 局部坐标 | P0 一致 |
| linked variables / `linked_var` | 受限原生 | v36 reference/AOT 支持 NeuronGroup 间可变状态的同尺寸、1→N、固定数组/subgroup、本地整数动态索引及 identity self-link。reference 与非 MPI AOT 另支持 Synapses 链接到固定 source index 的确定性 NeuronGroup subexpression（含 1→N 广播），进入全局 effect DAG，并可由 Synapses StateMonitor 重建记录；动态 synapse 映射、MPI/GPU synapse link、non-identity self-link 与 constant source 尚未开放 | P1 核心 |
| integer/boolean model state | 原生 | v33/v34 公开 bool 与 i32/i64/u32/u64 SoA、类型化 literal/cast/比较/赋值及二进制 Dump；reference/AOT 保留包括 >2^53 的整数位值 | P1 核心 |
| float32 numeric profile | 原生 | v31 reference/AOT 使用真实 `Vec<f32>` population/synapse SoA，显式 f32↔f64 IR 转换，并以 Dump v3 原生宽度 memmap；Rust 两后端逐位一致 | P1 核心 |

## 表达式、namespace 与赋值

| Brian2 用户能力 | Rust 状态 | 当前边界或差异 | 优先级 |
| --- | --- | --- | --- |
| SI units / dimension checking | 原生 | Python 前端检查；B2IR v23 保存 statement/TimedArray dimensions，Rust 从符号开始逐表达式重新推导并校验 | P0 一致 |
| scalar namespace constants | 原生 | 构建时冻结到 instance | P0 一致 |
| per-neuron constant 参数 | 原生 | 必须在 equations 中声明 | P0 一致 |
| `i/N/t/dt`、`int()`、`timestep()` | 受限原生 | `i/j/N/N_pre/N_post` 使用 `index`、`timestep()` 与有符号偏移使用 `tick`；进入浮点运算时显式 conversion；i32/i64/u32/u64 状态运算使用已验证的定宽 wrapping 语义 | P0 一致 |
| Brian 内置确定性数学函数 | 原生 | `abs/clip`、三角/双曲、指数/对数、`sqrt/sign/ceil/floor/exprel` 全部覆盖；Brian state updater 展开的 dimensionful `clip(x, 0, inf*unit)` 保留多态零量纲，并按 finite-only B2IR 将内置 `inf` 饱和为最大 f64 | P0 一致 |
| 比较和 `and/or/not` | 原生 | f64/bool/index/tick typed IR；同类逻辑值直接比较 | P0 一致 |
| 字符串初始化和 slice/array 赋值 | 前端 | 建模阶段 NumPy CodeObject；结果数组由 Device 所有 | P0 一致 |
| 初始化 `rand()/randn()` | 前端 | `seed()` 控制 NumPy frontend；不属于 runner RNG | P0 一致 |
| 运行期 `rand()/randn()` | 原生 | counter uniform + Box–Muller normal；reference/AOT/分段重放一致，不承诺与 C++/NumPy 相同位流 | P1 核心 |
| 运行期 `poisson(lambda)` | 原生 | 小均值 inversion、大均值 PTRS；按 draw-site/tick/index 定位并校验有限非负 lambda | P1 核心 |
| namespace `BinomialFunction` | 原生 | exact inversion 与可选 normal approximation；PoissonInput 复用同一 IR | P1 核心 |
| user-defined `Function` | 受限原生 | v29 Function Contract 支持带静态参数/返回单位的单 `return` 纯表达式；f64/i64/bool 参数与返回值由 Rust 独立验证 dtype/dimension/effects，reference/AOT 可从同一 body 执行；v37 支持显式 `@implementation("b2ir-c-abi-v1", ...)` 的内容寻址 C source/symbol，AOT 编译独立 object 并静态链接，native-only 在 reference fail closed；未声明 native ABI 的闭包/global 捕获、控制流、随机/有状态调用仍拒绝 | P1 核心 |
| `TimedArray` | 原生 | 1D `f(t)` 与 2D `f(t, i)`；使用 Brian 的 owner-clock upsampling、边界 clamp 和 C-order 布局；有限 float64 数组；大型 AOT/MPI 表以内容寻址的校验二进制资源保存，避免生成巨型 Rust 字面量 | P1 核心 |
| 一般整数运算与非 f64 dtype | 受限原生 | 支持 f32/f64/i32/i64/u32/u64/bool 状态、显式 cast、定宽加减乘/取模/整除/比较；portable Function 支持精确 i64/bool 签名；ODE 仍只允许浮点，尚无混合宽度整数函数泛型 | P1 核心 |

## 随机性与输入源

| Brian2 用户能力 | Rust 状态 | 当前边界或差异 | 优先级 |
| --- | --- | --- | --- |
| `seed(integer)` 建模重放 | 受限 | topology 与前端初始化可复现 | P0 一致 |
| `Synapses.connect(p=scalar)` | 前端 | Python source-major 物化显式 i/j；同 seed 可重放 | P0 一致 |
| `Synapses.connect(j=generator)` | 前端 | postsynaptic `range`/`sample`、单条件、`rand()` 与 `skip_if_invalid`；物化后进入相同显式拓扑预算；presynaptic generator 仍拒绝 | P0 一致 |
| 运行期 seed 状态 | 原生 | seed 写入 B2IR；绝对 tick 定位使连续/分段 run 一致 | P1 核心 |
| worker-count-independent counter RNG | 原生 | key-based draw；并行 lane 仍由 seed/stream/absolute tick/index 定位，与 worker 分片无关 | P1 核心 |
| `PoissonGroup` | 受限 | scalar/per-neuron constant，或基于 `t/i`、标量常量、TimedArray 的运行期 rate；stored per-neuron `rates` 可由 `run_regularly` 更新并以独立 clock `StateMonitor` 记录；可作 Synapses source | P1 核心 |
| time-varying/string `PoissonGroup.rates` | 原生 | 支持 `t/i`、标量 namespace 常量及 1D/2D `TimedArray`；每 tick 在 threshold IR 内求值 | P1 核心 |
| `PoissonInput` | 受限 | full NeuronGroup 或连续 subgroup；constant `N/rate`、constant 或表达式 weight；默认 synapses/order=0；subgroup 写入由父群体窗口掩码限制；v32 将逐步概率存为可替换的 Instance 标量 | P1 核心 |
| `SpikeGeneratorGroup` | 原生 | 静态或周期预置事件、SpikeMonitor、Synapses source、连续/分段 run；按 `(tick, index)` 排序并验证同神经元同 tick 不重复 | P1 核心 |
| `BinomialFunction` | 原生 | PoissonInput 与用户 namespace 函数均可进入 B2IR；exact inversion 或显式 normal approximation，不复制 Brian2 位流 | P1 核心 |

RNG 兼容目标分三层：

1. 同一个 Rust artifact、seed 和 logical model 在 1/8/32 worker 下产生相同 draw；
2. reference 与 AOT 对同一 B2IR 逐位一致，多段 run 与一次连续 run 一致；
3. 与 NumPy/C++ 保持事件概率和统计分布，不承诺复制其随机位流。若未来需要 Brian2
   compatibility stream，应作为独立 numeric/RNG profile 实现，而不是改变 counter profile。

## Synapses 与连接

`connect_binary_csr(S, path, parameters={name: column})` 新增通用 empirical CSR 路径：最多 1 亿条有向边/文件；constant f64 逐边参数、uniform pathway delay、单段运行。原始数组在 Python 中延迟物化，AOT 将数据流式写入自包含 instance。FlyWire 全图导入、限制及公平对比命令见 [FLYWIRE.md](FLYWIRE.md)。


| Brian2 用户能力 | Rust 状态 | 当前边界或差异 | 优先级 |
| --- | --- | --- | --- |
| self/cross-population `Synapses` | 受限 | 任意对象数，受 IR/数组/运行预算约束；source 可为 NeuronGroup、PoissonGroup、SpikeGeneratorGroup 或连续 subgroup；target 可为这些 population（连续 subgroup 仅 NeuronGroup），stateless target 支持 post event pathway、但自然没有可写 post state/summed destination | P0 一致 |
| `connect()` 默认全连接 | 原生 | 固定 topology | P0 一致 |
| 显式 `i/j`、重复 pair、追加 connect | 原生 | 支持 Brian2 标量 `n>=0`，按 pair 就地展开并保留创建顺序 | P0 一致 |
| condition expression | 受限前端 | 支持 `i/j/N_pre/N_post`、source/target 状态的 `_pre/_post` alias、标量 namespace、单位和受限纯数学/比较表达式；Python source-major 物化显式 topology | P0 一致 |
| scalar `p` | 前端 | 使用 NumPy 抽样后保存显式 topology | P0 一致 |
| `brian2_rust.connect_fixed_total(S, E, seed=..., initializers=..., delay_initializer=...)` | 受限原生 | B2IR v23 只保存 E/seed 与 versioned clipped-normal descriptors，Rust 内 source-major 构建；允许 autapse/multapse；支持逐边 `(constant)` 参数和 heterogeneous delay；仍限无可变逐边状态、单个 `on_pre` delay initializer，且每次 activation 只运行一段 | PD14 核心 |
| string/generator `i/j/p` | 受限 | `j=` 支持由 `i`、整数常量及整数 `_pre` 状态组成的确定性加减表达式，如 `j='i'`、`j='label_pre'`；condition/all-to-all 的 `p=` 支持与 condition 相同的状态/单位/纯数学表达式并按 Brian 的 `rand() < p` 语义物化；`i=` generator、generator comprehension、显式 i/j 或 j-generator 搭配 string `p` 仍拒绝 | P1 核心 |
| `n != 1` / `multisynaptic_index` | 原生 | `n` 多重边及 Brian 生成的命名 occurrence index 均进入 canonical edge identity；IR 中以精确 f64 只读参数执行 | P1 核心 |
| per-edge mutable/constant/shared variable | 原生 | 无 flag 的逐边变量可写；constant/shared 运行期间只读；float64 | P0 一致 |
| clock-driven synaptic ODE | 原生 | 最多 32 states；支持 Brian 默认 method selection，以及显式 `exact/linear/independent/euler/rk2/rk4/exponential_euler/heun/milstein`（`independent` 同样限无跨 ODE-state 依赖），包括受支持的 `xi` 随机方程；可读取 `_pre/_post` 神经元状态，确定性 endpoint subexpression 会按对应 i/j 域内联 | P0 一致 |
| event-driven synaptic ODE | 受限 | Brian 展开的独立一维 event-driven 更新与 `lastupdate`；pre/post 触发并可跨 run 延续 | P1 核心 |
| synaptic subexpression | 受限 | 动态及标量 `(shared)` subexpression 就地物化；`(constant over dt)` 作为独立 `before_start` CodeObject 存储并逐 source clock 更新；reference 使用全局 scheduler，AOT 跨序执行必须先通过 effect 等价证明 | P1 核心 |
| 一个默认 `on_pre` | 原生 | default `spike`、default schedule | P0 一致 |
| `on_pre` 写突触前状态 | 受限 | 支持可变 `_pre` 状态；refractory 冻结状态写入仍拒绝；target-owner 并行路径保守回退串行 | P1 核心 |
| `on_post` | 受限 | default `spike`/order=1，target CSR；支持独立标量/逐突触 delay，可写逐突触状态及 postsynaptic 可变状态；同一 post state 的无后缀与 `_post` 拼写会规范化为单一 B2IR alias | P1 核心 |
| multiple pre/post pathways / pathway delay | 原生 | 同一 Synapses 可有多个命名 pre/post pathway；按 Brian 默认 order/name 排序，各自使用 endpoint clock、标量或逐突触 delay queue，并跨多段 `run()` 保留 pending event | P1 核心 |
| scalar/per-edge heterogeneous delay | 原生 | 最近 tick 量化，跨 run 延续 | P0 一致 |
| summed variables | 受限 | full population 与连续 subgroup `_pre/_post`，可由没有 event pathway 的纯 `summed` Synapses 提供；按目标 clock/默认 order 只清零 endpoint window，并按连接创建顺序求和；独立 post reductions 可共享 target-owner dispatch；运行中不可观测的 destination 仅在 run/checkpoint 末 tick 求值，方程读取或 Monitor 观察时保持逐 tick 语义；AOT 固定 phase 的跨对象移动需通过 effect 等价证明 | P1 核心 |
| Synapses→Synapses endpoint | 受限原生 | reference 支持单层 edge domain：population→edge 的 pre-event 状态写入、edge↔population 的 summed reduction，以及双侧 edge→edge 的 post-summed reduction；edge domain 的 i/j 是被引用 Synapses 的创建顺序。嵌套 edge domain、edge endpoint 上的一般 clock/event-driven ODE 与 AOT/GPU/MPI 仍 fail closed；Brian2 的 `SynapticSubgroup` 仅是索引视图，本身不能作为 endpoint | P2 扩展 |
| structural plasticity | 未支持 | 运行期间 topology 固定 | 后续 |

## Monitor 与结果对象

| Brian2 用户能力 | Rust 状态 | 当前边界或差异 | 优先级 |
| --- | --- | --- | --- |
| `StateMonitor` | 受限 | 每组可选、任意多个；source 可为 full NeuronGroup 或连续 subgroup，subgroup 的 local record 顺序映射到父群体 physical trace；记录 ODE、可变或 constant/shared 参数；支持独立 monitor clock 与 canonical `when/order`，budget 和预分配以 monitor clock 的实际样本数计算；同一 NeuronGroup source 的多个 monitor 当前须共享 schedule；独立 clock 暂不与 rolling window 组合 | P0 一致 |
| StateMonitor record subset/order/duplicates | 原生 | 公共布局保持 `(recorded_index, time)` | P0 一致 |
| `SpikeMonitor` | 受限 | full spiking group 或连续 subgroup，可在同一父群体上监视多个不同 endpoint；subgroup 结果恢复局部 i/count；NeuronGroup 可同时记录 ODE state、parameter 与 linked variable | P0 一致 |
| 多段 monitor append | 原生 | 时间与数组连续追加 | P0 一致 |
| `recording_window_steps=N` | 受限扩展 | 显式 opt-in rolling window；State/SpikeMonitor 只暴露最后 N 个 source-clock tick，原生缓冲、Dump 和 Python 回填均有界；默认仍完整追加 | 长时程核心 |
| `StateMonitor` 记录参数/表达式/突触 | 受限 | NeuronGroup 的可变及 constant/shared 参数、确定性 subexpression（含比较与 `int`）、`t/dt/i/N`、1D/2D `TimedArray` 与 fixed refractory 的 `lastspike/not_refractory` 已支持；源状态于配置的 schedule slot 采样，派生读数由 Python adapter 按自包含表和冻结 SI 参数计算。reference 与非 MPI AOT 均支持以 Synapses 为 source、每个 monitor 独立 clock/schedule、record subset/order、逐突触可变/clock-driven/event-driven 状态及 pre/post neuron-state alias；reference 另支持受限 fixed-index linked subexpression。MPI/GPU AOT、突触 constant/shared 参数与普通突触 subexpression仍明确拒绝 | P1 核心 |
| `SpikeMonitor` 自定义变量 | 受限原生 | NeuronGroup 的 state/parameter/linked variable 通过同一 spike event 的 typed sidecar 回填，并校验 i/t 对齐；PoissonGroup/SpikeGeneratorGroup 仍只支持 i/t/count | P1 核心 |
| `EventMonitor` | 原生 | 支持 i/t/count 与所有公开状态/参数/linked dtype；事件数据通过校验后的 memmap 二进制 sidecar 回填；AOT 当前要求默认 `after_thresholds/order=1` | P1 核心 |
| `PopulationRateMonitor` | 受限原生 | full NeuronGroup 或连续 subgroup、默认 `when='end'/order=0`、f32/f64；subgroup 按窗口过滤尖峰并以 endpoint 大小归一化；保留公开 `rate/t/N`，由同一 B2IR 的逐 tick 尖峰记录还原；与 NumPy 后端确定性数组差分通过 | P1 核心 |
| memory result | 原生 | versioned binary dump + NumPy memmap/view | P0 一致 |
| streaming disk sink | 受限扩展 | reference/AOT 自动执行模式可用 `monitor_streaming_steps=N` 把一次 run 切成有界 continuation；State/Event/Spike/PopulationRateMonitor 每段落盘到 `monitor-stream/chunk-*`，磁盘保留完整历史，Brian 对象只保留最后一段；暂不与 NetworkOperation、显式排队 build、GPU/MPI 或 profile 组合 | P2 扩展 |
| lazy result / streaming reader | 原生扩展 | `open_monitor_stream(path)` / `iter_monitor_chunks(path, name)` 按 chunk 惰性校验并读取；StateMonitor 数值采用适合顺序处理的 `(time, recorded_index)` 布局 | P2 扩展 |

## 其他 Brian2 用户对象

| Brian2 用户能力 | Rust 状态 | 当前边界或差异 | 优先级 |
| --- | --- | --- | --- |
| `SpatialNeuron`、Morphology/Soma/Section/Cylinder | 受限原生 | CPU reference/AOT 支持树状 morphology 的半隐式电缆求解、point current、subgroup 初始化及 StateMonitor；当前要求 f64、默认 spatial schedule，不支持 GSL、GPU/MPI | P1 核心 |
| Import/Export connectivity formats | 未支持 | 可在 Python 侧先转为显式 i/j | P2 扩展 |
| inserted native code / external native library | 未支持 | artifact threat model 前不开放 | 后续 |
| custom CodeObject / codegen target | 未支持 | Rust 只执行完整、已验证 B2IR | 后续 |
| runtime/Cython backend 混跑 | 未支持 | 明确禁止，避免 schedule/RNG/state 分裂 | P0 一致 |

## 当前首批兼容完成线

首批 Rust Beta 不以覆盖 Brian2 全部 `__all__` 为目标。它至少要求：

1. 当前 LIF/CUBA/COBA 能力继续保持四后端差分，并以相同线程数对 Rust AOT 与 C++ OpenMP 建立性能门；
2. 已完成 counter RNG/Poisson、常用输入源、动态及存储 subexpression、多 pathway/delay、基础 event-driven plasticity、full/subgroup summed variables、checkpoint、有界 rolling monitor，以及 reference 的全局 `(slot, order, name, id)` schedule/effect DAG；下一阶段开放 adapter 自定义调度并让 AOT 优化由 DAG 合法性证明驱动；
3. 对所有未支持功能在 build 前输出 capability report，绝不静默改变 schedule、RNG、
   result shape 或切换 backend；
4. P0 项进入自动测试；P1 项只有在实现和差分测试同时落地后才能从“未支持”改为“受限/原生”。

这份矩阵是兼容性契约草案。实现边界变化时，必须在同一个提交中更新矩阵和对应测试。
