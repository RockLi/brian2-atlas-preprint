# Brian2-Rust Standalone 高性能执行平台架构方案（V4.0）

**日期：2026-09-03**

**评审基线：Brian2 `master@cf0b25cc`、原始方案、共享会话 V3.0、当前会话仓库审计与技术调研**

讨论输入：[共享会话「评估实施方案」](https://chatgpt.com/share/6a99511c-5a6c-83eb-8393-cb7e2a9ee76d)与原始附件。本文为合并后的建议稿；源代码事实、架构建议和待实测目标分开陈述。

**项目定位：语义兼容优先、单机性能优先、内存有界、可恢复、按场景显式分布式**

**立项结论：立即启动 Gate 0；通过后进入 Gate 1，后续按正确性、性能和真实需求逐 Gate 放行**

---

## 0. 执行摘要

本项目不是把现有 C++ 模板逐行翻译成 Rust，也不是在第一版承诺 100% Brian2 兼容、所有模型 2–4 倍提速或任意规模分布式。项目的正确目标是：

> 保留 Brian2 Python 建模前端，以纯语义、版本化的 Canonical B2IR 隔离模型与执行实现；先交付可独立运行、结果可验证的 Rust standalone，再解决单机并行、有界 Monitor 和 crash-safe checkpoint；只有单机容量不足或经评估确有强扩展价值时，才显式启用独立的 cluster backend。

最终替换路线不是一次性删除 `cpp_standalone`，而是：

```text
实验性 opt-in
→ 支持范围内推荐 Rust
→ 支持范围内默认 Rust
→ 至少两个稳定版本双栈
→ 再依据兼容率、性能和用户迁移数据决定 C++ 后端是否退役
```

### 0.1 核心裁决

| 议题 | 最终决定 |
|---|---|
| Rust standalone | 项目主线，Go |
| Rust runtime / PyO3 | 共享核心的后续交互入口，不阻塞 standalone 交付 |
| Canonical Typed B2IR | 最重要的长期资产 |
| Reference Executor | 正确性 oracle；不承担性能目标，不掩盖不支持的语义 |
| Native B2K Artifact | 正式性能路径需要；具体 codegen 在基准后选择 |
| Cranelift | 候选而非既定唯一方案 |
| libtcc | 不进入默认或静默 fallback 路径 |
| CSR/CSC、Effect Algebra | 核心方向 |
| 图着色 | 仅保留为未来可选策略，不作为主算法 |
| 有界 Monitor / Checkpoint | 主线能力；disk/window 显式选择，从 Gate 1 预留接口 |
| Ensemble | 工作负载编排方式，不是第四个计算后端 |
| Cluster | 独立、显式启用，区分 capacity 与 speed |
| 1D target-owned | 首个分布式物理计划，但只支持经 Effect Analyzer 证明的语义子集 |
| 2D edge partition | 稠密、可归约模型的后期专用计划 |
| GPU / Autodiff / WASM | 独立立项，不阻塞 standalone 替换 |

### 0.2 核心资产优先级

1. Canonical B2IR 与兼容性契约；
2. Schedule、Effect、事件时序的正确性；
3. Reference Executor 与差分测试；
4. Native Artifact 构建、缓存和验证链路；
5. Target-owned / deterministic reduction；
6. Persistent phase-aware executor；
7. Bounded Monitor 与 crash-safe checkpoint；
8. 双后端迁移机制；
9. 后置、场景化的 Cluster Planner。

### 0.3 相对原稿与共享会话 V3 的关键修订

- 把 Logical Tile 从 Canonical B2IR 移到独立 LogicalPlan；
- 将 Ensemble 从“第四后端”改为正交的 workload orchestration；
- Cranelift 从既定首选降为需要 bake-off 的候选；
- 将模型定义、运行实例、执行计划、放置计划和 native artifact 分开 hash；
- Reference fallback 只覆盖 B2IR 已表达语义，其他情况报错或显式重建整个模型后回退 C++；
- reproducible 模式冻结任务与归约计划，禁止运行时任意自适应切换；
- 分布式构网拆成 canonical edge stream 与 regenerable recipe；
- target-owned 明确限定为 Effect Analyzer 证明安全的子集；
- zero-delay 通信边界由 schedule/effect DAG 推导，不硬编码一个通用 MPI 循环；
- `stop_with_checkpoint` 增加独立路径/预留空间前提；
- 不同 rank 数恢复从 v1 承诺中移除；
- B2K 增加不可信 native artifact 的隔离、校验和平台约束。
- 不承诺普遍的 state/threshold/reset fusion；任何融合先证明不改变可观察顺序。

---

## 1. 已核实的现状与问题边界

审计基于当前 checkout `cf0b25cce13b620aac740873e88ba91061a70275`。现状不是“所有计算都串行”，而是若干关键阶段和生命周期能力存在明确缺口。

| 现状 | 代码证据 | 对本项目的含义 |
|---|---|---|
| 默认 schedule 为 `start → groups → thresholds → synapses → resets → end`，同槽再按 `order`、对象名排序 | [`network.py`](brian2/core/network.py#L864) | 并发与 fusion 必须保持精确可观察顺序 |
| state update、reset 已有 OpenMP 并行 | [`stateupdate.cpp`](brian2/devices/cpp_standalone/templates/stateupdate.cpp#L14)、[`reset.cpp`](brian2/devices/cpp_standalone/templates/reset.cpp#L15) | 不能把 Rust 收益建立在“旧后端完全串行”的错误基线上 |
| threshold 扫描仍串行 | [`threshold.cpp`](brian2/devices/cpp_standalone/templates/threshold.cpp#L18) | 适合做稳定的并行 compaction |
| summed variable 只并行清零，scatter-add 串行 | [`summed_variable.cpp`](brian2/devices/cpp_standalone/templates/summed_variable.cpp#L11) | Target CSC / reduction 是高价值优化点 |
| 某些 synaptic write 会把传播放入 `omp master`；其他路径可以并行 | [`synapses.cpp`](brian2/devices/cpp_standalone/templates/synapses.cpp#L20) | 应做 Effect Region Splitting，而非笼统重写 |
| SpikeQueue 使用嵌套 vector，并按 OpenMP 线程建立 queue | [`spikequeue.h`](brian2/synapses/spikequeue.h#L14)、[`synapses_classes.cpp`](brian2/devices/cpp_standalone/templates/synapses_classes.cpp#L15) | Source CSR 和按 delay 分组有明确内存、局部性价值 |
| StateMonitor 随采样扩容；文档示例单变量即可约 8 GB | [`statemonitor.cpp`](brian2/devices/cpp_standalone/templates/statemonitor.cpp#L5)、[`recording.rst`](docs_sphinx/user/recording.rst#L288) | OOM 首先是数据生命周期与输出策略问题，不是语言问题 |
| standalone 不支持 `Network.store/restore` | [`device.py`](brian2/devices/cpp_standalone/device.py#L1981) | Checkpoint 是独立的一等语义能力 |
| OpenMP 路径本身仍提示测试不足、可能不准确 | [`device.py`](brian2/devices/cpp_standalone/device.py#L758) | 新后端必须把多线程 conformance 纳入阻断式 CI |

现有扩展点可复用：`Device`、`CodeGenerator`、`CodeObject`、模板和 `device_override` 已经构成清晰入口，分别见 [`Device`](brian2/devices/device.py#L86)、[`code_object`](brian2/devices/device.py#L285)、[`device_override`](brian2/core/base.py#L331) 与 [`CPPStandaloneDevice`](brian2/devices/cpp_standalone/device.py#L144)。第一版应作为外部/可选 Device 孵化，避免先大改 Brian2 前端。

Rust Device 应直接继承基础 `Device`，不能继承已经耦合 C++ 文件、Makefile、Jinja 模板和 results 目录的 `CPPStandaloneDevice`。真正的 lowering 切口位于 Brian2 完成变量、函数、索引和单位解析之后、渲染目标语言字符串之前；现有 [`create_runner_codeobj`](brian2/codegen/codeobject.py#L272) 和 [`Statement`](brian2/codegen/statements.py#L6) 可以提供基础信息，但需要新增后端无关的 `CodeObjectSpec`。仅重写 `Device.code_object()` 不足以解除 `templater` 耦合。

### 1.1 尚未证实的项目数据

上述源码能证明扩容、串行阶段与生命周期限制，但尚不能断言用户实际 OOM 全部由 Monitor 引起。本次未运行真实负载的 RSS profile、性能 benchmark 或 MPI 实测。Gate 0 必须收集：

- 至少三个代表性模型的 neuron/edge 数、变量 dtype、连接生成方式、delay 分布、发放率与 burst 上界；
- Monitor 的变量、record 集合、采样间隔、模拟时长，以及是否读取完整 ndarray；
- OOM 出现在 Python 建模、构网、编译、初始化、运行、输出还是结果读取；
- 峰值 RSS/分阶段内存、分阶段耗时、线程配置，以及目标 CPU、RAM、磁盘、网络和调度器；
- 哪些现有模型必须不改脚本，哪些可接受新 RNG、lazy result 或显式分布式配置。

立项数字应来自这些负载。Rust 的内存安全减少一类错误，不自动减少存储量，也不自动比经过同等布局与并行优化的 C++ 更快。

---

## 2. 项目目标、非目标与替换终点

### 2.1 按优先级排序的目标

1. **语义闭环**：一个受限但明确的 Brian2 子集可以从 Python lowering 到 B2IR，并由独立 Rust runner 重放。
2. **可靠性**：长时间运行时 Monitor 内存有界，进程可在一致性切点恢复。
3. **单机性能**：解决 threshold、synaptic effect、summed variable、delay queue 和 CodeObject 调度热点。
4. **兼容迁移**：逐项扩大支持面，并在不支持时构建前明确报告。
5. **容量分布式**：运行单机内存无法容纳的真实模型。
6. **速度分布式**：只服务通信/计算比合理的模型。

### 2.2 第一主线不包含

- 自动微分、EventProp、surrogate training；
- GPU、WebGPU、WASM；
- structural plasticity 的实际实现；
- 任意 C++ 用户代码自动翻译；
- optimistic Time Warp；
- MPI rank 故障后的原作业内在线恢复；
- 千核、十亿神经元等没有真实模型和硬件约束的宣传目标。

### 2.3 “替换完成”的定义

Rust 成为默认后端之前，至少满足：

- Tier A/B 兼容矩阵通过；
- 目标平台的 conformance、性能和 soak test 通过；
- unsupported feature 在 build 前可解释地拒绝；
- checkpoint、结果读取和多次运行生命周期稳定；
- 至少一个稳定版本以 opt-in 方式获得真实用户验证；
- whole-model C++ fallback 和迁移文档可用。

删除 C++ backend 不属于本项目早期承诺；第一目标是替换默认路径，而不是删除恢复路径。

---

## 3. 产品形态：三个后端、两个执行档位、两种编排方式

### 3.1 后端

| 后端 | 定位 | 是否依赖 Python 驱动 | 是否含 MPI |
|---|---|---:|---:|
| `rust_runtime` | Jupyter、调试、短模拟 | 是 | 否 |
| `rust_standalone` | 默认正式模拟、独立 CLI | 否 | 否 |
| `rust_cluster` | 超大单模型 | 否 | 是，独立包/镜像 |

三者共享 B2IR validator、算子语义、状态布局和 checkpoint 协议，不能演化成三套引擎。

这是长期产品划分，不是第一版同时开发三个完整后端。MVP 只有 standalone；PyO3 最初可用于转换和诊断，完整交互 runtime 在核心稳定后单独排期。Cluster 按 D0/D1/D2 放行。

### 3.2 执行档位

```text
reference = runner + B2IR
native    = runner + B2IR + compatible B2K
```

`runner + B2IR` 已符合“脱离 Python 独立运行”的 standalone 定义；正式高性能运行通常需要模型专用 B2K。Reference 只允许执行 B2IR 已完整表达的语义，不能成为任意 Python/C++ 回调的万能 fallback。

### 3.3 编排方式

- `single`：一个模拟；
- `ensemble/sweep`：多个参数、seed 或实验组，每个 job 仍调用 standalone 或 cluster。

Ensemble 通常比拆分单个模型更容易近线性扩展，应优先推荐给参数扫描用户。

### 3.4 API 原则

下面仅是方向性 API，Gate 0 后再冻结：

```python
set_device(
    "rust_standalone",
    build_on_run=False,
    numeric_profile="reproducible",
)

network = build_model()  # 用户的建模函数，在 Device 选定之后创建对象
network.run(10 * second)  # standalone 模式下先记录运行段
report = b2rust.plan(network, target="local")

device.build(
    directory="simulation",
    artifacts=["baseline", "avx2"],
    run=False,
)
```

Cluster 必须显式选择：

```python
set_device(
    "rust_cluster",
    objective="capacity",  # 或 speed
    resources=cluster_profile,
)
```

不得在 `rust_standalone` 背后自动启动 MPI，也不得静默改变 numeric profile。

由于 `Variables` 在对象创建时就绑定当前 Device，使用约定应明确为“先 `set_device()`，再创建 NeuronGroup/Synapses/Monitor”；创建对象后切换到 Rust Device 应被检测并拒绝，避免 Python/C++/Rust 数组所有权分裂。

---

## 4. 总体架构与对象边界

```text
Brian2 Python API
       │
       ▼
Frontend Adapter / Lowering
       │
       ├── ModelDefinition ─────── DefinitionHash
       └── ModelInstance   ─────── InstanceHash
                  │
                  ▼
        Canonical B2IR Validator
                  │
          Capability Report
                  │
          ┌───────┴────────┐
          ▼                ▼
 Reference Executor   Effect + Topology Planner
                           │
                    Logical Execution Plan
                           │
                    Physical Placement Plan
                    ┌──────┴──────┐
                    ▼             ▼
               Local Plan     Cluster Plan
                    │             │
                    └──────┬──────┘
                           ▼
               KernelCodegen / B2K
                           │
                           ▼
                 Persistent Executor
                    ├── bounded sinks
                    └── checkpoint
```

### 4.1 必须分开的六类对象

| 对象 | 内容 | 不应包含 |
|---|---|---|
| `ModelDefinition` | 方程、函数、clock、schedule、event、effect、连接 recipe、稳定 ID 规则 | CPU、MPI rank、输出路径 |
| `ModelInstance` | 具体 shape、参数、初始数组或初始化程序、拓扑实例或生成 seed | codegen 版本、物理 tile |
| `LogicalPlan` | 语义安全的 region、reduction、逻辑 tile/owner、通信依赖 | rank 编号、NUMA core |
| `PhysicalPlan` | CSR/CSC 布局、tile→worker/rank、buffer、通信算法 | 模型语义 |
| `KernelArtifact/B2K` | 机器代码、目标 ABI/ISA、specialization metadata | 运行输出、checkpoint |
| `RunManifest` | 有序 RunProgram、资源上限、输出、checkpoint、所有输入 hash | 可变模拟状态本身 |

Logical Tile 不属于 Canonical B2IR。B2IR 提供 Stable Global ID、region hint 和可分片 connectivity recipe；`LogicalPlan` 再据此生成 tile，`PhysicalPlan` 最后映射到 worker、NUMA domain 或 MPI rank。

`RunProgram` 是必须保留的控制层：按顺序记录初始化赋值、`seed()`、支持的参数/active/dt 更新、`RunSegment(duration)` 以及未来的 store/restore 命令。不能把多次 `run()` 和运行段间赋值折叠成一个“最终初值 + 总时长”，也不能提前在 Python 求值本应在构网后或 runner 中执行的表达式。Gate 1 只支持一个运行段；其他命令按兼容矩阵逐项开启。

### 4.2 Hash 与缓存

不能把所有初始状态塞进一个 `SemanticModelHash`，否则每次换 seed 或参数都会无谓重编译。至少区分：

```text
DefinitionHash
  canonical model semantics

InstanceHash
  DefinitionHash
  + topology realization
  + runtime parameters / initial state / seed bindings

LogicalPlanHash
  InstanceHash
  + effect regions / logical tiles / ownership
  + canonical event order / reduction plan / causal dependencies

KernelPlanHash
  DefinitionHash
  + specialization-relevant shapes/constants
  + code-relevant layout / execution strategy
  + numeric/math profile

PlacementHash
  LogicalPlanHash + KernelPlanHash
  + topology layout
  + worker/rank placement
  + buffers / communication strategy

ArtifactHash
  KernelPlanHash
  + engine / codegen / ABI version
  + target triple / CPU features / math backend

RunConfigHash
  PlacementHash + ArtifactHash (or reference engine build)
  + ordered RunProgram + output/checkpoint policy

RunId
  unique execution-attempt ID linked to RunConfigHash
```

只有真正影响生成代码的参数进入 `ArtifactHash`；rank 数、输出路径和普通运行期参数不应导致无谓重编译。Ensemble 中不同 seed 应尽可能复用同一 B2K。超大分片拓扑的 `InstanceHash` 由完整版本化 recipe+seed 或各 shard digest 的确定性 Merkle/root hash 得到，不能要求 Rank 0 重新物化全图。Recipe 必须绑定生成算法、RNG 与枚举版本；相同 recipe 文本配上不同生成语义，不能误命中同一实例缓存。

---

## 5. Canonical B2IR 语义契约

### 5.1 必须表达的语义

| 模块 | 内容 |
|---|---|
| Symbols | 对象、变量、函数、event、pathway 的稳定语义 ID |
| Tensors | dtype、shape、单位维度/缩放、mutability、index domain |
| Clocks | 每个 clock 的 dt 位表示、tick、起止状态及比较规则 |
| Schedule | `before_* / slot / after_* / order / name` 的确定全序 |
| Expressions | 强类型 DAG/SSA、statement order、数学 intrinsic |
| Effects | read/write/reduction、索引域、别名、顺序要求、时间可见性 |
| Events | 触发、pathway、量化 delay、delivery phase、payload |
| Topology | 显式边或声明式 recipe、连接调用序号和稳定枚举规则 |
| RNG | stable call-site ID、entity ID、draw schema、distribution version |
| Observability | monitor 采样时点、可观察中间状态 |
| Extensions | pure/random/stateful/external function effect declaration |

单位检查仍由 Brian2 前端完成；B2IR 保存足够的维度和缩放信息用于验证与诊断，热循环使用规范化数值。

### 5.2 时间模型

不能假定所有 clock 都可映射到一个简单的全局整数 tick。建议：

```text
TimeKey = {
  clock_id,
  clock_tick,
  schedule_phase,
  object_order,
  semantic_object_id,
}
```

B2IR 保存 dt 的 IEEE-754 位模式及 Brian 当前的 clock 选择/相等规则。Cluster v1 只支持单 clock 或经 validator 证明可安全同步的 commensurate clocks；更一般的 multi-clock 分布式放到后续 Gate。

`TimeKey` 是时间与执行位置的结构化描述，不按字段字典序比较。跨 clock 先使用 Brian 的时间比较/同刻判定，再按 schedule 的 `when/order/name` 全序选择对象；`clock_id` 不得改变同刻执行顺序，语义 ID 只用于寻址与保存已确定的 tie-break 结果。

Delay 必须在对应 pathway/clock 下显式量化并记录，禁止在不同后端各自重新舍入。

### 5.3 Effect Algebra

```text
IndexDomain:
  Scalar | Edge | SourceNeuron | TargetNeuron | Population | Global

AccessKind:
  Read | UniqueWrite | ReductionWrite | OrderedWrite | ExternalEffect

ReductionKind:
  Add | Multiply | Min | Max | Logical | DeclaredCustom

TimeVisibility:
  emission_time | delivery_time | phase_boundary
```

还必须表达：

- aliasing 和 linked variable；
- custom function 是否 pure、random、stateful；
- 同一 CodeObject 内 statement 与 RNG 调用顺序；
- reduction 是否允许重排；
- event/monitor 是否观察中间状态。

`Effect Region Splitting` 只能由编译器证明。若后续语句读取刚写入的 `v_post`，就不能简单把全部 edge-local 计算与 target reduction 分开。

`ReductionWrite(Add)` 只表示一种写入形态，不等于浮点加法满足结合律。IR 必须单独保存 `reassociation_allowed` 与 numeric profile 的约束；不能借“可归约”之名改变顺序敏感语义。

### 5.4 稳定 ID 与连接顺序

必须分开：

```text
SemanticNeuronId / SemanticEdgeId
PhysicalNeuronIndex / PhysicalEdgeIndex
```

Stable Edge ID 不能只用 `(pre_gid, post_gid, multiplicity)`，还需包含 synapse object、connect 调用/recipe ID 和 canonical enumeration ordinal，因为多次 `connect()` 和突触创建顺序对用户可见。需要连续 synaptic index 时，再以确定性 prefix scan 生成。

第一版只实现 immutable topology；为 structural plasticity 预留 generation/version 字段，但不实现动态增删。

Stable Edge ID 也不等于 event 的执行顺序。延迟、多个 pathway 和同一 edge 的重复事件需要独立的 `CanonicalEventOrderKey`：先遵守 delivery TimeKey 与 CodeObject/pathway schedule，再保持 reference queue 的事件出现/入队顺序及对应的 logical synapse 顺序。不得把全部 active events 简单按 edge ID 重新排序。第一版在初始化结束后冻结拓扑；多次 build 前 `connect()` 和运行段之间增删连接是两类不同能力。

### 5.5 容器格式

Gate 0 冻结逻辑 schema、canonicalization 和版本演进规则，不提前锁死所有物理格式。候选实现：

```text
model.b2ir/
├── manifest.pb             # schema / feature / hashes
├── expressions.pb          # typed semantic IR
└── arrays/                 # Arrow IPC 或自描述连续 buffer
```

大型数组与元数据分离计算 digest。B2IR validator 必须限制 shape、长度、offset、整数溢出、资源预算和扩展 capability，不能因为输入宣称一个超大长度就直接分配。

内置 IR 的 read/write/effect 应由 validator 从指令重算并与声明交叉检查，而不是信任输入文件自称“pure/parallel-safe”。外部 native function 的 effect contract 只能在明确的信任边界内接受，未知 effect 默认按串行或 unsupported 处理。

---

## 6. 编译、Artifact 与安全边界

### 6.1 三条执行路径

1. **Reference bytecode/interpreter**：单线程 oracle、fuzz、恢复验证；
2. **预模板 kernel**：常见固定算子、低编译延迟；
3. **模型专用 native codegen**：正式性能路径。

MVP 最终只保留一条主要 native codegen 路径，避免同时维护多个编译器后端。

### 6.2 Gate 0/2A codegen bake-off

至少比较：

- Cranelift native object；
- 生成 Rust 后由 rustc/LLVM 在构建机 AOT；
- 生成 C/C++ 后由 Clang/LLVM 在构建机 AOT；
- 少量预模板 SIMD kernel。

Bake-off 是限时的代表性 hot-loop 原型比较，不是先实现四套完整后端；Gate 0 淘汰明显不合适的路径，Gate 2A 前冻结一条 MVP native 路径。

评价维度：冷编译、增量编译、标量/超越函数性能、SIMD、object emission、调试/性能分析信息、x86-64 与 AArch64、动态加载和发布复杂度。Cranelift 支持嵌入式 JIT/AOT，且明确在编译速度与激进优化之间作取舍；因此适合作为候选，不能在测量前假定其最终 hot-loop 一定优于 LLVM。[Cranelift](https://cranelift.dev/)

TinyCC 只做有限 C 优化，且不能直接接收现有 C++ 模板，不进入默认架构；其官方手册也说明多类 GCC 优化选项并不起作用：[TinyCC documentation](https://bellard.org/tcc/tcc-doc.html)。

### 6.3 B2K Artifact

```text
model.b2k/
├── manifest.json
├── kernels/<target>/<isa>/...
└── checksums.json
```

Manifest 至少记录：engine/codegen/ABI、target triple、CPU features、numeric/math profile、Definition/KernelPlan hash、specialization inputs。通用分发包不能使用 `target-cpu=native`；可携带 baseline 与 AVX2/NEON 等变体，由 runner 在验证后选择。

### 6.4 构建机与目标机

- build-host AOT：构建机需要工具链，目标机只需 runner 和匹配的 B2K；
- embedded codegen：目标机不需要系统 compiler，但 runner 自身包含 codegen；
- B2K 不兼容时必须选择匹配的 baseline、显式重新构建或拒绝运行；不能静默执行另一种数值语义。

### 6.5 信任模型

- B2IR 是不可信输入：完整验证后才创建数组或执行；
- B2K 是 native code：默认仅在独立 runner 加载经验证、由受信任构建链生成的 artifact；hash 只证明完整性，不证明代码安全；
- PyO3 runtime 默认运行 reference/内置受信任 kernel。未来若启用可信自建 B2K 的 in-process 路径，必须显式选择，并说明与 Python 同一进程、缺少崩溃隔离；来源不明的 B2K 不得直接执行；
- 插件 ABI、签名策略、W^X、macOS code signing、Windows DEP 进入 Gate 0 threat model；
- public C plugin ABI 延期，避免在语义和内存所有权尚未稳定时冻结接口。

跨 FFI buffer 必须明确由哪一侧分配/释放；Python live-state 只在 safe point 保证可见，不能承诺任意 mutable NumPy view 与 Rust worker 同时零拷贝读写。

独立进程提供故障隔离，不自动构成权限沙箱；运行第三方不可信 native code 仍需要操作系统沙箱/容器策略，属于另一个明确的部署能力。

### 6.6 分发形态

正式 standalone bundle 建议为：

```text
simulation/
├── run_manifest.json
├── model.b2ir/
├── instance/
├── artifacts/model.b2k/
├── runner
├── results/
└── checkpoints/
```

普通 Python wheel 可用 Maturin/PyO3 打包 adapter、runtime 和本地 runner，但 native artifact 仍按 target triple、ABI 和 ISA 校验。Linux x86-64 作为首个 production 平台，随后 macOS arm64，再扩展 Windows/Linux ARM。MPI runner 使用站点 MPI 或集群镜像单独发布，不能让普通 `pip install` 强依赖 MPI、bindgen 或系统 C toolchain。

Rust MPI 绑定依赖可用的 MPI 实现和构建发现机制，部署前应校验站点 ABI、版本和线程等级，而非假设通用 wheel 能覆盖所有 HPC 环境。[rsmpi](https://github.com/rsmpi/rsmpi)

---

## 7. 单机执行核心

### 7.1 数据布局

- 热状态使用 typed SoA，例如 `Box<[f32]>`、`Box<[f64]>`、`Box<[i64]>`；
- 序列化索引使用明确的 `u32/u64`，禁止持久化 `usize`；
- Source CSR：`source → stable edge ranges`；
- 可选 Target CSC：summed variable、target-major reduction、post 操作；
- edge state、weight、delay 分离连续存储；
- 只有在 benchmark 证明有效时才引入替换 allocator。

CSR 的收益取决于 edge state、delay 分布、fan-out 和线程数，不预先承诺固定 70% 内存节省。

### 7.2 执行器

第一阶段采用贯穿 `run()` 的 persistent worker pool：

```text
phase barrier
  ├── independent object tasks
  ├── large-object tiles
  └── batched small objects
```

完整动态 Task-DAG、NUMA affinity 和 work stealing 后置。`reproducible` profile 固定 logical tile、任务顺序和归约树；只有 `fast` profile 才允许依据运行时统计切换策略。

Kernel fusion 同样由 Effect/Observability 决定。默认 schedule 中 synapses 位于 thresholds 与 resets 之间，不能把 state update、threshold、reset 无条件合为一个逐 neuron 循环。MVP 从同一 phase 内可证明无观察点、无跨对象依赖的融合开始；跨 phase 融合需要独立的等价性证明与差分测试。

### 7.3 初始突触策略集合

MVP 只要求三种：

1. `serial`：active edge 很少或操作顺序不可交换；
2. `edge_parallel`：只写 edge-local state，且每个 edge 在本阶段独占；若同一 edge 有多个事件，先按 canonical event order 串行处理该 edge 的事件；
3. `target_partitioned`：只有跨 target 无别名和读写依赖时，target 间并行；target 内按 canonical event order 执行，归约则采用 profile 允许的固定顺序/树。

后续按真实数据增加 atomic scatter、thread-local sparse accumulator、segmented reduction。不能给每个线程分配完整 `N_post × variables` 缓冲；图着色不作为默认方案。即使代码只写 post，recurrent network 的 pre/post alias 或后续读依赖也可能阻止 target 间并行，必须经 Effect Analyzer 证明。

### 7.4 Threshold 与 Summed Variable

Threshold 使用稳定 compaction：

```text
worker 扫描固定 neuron block
→ block-local count
→ prefix sum
→ 按 block 顺序写 eventspace
```

Summed Variable 优先采用 Target CSC，使不同 target 独立并行；target 内保持稳定 edge 顺序。浮点加法不是结合律，`reproducible` 不允许依线程数改变归约树。

### 7.5 Delay Queue

- homogeneous pathway delay：ring slot 保存 source event，到期后经 Source CSR 展开；
- heterogeneous delay：按 `(source, pathway, delay class, target tile)` 建 bundle；
- delay 高度离散、每 edge 几乎唯一时允许退回显式 edge event，不宣传固定压缩比；
- queue/page pool 必须配置容量和增长上限，超过预算时给出可诊断错误。

### 7.6 RNG 与数值 profile

默认新 RNG 使用 Philox/Threefry 类 counter-based 映射：

```text
RNG(
  global_seed,
  semantic_callsite_id,
  logical_entity_id,
  TimeKey,
  invocation_index,
  lane,
)
```

[Random123](https://random123.com/)为无共享可变线程状态的并行 RNG 提供成熟先例。面向用户的 profile 规划为：

| Profile | 含义 |
|---|---|
| `compat` | 限定平台、target、线程配置下尽量对齐旧序列 |
| `reproducible` | 默认新语义；逻辑 tile、事件序、归约树、counter RNG 固定 |
| `fast` | Beta 后启用，可使用 atomic、动态调度、fast-math |

逐位承诺必须限定 engine、数学库、ISA 和 numeric profile。不能承诺 CPU/GPU、AVX2/AVX-512/NEON 或不同 libm 之间天然逐位一致。

---

## 8. Monitor、OOM 与 Checkpoint

### 8.1 OOM 契约

Monitor 原始数据量约为：

```text
ceil(duration / record_dt)
× recorded_entities
× sum(dtype_bytes)
```

Rust 不会改变这个信息量。能够承诺的是 disk/bounded 模式下：

```text
PeakRSS ≤ model state
        + topology
        + event queues
        + kernel scratch
        + configured channel capacity
        + writer working set
        + bounded checkpoint staging
        + bounded safety margin
```

不能对 `storage="memory"` 承诺固定 RSS。`disk` 模式只保证常驻内存有界，磁盘占用仍随记录时长增长，必须纳入预估、配额和保留空间。

该内存保证以固定拓扑、受预算控制的事件/工作区和可用的系统内存为前提。初始化、拓扑构建、结果读取同样纳入预算，避免 Python cache、临时复制和 Rust state 多份共存。关键分配使用 checked size/fallible reserve；发事件前尽量预留容量，不能在半个 phase 已提交后假装可安全保存当前状态。无法到达一致性切点时应失败并保留上一有效 checkpoint，而不是提交部分状态。

这是一份可监测、可验收的运行时预算契约，不是“任何环境绝不 OOM”的保证。必须计入 allocator/库工作区和 OS 余量，校验进程或 cgroup 限额；外部内存争抢、操作系统强杀等情况由最近的有效 checkpoint 承担恢复边界。

### 8.2 存储模式与 API

| 模式 | 语义 | 兼容性 |
|---|---|---|
| `memory` | 全历史在 RAM | 默认，保持现有 ndarray 风格 |
| `disk` | 固定 chunk 持久化 | 显式启用，返回 lazy view |
| `window` | 只保留固定窗口/降采样 | 显式新语义 |

`.to_numpy()` 会物化完整数据；超过预算时应要求显式 `force=True`，不能用零拷贝宣传掩盖潜在 OOM。

### 8.3 有界流水线

```text
Simulation
  → immutable fixed-size chunk
  → bounded channel
  → dedicated writer
  → committed watermark
```

- 慢盘：默认 backpressure/block；
- writer 永久错误、超时或磁盘满：错误必须反向传播，不能无限阻塞；
- `drop` 仅显式允许，并记录丢失范围；
- `stop_with_checkpoint` 只有在独立 checkpoint 路径或预留紧急空间经过 preflight 后才可承诺；输出盘已经写满时不能假定仍能在同一位置保存快照。

格式建议而非热循环依赖：Spike/Event 使用 Arrow IPC 或分片 Parquet；密集 State 使用 chunked Zarr。Parquet writer 会缓冲 row group，仍需显式 flush 和预算；Zarr 适合 chunked N-D 数组。[Arrow Parquet writer](https://arrow.apache.org/rust/parquet/arrow/arrow_writer/struct.ArrowWriter.html)、[Zarr](https://zarr.dev/)

Row-group flush 不等于底层文件已经持久化，也不等于 Parquet footer 已写完。Committed watermark 只能指向已完成格式收尾、校验和持久化提交的 immutable part；本地文件需完成 footer/close 与相应 fsync，Zarr/对象存储需定义 chunk 与 metadata 的提交协议。不能把 writer 收到 chunk 或调用一次 `flush()` 当成 crash-safe commit。[ArrowWriter API](https://arrow.apache.org/rust/parquet/arrow/arrow_writer/struct.ArrowWriter.html)

Part 目录、时间索引与 metadata 也须分页持久化，checkpoint 引用分层 manifest/index root；不能把全部 chunk 路径和历史 metadata 长期累积在 RAM，否则仍会随运行时长增长。

### 8.4 Checkpoint

第一版仅在完整 schedule safe point 保存，不支持 kernel 中途快照。至少包括：

- clocks/ticks、RunProgram 命令/运行段游标与下一执行位置；
- neuron/synapse state；
- topology 或稳定引用；
- delay wheel 和未投递 event；
- RNG schema/key/counter；
- monitor committed watermark 与 writer shard index；
- Definition/Instance/LogicalPlan/KernelPlan/Placement/RunConfig hash；
- B2K ArtifactHash 或 reference engine build；
- numeric/math profile 和兼容元数据。

本地 POSIX 文件系统第一版采用：

```text
write immutable shards
→ fsync(shards)
→ write manifest.tmp
→ fsync(manifest.tmp)
→ verify checksums
→ atomic rename(manifest)
→ fsync(parent)
```

对象存储不假定 rename 原子性，应使用不可变对象和最后提交的小 manifest。

正常 checkpoint 在 safe point 暂停推进，排空并提交截至该切点的 Monitor 数据，再提交状态快照。Writer 已失败时，只有把尚未提交的有界 Monitor chunks 连同 watermark 一并保存到可靠 checkpoint 路径，才可承诺继续后不丢样本；否则回退至上一有效 checkpoint，不能只保存较新的神经状态和较旧的输出位置。

第一版在暂停期间逐块写状态，避免先复制一份完整模型再异步落盘；异步/COW 快照属于后续优化。保留策略和配额须覆盖 checkpoint 总空间，且不能删除最后一个有效 manifest 仍引用的 shard。

`rkyv/mmap` 只允许作为同 engine/build/ISA 的 hot restart cache；正式 checkpoint 使用自描述 schema。第一版可靠恢复限定为匹配的 engine build、schema、kernel/numeric profile 与兼容清单；不能只凭 major version 相同就接受。跨 minor/major 版本或硬件的恢复需要显式迁移和 conformance 认证，Beta 后逐步提供。

验收必须满足：

```text
continuous(T)
== checkpoint(T1) → restore → continue(T)
```

在约定的 deterministic/`reproducible` profile 下，比较范围包括 event order、spike tick、RNG、delay queue 和 monitor 逻辑记录。Checkpoint flush 可能改变 part 边界、压缩结果与 metadata，不要求物理文件逐字节相同；但逻辑样本必须无缺失、无重复、顺序一致。`fast` profile 不承诺中断前后逐位相同，只承诺状态完整性和声明的数值/统计契约。

### 8.5 当前 OOM 与新引擎建设分两条线

不能让当前内存问题等待完整 Rust 引擎交付。Gate 0 先定位实际 OOM 阶段；在用户确认观测需求后，可通过缩小 record 集合、降低采样频率、只保留计数或统计摘要控制数据量。这些会改变输出，必须显式记录，不能作为同语义 benchmark 的提速手段。

若必须保存完整轨迹，长期解法是有界流式输出，而非换语言或 mmap 全部热状态。Gate 1 后优先交付最小 disk sink 与背压，checkpoint 完整可靠性再由 Gate 3 实现。现有 standalone 多次调用 `run()` 不自动清空累计 Monitor 数据，不能把“分段运行”直接当作可靠的 OOM 修复。

---

## 9. 分布式：只在需要时启用，但从语义层预留

### 9.1 两种目标

| 模式 | 首要 KPI |
|---|---|
| `capacity` | 单机放不下的模型能够运行；每 rank 内存随局部图缩放 |
| `speed` | 对可扩展模型缩短 wall-clock，并报告总资源成本 |

Planner 提建议，用户显式决定是否申请集群资源。

### 9.2 Cluster v1 支持边界

首版只接受 validator 能证明满足以下条件的模型：

- immutable topology；
- 单 clock 或受限 commensurate clocks；
- 跨 tile 因果关系全部显式表示为 event/delay、phase-boundary reduction 或其他有序 Effect DAG 边；
- edge-local state 与 target-local write；
- 或经证明安全、且 numeric profile 明确允许其固定归约顺序/树的 write；
- 无任意外部副作用和未经声明的远程访问。

Brian 的 `on_pre/on_post` 可以读取或写入 `_pre/_post`，linked/summed/global state 也可能引入远程 effect。因此 target-owned 是默认物理计划，不是对全部 Brian 模型自动成立的语义。未来可以增加共置、ghost refresh 或 remote transaction；v1 若不能证明 target-local/reduction-safe，就明确拒绝 cluster mode。

### 9.3 LogicalPlan 与 PhysicalPlan

```text
Canonical B2IR
  → LogicalPlan: stable tiles + ownership + causal edges
  → PhysicalPlan: tile → rank / NUMA domain / worker
```

Checkpoint 保存 LogicalPlan hash、PhysicalPlan/PlacementHash、rank/domain mapping、通信 epoch/watermark，以及 subscriber graph hash 或其可重建依赖。该设计允许未来把相同 logical tiles 重新映射到不同 rank 数，但 v1 只承诺相同 rank 数和相同 PhysicalPlan 恢复；不同 rank 数恢复还需要重建 subscriber graph、迁移 delay queue 和重新分配 monitor ownership，属于后期能力。

Logical tile 也不能成为新的语义 schedule 单位。默认必须先完成某一全局 code-object/phase 的所有相关 tiles 和远程 effects，再进入下一有依赖的阶段；只有 Effect DAG 证明无依赖时，才能跨对象或跨阶段流水化。

### 9.4 默认 1D target-owned

每个 target tile 拥有：

- 本地 target neurons；
- 全部 incoming edges 及其 plasticity state；
- 本地 delay/event queue；
- 本地 monitor shard。

Source owner 维护压缩订阅目录，source event 对同一 destination rank/domain 通常只发送一次：

```text
SpikeEnvelope {
  TimeKey,
  source_gid,
  source_event_id / event_kind,
  optional_routing_key,
  declared_payload,
}
```

Target 收到后，通过 `(source_gid, source_event_id) → eligible local synapse/pathway ranges` 展开。只有采用显式 per-pathway routing 时才携带 `pathway_id`，此时同一 source event 可能向同一 domain 产生多个 envelope。只有 Effect IR 明确规定 emission-time payload 时才随事件捕获远程变量；delivery-time read 不能被错误地提前采样。

### 9.5 分布式构网

禁止 Rank 0 先物化全局 `i/j/w/delay` 再 scatter。构网必须提供两条明确路径。

**兼容路径：Canonical edge stream**

```text
connection operation
→ 按 canonical generation order 产生 creation key
→ 在 target shuffle 前确定 SemanticEdgeId / logical synaptic index
  （或先对 global creation key 做分布式全序/range prefix）
→ 携带 logical ID 与 (pre_gid, post_gid, multapse_ordinal, state...)
→ chunked/external shuffle 到 target owner
→ shard 内生成独立 PhysicalEdgeIndex
```

这一路径服务旧连接顺序和公开 `S.i/S.j/S.w` 语义，读取时按 logical synaptic index 重建用户顺序；物理 edge 数组不必全局连续。输入可以流式分片，不能先完整进入一个 Python NumPy 数组。

**扩展路径：Regenerable distributed recipe**

B2IR 保存 pure declarative recipe 或 pre-sharded explicit edges：

- fixed indegree：可按 target tile 独立采样；
- fixed probability：采用确定性 skip/geometric 或等价并行算法，不能朴素扫描完整笛卡尔积；
- fixed outdegree：可能需要 source-owned 生成后按 target 分发；
- arbitrary predicate：必须有成本估算和 unsupported/expensive 报告；
- explicit huge graph：分片读取、外部排序/分区，不先进入单个 Python NumPy 数组。

Recipe 路径只接受 partition-invariant counter RNG、由 Global ID 决定的结果和已定义的 edge ordering。新 counter 模式保证分区无关；legacy exact connection order 只在 Canonical edge stream 的限定兼容路径承诺。这两个目标不能被描述为同一种默认随机语义。

构网完成后还必须得到 source subscription directory，且 `N_outgoing` 等全局量需要由 recipe 推导或分布式归约，不能假定 target shard 本地天然可知。

若 recipe 读取 `x_pre/x_post` 等两端静态属性，应按依赖选择：由 Global ID/procedural rule 本地重建、读取 pre-sharded metadata、执行 chunked distributed join，或转 Canonical edge stream/拒绝。禁止为了构网方便在每个 rank 复制全部 source metadata。

### 9.6 零延迟与正延迟

Brian 默认允许 threshold 产生的 spike 在同一 tick 的 synapses phase 被观察。跨 rank 零延迟边因此需要在相应 phase 边界交换：

```text
State Update
→ Threshold
→ exchange zero-delay events
→ execute local independent synaptic work
→ complete remote-triggered effects/reductions
→ Reset
```

这一通信可与无依赖本地工作重叠，但不能删除因果屏障。分区器应对 zero-delay cut 赋高成本。

本地与远程事件需按 canonical event order 合并；MPI 到达顺序不是模拟执行顺序。对同一 target 或存在别名依赖的更新，不能先做完全部本地事件再追加远程事件。

正 lookahead 必须取所有跨域因果边的最小有效延迟，而不只是普通 synaptic delay。v1 使用全局/fixed lookahead；peer-specific horizon 后置。[NEST](https://nest-simulator.readthedocs.io/en/v3.4/nest_behavior/running_simulations.html)也利用最小突触延迟定义通信区间。

有效延迟按 B2IR 量化后的时间语义计算。窗口采用明确的半开区间及边界 phase 规则，例如 `[T, T+H)`；下一窗口开始前，先接收所有可能在该边界生效的事件，不能因 off-by-one 在交换前多执行一个 tick。

### 9.7 通信策略

| 结构 | 候选策略 |
|---|---|
| D0 正确性、小集群 | Allgatherv oracle |
| destination rank 稀疏 | Targeted Alltoallv / nonblocking P2P |
| 稳定局部邻接 | MPI neighborhood collective |
| destination 密度接近全部 rank | Dense collective |
| 稠密且 reduction-compatible | 后期 2D source×target edge plan |

Allgather 不能是唯一生产架构，但在目标密度接近 100% 的中小集群可能仍合理；策略应由 destination density、event rate、payload、batch 和实测网络特征选择。Arbor 的全局收集路径可作为 correctness precedent：[Arbor distributed communication](https://docs.arbor-sim.org/en/latest/dev/communication.html)。

MPI 初始化线程应成为唯一 communication agent，按实现实际提供的 `FUNNELED/SERIALIZED` 等级设计；不默认 `MPI_THREAD_MULTIPLE` 高效。MPI 依赖只进入 `brian2-rust-cluster` 集群包/镜像，不进入普通 wheel。

通信本身也必须服从有界内存契约：send/receive page pool 设置 hard high-water mark；超预算 burst 采用稳定、确定性的分块多轮 exchange；检查 count/displacement 与总长度整数溢出。Capacity 模式禁止每 rank 物化全局 spike stream。每 rank 估算必须包含：

```text
M_rank = local neuron state
       + owned edge state / topology index
       + delay/event queues
       + bounded send/receive staging
       + bounded monitor buffers
       + checkpoint staging
       + runtime metadata / margin
```

运行时记录 communication、event queue 和 writer 的 high-water marks。

### 9.8 分片输出与恢复

- SpikeMonitor 由 source owner 写 shard；
- StateMonitor 由 state owner 写 shard；
- 运行期不 gather 到 Rank 0；
- monitor part 保存 `monitor_id`、logical tile、record ordinal/semantic entity ID、tick range、event order key、part sequence、checksum 和 IR hash，读取时恢复 Brian 的 canonical record/event order；
- PopulationRate 等非因果全局统计可以保存 local partial 后确定性后处理；运行中可观察的全局统计需专用 reduction stage，否则 v1 不支持；
- checkpoint 在逻辑上按 tile 可寻址，但一个 immutable physical shard 可以打包多个 tiles，manifest 保存 `tile range → shard/offset`，避免每 tile 一文件造成 metadata storm；
- consistent cut 必须让所有 matched sends/receives 完成，receiver 已解析并把未来事件放入本地 delay queue，且所有通信 channel 达到同一 epoch watermark；随后才 flush monitor watermark；
- 每个 rank 写临时 shard 和 checksum，再通过 collective 确认；
- 只有所有 shard 成功时 coordinator 才提交 global manifest；
- 第一版 rank 失败导致整个 job 失败，从上次 checkpoint 重启。

未来 2D edge partition 只支持静态稠密图和经验证的 target reduction。浮点 Add 不能因数学形式可加就被视为结合律成立；若改变归约顺序，必须由显式 numeric profile 授权并固定、测试相应树，不进入早期默认路线。

---

## 10. 兼容性与双栈迁移

### 10.1 Capability Report

每次 build 前输出：

```text
SUPPORTED_NATIVE
SUPPORTED_REFERENCE_ONLY
SUPPORTED_WITH_EXPLICIT_SEMANTIC_CHANGE
UNSUPPORTED_USE_CPP
```

报告必须区分：Python API、模型语义、数值/RNG、result object、build/run lifecycle、custom function 和 cluster subset。

Capability 检查应在 `Network.before_run`/build 前 fail-fast。不能依赖 `CurrentDeviceProxy` 的缺失属性警告或把 `NotImplementedError` 当作测试 skip；Rust CI 必须启用 `fail_for_not_implemented=True`。

### 10.2 兼容矩阵

| Tier | 范围 |
|---|---|
| A：纵向核心 | LIF/CUBA/COBA、单 clock、固定拓扑、scalar delay、SpikeMonitor |
| B：性能核心 | common state updater、threshold、Poisson、heterogeneous delay、基础 STDP、summed variable、StateMonitor |
| C：Beta Core | multi-clock、multiple pathway、custom event、event-driven state、TimedArray、subgroup、linked variable、multiple `run()` |
| D：后续 | SpatialNeuron、Synapses→Synapses、高级 linked/summed、external native library、structural plasticity |

还必须覆盖 refractory、monitor sampling order、synapse creation/index order、`run_regularly`、store/restore、build/run 分离、random/stateful user function 和异常路径。

### 10.3 Fallback 粒度

- 优化失败：可在同一 Rust semantic engine 内选择 serial physical plan；
- B2IR 能表达但 native codegen 暂不支持：只有用户显式允许时使用 reference；
- B2IR 无法表达或模型含仅 C++ 实现：默认 fail early；只有显式允许且可重放建模过程时，才在新的 Device 上重建整个模型并选择 C++ backend；
- 禁止单个 CodeObject 静默混跑 C++/Rust，避免状态、RNG、调度和输出语义碎片化。

自定义函数处理：Portable Function IR → 正常支持；显式 Rust implementation → 受信任、按 effect contract 审核的扩展；只有 C++ implementation → whole-model C++ 或明确报错。使用 Rust 编写并不自动证明该函数 pure 或并行安全。

Whole-model fallback 不是在已创建的 Brian 对象上热切换 `set_device()`。需要可重放的 model factory/build program、固定输入与初始化随机约定，并从全新的对象与 Device 状态构建；无法重放时，报告原因并由用户重新建模。回退不能静默丢掉用户要求的 disk sink、checkpoint 或 cluster 能力；旧后端不支持这些要求时应拒绝回退，而不是成功运行一个不同契约的任务。

### 10.4 Monitor 兼容性

默认 `memory` 保持旧 API。用户显式选择 disk/window 后，返回 lazy result，并清楚标注 `.to_numpy()` 的物化成本。这是有意识的新语义，不能伪装成 100% 无感兼容。

---

## 11. 测试、Benchmark 与验收

### 11.1 四层 Oracle

```text
Python Runtime
↔ C++ Standalone
↔ Rust Reference
↔ Rust Native
```

验证分层：

- 离散语义：schedule、sample tick、event/pathway、stable ID 和确定性 event order 精确比较；
- 数值状态：按 dtype、积分器、量纲和时长设置 `rtol/atol`；
- 阈值敏感/混沌模型：报告 spike-time bin、轨迹发散和模型 invariant，不能只用统一绝对误差；
- 随机模型：compat 路径验证限定配置，新 counter 路径验证 1/8/32 worker 分区一致，fast 路径验证统计性质。

还要验证公开对象形状和顺序，而不只是文件内容。例如 StateMonitor 内部是 time-major，而现有公共 API 返回 `(recorded_indices, time)`；Spike/Event/Rate Monitor 对 eventspace 的稳定顺序也必须保持。

### 11.2 Benchmark Corpus

| Workload | 主要压力 |
|---|---|
| Large recurrent LIF | synaptic delivery |
| CUBA / high fan-in | target reduction/collision |
| COBA / HH | state math、vector math |
| STDP + Poisson | RNG、edge state、reduction |
| Sparse low firing | 调度开销 |
| Heterogeneous delay | queue 与内存搬运 |
| Gap junction / summed | Target CSC |
| Many small groups | CodeObject/phase overhead |
| Multi-clock | scheduler correctness |
| Monitor-heavy | RSS、I/O、backpressure |
| Zero-delay modular | cluster 同步 |
| Positive-delay regional | lookahead 与通信重叠 |

每次拆分报告 lowering、planning、native compilation、initialization、connectivity、simulation、I/O、checkpoint、communication 和 total wall time。固定机器、compiler、CPU governor、线程绑定、warm-up，并报告重复试验分布/置信区间。

### 11.3 性能 Gate

Gate 0 冻结最终 benchmark 后再确认数字。建议初始门槛：

- Gate 1：只要求正确性和独立运行，不设提速 KPI；
- Gate 2A：单线程核心 workload 不低于同 numeric profile 的 C++ 90%；至少两个已知串行 phase 明显扩展；
- Gate 2B 继续投资门槛：可并行 CPU 套件几何平均 ≥1.25×；
- 产品目标：标准套件约 1.5×；特定 synapse/summed-heavy workload 2× 以上为 stretch；
- 任何关键 workload 回退 >10% 必须解释或选择同语义的 serial Rust plan；
- 编译冷启动、warm execution 和 end-to-end 分开报告。

现有 C++ 默认可能使用 `-O3 -ffast-math -march=native`，所以必须同时做“现有默认配置”和“同 numeric profile”两类比较，禁止通过改变精度、关闭 Monitor 或排除初始化虚标收益。

### 11.4 内存与可靠性 Gate

- disk/window 模式 Peak RSS 不随模拟物理时长线性增长；
- 测试 writer stall、磁盘满、权限错误、配额、单 shard/manifest 损坏；
- 强杀后只恢复到最后 committed manifest；
- deterministic/`reproducible` 的 checkpoint-resume 与 continuous run 逻辑结果相同；`fast` 按声明的数值/统计契约验收，不要求文件字节或轨迹逐位相同；
- 记录 queue/channel 高水位、bytes/neuron、bytes/edge 和输出吞吐。

### 11.5 分布式 Gate

共同正确性门槛：

- 1-rank ClusterPlan 与单机 Rust reference 对齐；
- `reproducible` 下 2/4/8 rank 及不同 tile mapping 保持约定的事件、RNG 和归约结果；
- zero-delay phase trace 与 schedule oracle 对齐；
- positive-lookahead 结果与逐 phase 同步 oracle 对齐；
- unsupported remote effect 在 build 前 fail closed；
- shard merge 后恢复 StateMonitor record order 和 Spike/EventMonitor canonical event order；
- Distributed D1 以后在相同 PhysicalPlan 和约定 deterministic/`reproducible` profile 下，checkpoint-resume 与 continuous run 逻辑结果相同。

Capacity 门槛：每 rank 不保存全局 topology 或全局 spike stream，Rank 0 不构造全局边，monitor/checkpoint 不 gather；construction、communication、delay queue、monitor 和 checkpoint staging 的 peak RSS 均有硬上限；weak-scaling 下每 rank 内存近似局部化。

Speed 基线必须是同 numeric profile 下的最佳单节点 Rust，而不是单核：

```text
1 → 2 → 4 → 8 → 16 nodes
S(P) = T_best_1_node / T(P)
E(P) = S(P) / P
```

同时报告 construction、steady simulation、I/O、checkpoint、communication、total wall time、总 CPU/节点小时、峰值内存与边际收益。不对不适合的模型宣传加节点必然更快。

### 11.6 安全与鲁棒性

Property/fuzz 覆盖非法 shape/offset、循环依赖、dtype 错配、越界 ID、整数溢出、恶意 bundle 和不匹配 Artifact。Native loader、checkpoint parser、B2IR validator 必须在 CI 中接受 sanitizer/Miri（适用部分）与 fuzz 测试。

---

## 12. Gate-based Roadmap

以下时间按 4 名核心工程师估算，属于规划范围而非交付承诺。

### Gate 0：边界与风险验证，4–6 周

交付：产品边界 ADR、Compatibility Matrix v0、Definition/Instance/Plan/Manifest 分层、B2IR v0、schedule/effect/time/RNG/hash ADR、benchmark corpus、codegen bake-off、内存估算器、许可与 artifact threat model。

Go 条件：LIF/CUBA/COBA 可完整表达；任意方程如何执行没有架构空洞；codegen 路径有测量结果；未决语义都有 owner 和后续 Gate。

### Gate 1：Reference Standalone，8–12 周

交付：Python lowering、validator、单线程 reference executor、独立 runner、Tier A、Stable ID、Source CSR v0、memory SpikeMonitor 与 bounded sink 接口、C++ differential harness。

Go 条件：脱离 Python 重放；spike tick/event order 对齐；状态在约定容差内；非法 IR fail closed。

Gate 1 后由存储负责人优先完成最小 disk sink、固定 chunk、背压和错误传播，可与 Gate 2A 并行；不必等 checkpoint 全部完成才验证 Monitor RSS 是否有界。

### Distributed D0：语义探针，Gate 1 后 3–5 周

仅使用 deterministic、fixed-topology、小型模型，做 2–4 rank、target-owned、由 schedule/effect DAG 插入的 zero-delay causal phase-boundary sync、Allgather oracle、sharded result 和 checkpoint schema/ownership 骨架，并与单机 reference 逐 phase 对齐。这里不承诺真实 checkpoint 恢复，也不依赖尚未交付的 counter RNG/复杂归约；目的只是验证 B2IR、时序和 state ownership 没有堵死分布式，不承担生产性能。

### Gate 2A：Native 与低风险并行，8–12 周

交付：选定的一条 native codegen、B2K、persistent pool、threshold compaction、edge-local synapse、static planner v0、phase profiler。

Go 条件：artifact 可缓存/验证；多线程 conformance 通过；已知热点出现明确可重复收益。

### Gate 2B：归约与核心性能，10–14 周

交付：受限 Effect Region Splitting、target-partitioned reduction、CSR + optional CSC、counter RNG、heterogeneous delay、第二种确定性 reduction、reproducible profile。

这是决定是否把 Rust 定位为通用高性能替代的主要 Gate。

### Gate 3：有界输出与 crash-safe checkpoint，6–10 周

Sink 接口从 Gate 1 存在，并在随后尽早验证 disk sink；本阶段完成存储格式/lazy reader 稳定化、quota/reserved checkpoint space、匹配 engine build/schema/kernel/profile 的 checkpoint 和 failure injection。可与 Gate 2 局部并行；跨版本恢复不是本 Gate 的默认承诺。

### Gate 4：Beta Core 与双栈迁移，12–16 周

扩大到 Tier B/C 中经优先级确认的功能；完成 multiple run、常见 user function、跨平台 packaging、migration guide、opt-in community beta。C++ 仍保留。

### Gate 5：生产稳定化，8–12 周

长时间 soak、Linux x86-64 production、macOS arm64、Windows/ARM 评估、性能回归 dashboard、B2IR v1 freeze、checkpoint migration policy。

### Distributed D1/D2：真实需求触发

- 前置条件：Gate 2B 的 Stable ID、counter RNG、delay/reduction 语义通过，Gate 3 的 bounded sink 与 checkpoint 通过，并存在真实超大模型、目标硬件和合作用户；
- D1 Capacity Alpha（12–16 周）：分布式构网、local-only topology、source subscription、bounded targeted Alltoallv/P2P、same-PhysicalPlan restart、capacity weak scaling；
- D1 Go 条件：无全局 topology/spike 副本，内存落在估算误差内，burst 不导致失控 OOM，约定 deterministic/`reproducible` profile 下恢复与连续运行逻辑一致；
- D2 Speed Alpha（16–24 周）：nonblocking P2P、node aggregation、overlap、neighborhood collective、load-aware mapping、positive-delay fixed lookahead；
- D2 Go 条件：在预先冻结的模型/集群上，相对最佳单节点 Rust 达到约定 strong-scaling 门槛，并完整报告通信和负载不均；
- D1/D2 始终受 9.2 的 capability boundary 约束，不代表任意 STDP、ghost、remote write 或 multi-clock 已支持；任何跨域零 lookahead 因果边都会令相关区域退化为 phase-boundary sync；
- 2D partition、peer-specific horizon、dynamic remapping、千核优化另行立项。

粗略日历时间：Gate 0–1 可验证 PoC 约 3–5 个月；单机性能 Alpha 约 7–11 个月；可用 Beta 约 12–18 个月；完成约定平台与兼容矩阵的 Production 约 15–24 个月。Cluster Capacity Alpha 在真实需求出现后追加约 6–9 个月。D0/D1 与单机工作共享四人团队时必须重排关键路径，不能把并行工作都按免费资源计算。

---

## 13. 仓库、团队与上游策略

### 13.1 建议仓库结构

```text
brian2-rust/
├── python/brian2_rust/
│   ├── device.py
│   ├── lowering/
│   ├── compatibility/
│   └── api/
├── crates/
│   ├── b2ir-schema
│   ├── b2ir-validate
│   ├── b2-reference
│   ├── b2-effects
│   ├── b2-plan
│   ├── b2-codegen
│   ├── b2-runtime
│   ├── b2-topology
│   ├── b2-events
│   ├── b2-rng
│   ├── b2-monitor
│   ├── b2-checkpoint
│   ├── b2-runner
│   └── b2-cluster
├── conformance/
├── benchmarks/
├── fuzz/
├── adr/
└── docs/
```

初期建议独立仓库孵化，通过薄的 Brian2 Device adapter 集成；B2IR schema、conformance 和 benchmark 作为一等交付物。`b2-cluster` 与 MPI 不进入普通 wheel。若后续进入 Brian2 主仓库，也应新增同级 `RustStandaloneDevice(Device)`，而不是继承 `CPPStandaloneDevice`。

上图表达长期模块边界，不要求 MVP 立即拆成十余个 crate。初期可收敛为 4–6 个 crate，将 topology/events/RNG/effects 等放在内部模块；只在依赖、测试、发布或编译隔离确有收益时再拆分。

### 13.2 团队

| 角色 | 建议投入 |
|---|---:|
| Compiler/HPC Tech Lead | 1 |
| Brian2/Python/compatibility Engineer | 1 |
| Rust runtime/performance Engineer | 1 |
| Storage/reliability Engineer，后续转 distributed | 1 |
| Scientific validation advisor | 0.25–0.5 FTE |
| CI/HPC support | 0.2–0.3 FTE |

三人可以启动 Gate 0–1，但会明显拉长并行、存储和跨平台工作。

### 13.3 许可与上游

Brian2 使用 CeCILL 2.1，见仓库 [`LICENSE`](LICENSE#L8)。Python adapter、同进程 Rust library、独立 runner 和代码复用的法律边界不能只靠“进程隔离”推断，Gate 0 需要正式许可审查。许可结论前避免批量复制 C++ template。

尽早与 Brian2 maintainers 对齐 Device API、测试复用、命名、兼容期望，以及哪些通用改进应直接回馈 C++ backend。

---

## 14. 风险、降级与停止条件

| 风险 | 降级/处置 |
|---|---|
| B2IR 范围失控 | 冻结 Tier A；其余 capability-gated |
| Native codegen 性能不达标 | 保留接口，切换 AOT/模板方案；reference 不承担性能 |
| Effect 无法安全拆分 | Rust serial plan 或 whole-model C++ |
| Numeric compatibility 阻塞并行 | compat/reproducible/fast 分级，不伪称逐位通用 |
| Monitor writer 成为瓶颈 | 有界队列、bulk copy、分片、背压与明确失败 |
| checkpoint 格式不可迁移 | 第一版按 engine build/schema/kernel/profile 兼容清单恢复；跨版本迁移另行认证 |
| MPI scope 拖慢主线 | cluster 独立包；D1 必须由真实超大模型触发 |
| Python/Rank 0 构网先 OOM | declarative recipe、sharded ingestion |
| Native Artifact 安全风险 | 独立进程、hash/provenance、fail-closed validator |
| 上游不接受 | 外部 Device 双栈长期共存 |

Gate 2B 后，如果可并行套件几何平均低于约 1.15×，且没有主要模型类别达到 1.5×，停止扩张“通用高性能替代”叙事，转为以下一项或多项：

- 专注 bounded Monitor/checkpoint；
- 把关键并行优化回馈现有 C++ backend；
- 保留少数模型的 Rust 专用路径；
- 停止继续扩大兼容面。

介于停止线和 1.25× 继续投资门槛之间，进入限时优化/缩小支持面的复审区，不自动通过 Gate，也不无限延期。最终阈值仍以 Gate 0 冻结的真实模型集与同语义基线为准。

Gate D1 只有在存在真实模型、明确硬件、可量化的单机容量缺口和合作用户后才批准。

---

## 15. 最终项目定义

> **Brian2-Rust 是一个由版本化 Canonical B2IR 驱动的可替换科学计算执行平台。它保留 Brian2 Python 建模前端，优先交付独立 standalone runner，并为共享语义核心的交互 runtime 预留入口；首先解决单机多核利用率、有界 Monitor、可恢复执行和确定性并行。大量独立实验通过 Ensemble 编排扩展；只有单机容量不足或经分析具有明确强扩展价值的单模型，才显式启用独立的 cluster backend。Rust 后端在兼容性和性能 Gate 通过前保持 opt-in，现有 C++ standalone 继续作为显式重建整个模型后的备选路径。**

立项授权建议：

```text
立即启动：Gate 0
通过语义与风险评审后：Gate 1
条件批准：Gate 2A
数据决策：Gate 2B
可靠性主线：Gate 3
兼容性扩张：Gate 4–5
分布式生产：真实超大模型触发 D1/D2
```

---

## 16. 主要调研依据

- Brian2 standalone/runtime 边界与调度语义：[Brian2 eLife paper](https://elifesciences.org/articles/47314)、[Brian2 documentation](https://brian2.readthedocs.io/en/stable/advanced/preferences.html)
- 快速编译候选：[Cranelift](https://cranelift.dev/)
- Python/Rust wheel 分发：[Maturin](https://github.com/PyO3/maturin)、[rust-numpy](https://pyo3.github.io/rust-numpy/numpy/)
- 并行 counter RNG：[Random123](https://random123.com/)
- 分布式最小延迟与线程/MPI 模型：[NEST simulation timing](https://nest-simulator.readthedocs.io/en/v3.4/nest_behavior/running_simulations.html)、[NEST parallel computing](https://nest-simulator.readthedocs.io/en/v3.6/hpc/parallel_computing.html)
- 模型、资源与 domain decomposition 分离：[Arbor concepts](https://docs.arbor-sim.org/en/stable/concepts/)
- 计算引擎从前端抽离的先例：[CoreNEURON paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6763692/)
- Rust MPI 工程边界：[rsmpi](https://github.com/rsmpi/rsmpi)
- Arrow/Parquet 与 Zarr：[Apache Arrow Rust Parquet](https://arrow.apache.org/rust/parquet/)、[Zarr](https://zarr.dev/)
