# ExecutionPlan 研究与实施建议

日期：2026-09-06。以下为实施前研究快照；后续已启动实现，当前 API、验证与范围见 [ExecutionPlan v0](EXECUTION_PLAN.md)。下文“未实现/待复现”描述仅对应审计基线，不代表分支现状。
审计基线：`b87ee8d1`；分支：`codex/execution-plan-research`。
本分支从已提交版本创建，不包含原工作区正在进行的 Litwin-Kumar 修改。

## 1. 建议结论

值得引入显式 ExecutionPlan，但第一目标应是**现有 CPU 执行决策可检查、可解释、可独立测试**，随后解决 AOT slot-driven schedule，再基于实测改进 NUMA。GPU/MPI 是扩展方向，不应成为第一版抽象的交付条件。

采用两级概念：LogicalPlan 保存语义顺序、完整 effects、候选变换与证明条件；CpuPhysicalPlan 选择 kernel、布局、分片、队列和同步策略。ExecutionPlan 是这些信息及来源身份的容器，不是新的用户模型 IR。

第一版建议在 Python 抽取现有 planner。当前 AOT 实际由 Python 生成 Rust；直接先建 Rust Plan AST 会同时引入跨语言协议和迁移成本。Rust 继续独立验证 B2IR；待 Python plan 能驱动现有 emitter、结构稳定后，再决定是否把 canonical planner 迁到 Rust。不要为了未来后端现在冻结 Kernel ABI。

本次交付仅为源码审计、设计和可拆分开发任务；没有修改运行代码、声称性能提升或启动三机实验。

## 2. 已有机制与抽取位置

以下位置均相对 `brian2-rust/`，行号对应上述基线；函数名是后续重构的稳定检索入口。

| 职责 | 现有证据 | 建议归属 |
| --- | --- | --- |
| canonical schedule 与 linked resource 解析 | `python/brian2_rust/schedule.py:146` `build_schedule`；`:57` `_population_effects` | 保留 frontend wire 构建；planner 读取已验证结果 |
| 补齐隐式 refractory effects | `schedule.py:30` `execution_effects`；`src/main.rs:2264` `complete_execution_effects` | LogicalPlan 的 completed graph；Python/Rust 独立一致性校验 |
| fixed-phase 重排合法性 | `native.py:1851` `fixed_schedule_is_semantically_equivalent` | 先原样抽取；后由 slot-driven lowering 替代固定 rank 限制 |
| 单组/跨组融合合法性 | `native.py:1922` `_schedule_can_contract`、`:1935` `_schedule_can_fuse_bundles` | LogicalPlan 合法性事实；实际是否融合由 CPU cost policy 选择 |
| workload 启发式与 target-owner eligibility | `native.py:397` `code_work`、`:424` `parallel_task_limit`、`:471` `target_parallel_plan` | 分开 safety 与 cost；保留已有 `TargetParallelPlan` 判定结果 |
| route 合并、delay/pending 条件 | `native.py:2126` `generate_source` 前段 | CPU route planning，保留原始遍历顺序 |
| population/threshold dispatch 与 summed 分组 | `native.py:2241`、`:1773` `_v7_fused_summed_groups` | CPU kernel/dispatch planning |
| final-only summed 判定 | `native.py:1983` `_v7_summed_final_only` | 先兼容性抽取；单独审计 observable resources 后再推广 |
| 紧凑旧路径选择 | `native.py:2081` `_uses_v6`，`:1295` `_single_population_v6` | PhysicalPlan 明确记录 emitter variant；不能只迁移通用路径 |
| Linux CPU/NUMA 枚举和 pinning | `native.py:4241` `cpu_numa_node`、`:4243` `automatic_cpu_order`、`Parallel::from_env` | runtime binding；这不是从零新增的能力 |
| degree-balanced ownership | `native.py:4359` `degree_balanced_target_owners` | 物理布局策略；实际 owner map 在实例/worker 数确定后绑定 |
| artifact 身份及重用 | `native.py:4165` `write_project`；`artifact.py` | plan sidecar 与 artifact 联结，保留现有校验 |

`device.py` 的标准 AOT build 会先调用 Rust `--validate` 再 `write_project`。新的 explain 入口也要明确验证前置条件，不能让直接 Python helper 的输入字典被误认为已验证模型。

阶段报告中的 24 个门是统一性能套件的场景门。本次检查 `.github` 未找到该 Rust 套件的接入，不能称为“已有 24 个 CI 性能门”。报告也明确记录 EPYC 23/24、M3 噪声及版本不完全一致；后续回归要建立同一 clean revision 的基线。

## 3. 对原提案的关键修正

### 3.1 B2IR 已经包含 when 和顺序

B2IR 并非只有 What。`definition.schedule` 和 `run.clocks` 已固定可观察的时间及顺序。Plan 可以选择等价实现，不能重新定义何时发生事件、monitor 读到哪个时刻或 reset 的可见性。完整契约见 [B2IR §5](B2IR.md#5-schedule-and-effect-algebra)。

### 3.2 融合不是要求两个节点没有 RAW

update 写 `v`，threshold 读新 `v`，这个 RAW 正是要保留的依赖。判断包括：相同激活时间集合、组内依赖顺序不变、被跨越节点不受影响、逐元素交错不引入跨 neuron 的读写问题、scalar/old-state snapshot 语义保持，以及 RNG/Function 调用次数与顺序约束满足。

例如 canonical `update(A) → monitor(v) → threshold(A)`，不能把 threshold 越过会观察其隐式写入的节点；是否允许越过某个 monitor 要看完整资源 effects，不能仅按 slot 名判断。

更直接的反例是 `threshold → on_pre（读取 v_pre）→ reset`：提前 reset 会改变突触读值。因此不默认把 update、threshold、reset 合成一个 loop。

融合合法和融合划算是两件事。把脉冲条件/写队列放入向量更新可能影响向量化；现有生成器也有 work 门槛。不能承诺“访存必降 50%”；应测 memory traffic、kernel 时间和端到端耗时。

### 3.3 静态内存预算不等于 Peak RSS

LogicalPlan 可以给出资源、alias 和生命周期约束；物理重用需要已选择的 schedule、并行 overlap、dtype、alignment、地址空间和所有权。放在纯 LogicalPlan 里确定实际 arena offset 过早。

事件队列随 spike 活动及 delay 增长，monitor 可能跨 run 延续；拓扑构建 scratch、allocator capacity、线程栈、mmap resident pages、Python/编译子进程也不等于模型数组大小。内存报告应分 `exact_payload_bytes`、`bounded_bytes`（附上界条件）、`estimated_bytes`、`unknown_dynamic`，并区分 native child 与整个工作流。v0 只做资源清单，不做 arena reuse。

### 3.4 final-only 是时序敏感的 dead-store 优化

没有中间 observer 只是必要条件。还要覆盖 linked variables、所有 monitor/event/Function effects、后续 run 与 checkpoint 可见状态；最后一次求值必须仍在原来的最后一个 active tick/slot。若 summed 位于 tick 开头，输入之后还会更新，把它简单搬到 run 结束会算到不同值。

现有 `_v7_summed_final_only` 检查本 population 的 monitor/CodeObject 和突触 pre/post alias，没有通过 completed resource graph 扫描其他 population 的 linked reader。这是静态审计发现的**待复现风险**，本次未证明某个完整有效模型已有错误，也未修复。P0 应构造有效 B2IR、reference/AOT 差分用例确认；确认缺陷则单独修复，不能混入“字节不变”的抽取提交。

### 3.5 GPU 与 CPU 数值/资源能力不能统一假定

浮点原子或树状 reduction 会改变加法顺序，不能自动满足 `reference-f64` 下现有结果契约。先定义可支持的确定性 reduction；不支持时 fail closed，不在 B2IR v1 里偷偷启用 relaxed 数值语义。目标设备 f64、原子和同步能力需单独验证，不能从存在 `metal/cuda/wgsl` Function descriptor 推导已有执行支持。

GPU 首个可运行子集应从多 kernel 的明确依赖调度开始；persistent kernel 是后续候选，需要全局同步、占用率和运行限制等证据。command buffer/graph 重放同样值得实验，不预设唯一方案。OpenXLA 的官方管道将 scheduling、buffer assignment、thunk emission 和 command buffer 转换分开，可作为边界设计参考。[OpenXLA](https://openxla.org/xla/hlo_to_thunks)

MPI 的延迟/lookahead、跨 rank 同刻事件顺序、通信完成依赖及恢复状态仍需专项设计，不能仅把 owner 改名为 rank 就认为语义已解决。

### 3.6 架构动机成立，收益和新颖性尚待证据

EPYC 对 NEST 的 6.44 倍差距是阶段报告引用的旧结果，不是本研究重测，也不能单独证明 NUMA 是主因。必须拆测 compute、event delivery、barrier、queue 和内存访问，逐项消融。

连接概率 `p` 不能单独决定 source/target 最优策略；还需 firing rate、degree skew、delay、可塑性成本、可用内存和 worker 数。Autotuning 必须限定在已证明合法的候选内，采样需在隔离状态上执行或完整恢复，保持 RNG、queue、monitor 和外部 effects。

不采用“其他模拟器几乎都没有现代编译架构”的论文结论。GeNN 本身是代码生成系统；仅有三层图不足以证明独创性。当前只查阅其官方项目介绍，没有完成 NEST/GeNN/Brian2GeNN 的体系化比较。[GeNN 官方仓库](https://github.com/genn-team/genn)

## 4. 建议的数据与执行边界

```text
B2IR bytes → integrity + semantic validation
                     │
                     ├→ reference executor（独立正确性基准）
                     ↓
             completed execution effects
                     ↓
       LogicalPlan（canonical nodes + legality facts）
                     ↓ + topology facts / backend capabilities / policy
       CpuPhysicalPlanTemplate（kernels / routes / buffers / dispatch）
                     ↓
       existing Rust AOT emitter + instance writer
                     ↓ + actual worker count / cpuset / instance / run
       RuntimeBinding（owners / queues / affinity / selected paths）
                     ↓
                 execution + measured profile
```

RuntimeBinding 是 physical plan 的具象化阶段，不增加第三套语义 IR。当前 `B2_NUM_THREADS`、`B2_THREAD_AFFINITY` 是运行期输入，任务数也依赖工作量；编译期 explain 不能伪称已知实际 8 个 worker 或真实 NUMA placement。

### 4.1 v0 字段草案

以下是待实现的数据契约，不是已存在的 API 或冻结 JSON schema。

| 对象 | 最小字段/约束 |
| --- | --- |
| ExecutionPlan | `schema=b2-execution-plan-v0`、planner revision、B2IR layer hashes、policy revision、backend capability fingerprint、logical、physical、diagnostics |
| LogicalNode | 原 schedule node ID、owner/item、canonical ordinal、clock/activation、operation ref、completed reads/writes、dependency IDs |
| LegalityFact | rule ID/version、选中节点、保留的内部顺序、被跨越节点、资源/alias/clock 前置条件、accepted/rejected 与原因；是可复查证明记录，不声称机械化定理证明 |
| KernelGroup | ordered node bundles、iteration domain、scalar/vector snapshot contract、合法候选引用；CPU policy 决定最终 grouping |
| CpuPhysicalPlanTemplate | emitter variant（compact v6/general）、ordered dispatch、kernel refs、work estimates/task rules、routes、buffers、queue/sync descriptors |
| EventRoute | pathway/stream、source/target domain、delay policy、pending compatibility、serial/target-owned strategy、稳定事件及 edge ordering contract |
| Buffer | resource ID、dtype/shape、storage role、logical lifetime、alias set、persistent/scratch、address space、size certainty、allocation owner |
| RuntimeBinding | requested/effective workers、实际 CPU/NUMA map 与 pin 成功状态、owner map/topology fingerprint、queue capacities、动态选择的策略及原因 |

第一版复用 B2IR operation/statement 引用，不复制第二套表达式 AST。LogicalPlan 保留时间模板，不能把长 run 展开成每 tick 一个完整 DAG。Monitor、event_source、threshold/reset、summed、refractory 和 queue 跨 tick 状态均须有表达位置；无论是否使用显式边，都需保留其边界依赖。

Node IDs 和序列化顺序确定；explain 的耗时/主机探测等非确定字段与稳定 plan hash 分开。大图 owner/CSR 数据用内容哈希引用，避免为了打印计划复制亿级 edges。验证不能新增一次逐 tick 图遍历或额外全边 materialization。

### 4.2 验证、版本与缓存

1. B2IR 完整性和语义验证先于 planner；implicit effects 在验证 wire hash 后补齐，保持 B2IR bytes/hashes 不变。
2. PlanVerifier 检查节点覆盖/唯一性、引用与依赖、激活时刻、effects 完备性、跨越合法性、buffer/ownership 不重叠约束。早期只消费内存中派生 plan，不开放任意 JSON plan 执行；日后读入 plan 必须重新验证。
3. 切换到 plan-driven emitter 前要求它与旧判定一致；差异必须 fail closed 或显式测试失败，不能默默继续输出不一致的 EXPLAIN。
4. 初期按 definition/instance/run、planner/policy version、emitter/toolchain flags、native Function source、目标能力保守失效。以后才用显式 specialization dependency 缩小缓存 key；尤其 topology、delay、pending 和 run clocks 已影响现有选择，不能只用 Definition hash。
5. v0 sidecar 独立版本；第一阶段不更改现有 manifest schema。正式消费/重用 plan 时需另立 manifest 版本与迁移，继续校验已有 generated-source、Definition/Run 和 C translation-unit 身份。
6. runtime env 改变需重新绑定并验证；不能复用旧 worker 分片指针。绑定失败保持既有错误/回退契约，并在实际运行报告说明。

Effects 不等于所有变换的充分证明：访问之外还有 alias、异常、Function effects 和语义约束。MLIR 的官方讨论同样区分 memory effects、未定义行为、终止及非局部控制流；本项目应借鉴保守分析边界，不直接把其 CFG 规则当作 B2IR 图的证明。[MLIR Side Effects & Speculation](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)

### 4.3 EXPLAIN 的真实含义

建议输出 compile plan（候选/拒绝理由）与 bound plan（实际 worker/owner/affinity）两份。API 名可为 `device.explain_plan()`，但应在导出/验证后可用；调用本身不启动 simulation，不能伪造 profile 数字。

文本示意，非本次工具实测：

```text
CPU plan [unbound; v0]
  StateUpdate → Threshold: legal ordered bundle
    decision: policy-dependent; worker binding pending
  Reset: canonical slot retained
  Event route: target-owned eligible
    safety: no recurrent pre-read/post-write alias
    tasks: selected at runtime from workload and worker limit
  Memory: static payload known; dynamic queue capacity unresolved
```

每项优化都要能回答“为什么采用/不采用”，并区分 unsafe、unsupported、合法但工作量小和硬件信息未知。初期 emitter 仍可执行类型 lowering 和生成器断言；“纯打印机”是减少隐藏调度决策的方向，不是删除所有检查。

## 5. 可独立审阅的开发顺序

| 阶段 | 交付 | 完成门 | 不混入的工作 |
| --- | --- | --- | --- |
| P0 审计与基线 | 有效模型 corpus、现有 decision/source hashes、linked summed 风险复现、标准 build validation 边界检查 | compact/general、合法/拒绝 schedule、implicit effects 均有证据；若发现 bug 先单独处理并重建基线 | 新优化、性能承诺 |
| P1 观测性原型 | `plan.py` 数据对象、completed logical graph、现有 CPU decision 快照、JSON/text explain、sidecar | 不修改 B2IR；旧 source bytes 不变；每项报告与真实旧判定一致；未覆盖决策明确标记 | 改调度、arena、GPU |
| P2 决策抽取 | planner 计算一次，compact/general emitter 与实例布局消费同一 plan；PlanVerifier | 同 corpus 的 Rust/C 源码及实例布局不变；reference/AOT、1/N worker 语义门通过 | 抽取时顺带修改启发式 |
| P3 slot-driven CPU | 按 canonical slot/clock 降低，保留经证明的融合 | 新增原来明确拒绝的合法 when/order 用例通过；既有支持面无退化；此阶段不要求 source 字节不变 | GPU/MPI |
| P4 NUMA 实验 | 实际 placement/queue/同步 profile，逐项 owner-local allocation 与同步候选 | 同模型/同机/同 revision 配对消融；严格数值及 lifecycle 门通过 | 未测量即宣称缩小 6.44x |
| P5 后端扩展 | 选择一个 GPU 和明确的最小受支持模型集合、能力矩阵及数值门 | f64/事件/队列/同步的可行性先通过；unsupported fail closed | 同时交付 Metal/CUDA/MPI |

P1 可先按如下模块草案落地：`python/brian2_rust/plan.py`（不可变数据）、`planner.py`（纯决策）、`plan_explain.py`（渲染）、`tests/test_execution_plan.py`（语义与一致性测试）。这不是要求四个空文件先行，而是避免继续把 planner、JSON 和代码模板都加到 `native.py`。

P1 的快照必须来自 emitter 真正使用的决定，或自动与其比较；单独复制一套“看起来一致”的 planner 不构成交付。P2 完成后删去重复权威实现，保留基线 fixture/差分证据。若 Python/Rust plan 边界届时确实需要，另开迁移任务；不维护两个会自行演化的优化器。

## 6. 验证与实验矩阵

已有测试入口包括 `test_b2ir_v1.py`（hash/implicit effects）、`test_native.py`（融合和 custom schedule）、`test_refractory.py`、`test_population.py`（linked/Function/跨 population dispatch）、`test_synapses.py`（target-owned/塑性/summed/delay）、`test_device.py`（segmented run/checkpoint）和 `test_artifact.py`（Function drift）。这些是后续实施门，本次文档变更未运行模拟测试。

必须增加或明确覆盖的反例：

- update/threshold 之间有可观察写入；threshold/reset 之间 pathway 读取 pre state；不同 clock 相同 dt 但 start_tick/steps 不同。
- linked fixed/dynamic alias 读取 summed；summed 在输入更新之前求值；中间 monitor；跨段 run、零步 run 和 checkpoint。
- recurrent pre-read/post-write、跨 projection 冲突、非交换赋值及稳定 edge creation order；heterogeneous pending delay；自定义事件。
- scalar statement 只执行一次；old-state snapshot、固定和 expression refractory、RNG stream/tick/index identity、stateful/non-thread-safe Function。
- 1/N worker、空 shard、degree skew、拓扑替换、run interval/线程 env/Function 实现变化导致的 plan 失效。
- malformed plan 引用/覆盖/重复节点/篡改 effect 被拒绝；超大 topology 的计划尺寸与构建内存不随复制边数组膨胀。

P1/P2 的硬门是相同输入下 generated source bytes（包含 C translation units）、实例布局及完整可观察结果不变。计时等元数据不作 byte equality；性能有自然噪声，不能要求每次耗时相同。测试比较 final state、轨迹、tick/index、event 顺序、refractory、pending、RNG/monitor continuation；不能只比较 spike count。

性能先在同一安静主机建立 old/new paired baseline，固定 revision、原生 rustc/LLVM、worker/affinity、模型、seed、run 与记录选项。沿用 `examples/performance_suite.py --repeats 5 --strict` 的现有门；另记 planner/build 时间和内存，不只看 simulation 时间。source 完全不变的抽取无需立刻重跑巨型 PD14；改变 runtime 或调度后再逐级扩大。

NUMA 专项从 EPYC 1/8/16/32/48/96 物理核及允许的 cpuset 起步，比较单 socket 与跨 socket，记录 binding 的实际结果；测量工具不可用时标为缺测。消融顺序为现有 affinity 开关、owner-local first-touch、queue locality、同步策略；每轮只改变一类因素。CUBA/COBAHH/PoissonInput 区分访存/计算与不同输入开销，PD14/FlyWire 验证真实拓扑与事件量。报告中同时保留 median、spread、RSS、初始化和端到端时间。

GPU 阶段才做目标设备、driver/toolchain、f64 与 reduction 可行性审计；其结果可能改变首个后端选择。研究分支不把缺少实测的 Metal/CUDA/WGSL 通用性写成既有能力。

## 7. 本次研究验证记录

- 已审阅 frozen B2IR、阶段报告、ADR 0001、Python schedule/native/artifact/device、Rust completed effects 和相关测试入口。
- 已确认独立 worktree 基线；本次仅新增本研究及 Proposed ADR，不更改模型、协议和 runtime。
- 外部资料仅为官方架构参考（访问日期 2026-09-06），未做完整竞品调研；引用不能作为本项目性能/独创性证据。
- 文档路径、关键符号与 `git diff --check` 在提交前检查；没有运行性能实验，P0–P5 都仍是待实施工作。

建议下一项实际开发限定为 P0/P1：先让现有 CPU 决策可解释、可复查，再迁移执行权。
