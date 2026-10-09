# 论点、证据与补证矩阵

> 写作阶段记录：以下内容按各节记录日期保留，不是当前实现或验收状态。当前稿件、图表数量和验证入口见[论文索引](README.md)。

## v3 增量证据 — 2026-10-07

以下为 2026-10-07 的写作期检查记录；后续修订与当前结果以正文和验收记录为准，不能将本节旧性能值当作最终表5。

| 论点 | 保留来源 | 可支持结论与限制 |
|---|---|---|
| 当前主线集成与兼容扩展 | [检查快照](data/v3/review_snapshot.json)、[兼容契约](data/v3/compatibility_review.md) | 实现范围与源码身份；不证明整树或每个后端都通过 |
| MPI 可塑性与边界恢复 | [MPI 契约](data/v3/mpi_training_review.md)、[交付 XML/摘要](data/v3/evidence.json) | 显式/binary 的 STDP、pending 和恢复；不继承到 procedural 大网络 |
| 三 realization 分布式描述 | [最新台账](data/v3/mam_acceptance_ledger.md) | 完成与描述分析；无正式科学等价、速度或成本优势 |
| 原生前向/反向与训练状态 | [独立训练阶段](data/v3/evidence.json)、Supplement S12 | 限定 plan/gradient 契约、逐阶段硬件结果；不合计重叠测试、不宣称任意模型可微 |
| 新树突 CPU 结果 | [三组归档 aggregate](data/v3/evidence.json)、[旧数组审计](data/v3/dendritic_network_prior_audit.json) | 12.49×、14.75×、0.861×；限定计时区间与已验收短协议 |
| 外部训练比较边界 | [资格记录](data/v3/external_training_qualification.md)、[架构记录](data/v3/external_training_architecture.md) | 不完整资格与混合 ISA；不构造公平竞品排名 |


证据盘点日期：2026-09-11。以下状态对应记录中的具体版本；分支后续更新不自动继承历史验收。本文档用于写作选材，尚未逐项重新审计所有大型原始文件。

## 1. 版本与依赖边界

| 证据族 | 源码身份 | 用途与限制 |
|---|---|---|
| 当前主目录 CPU/LK | HEAD `0bc46a102290846836c5d1c76e19e6c027c34d2c`，另有未提交改动 | LK 报告与后续冷启动实验分别绑定；不能把工作区当作 clean release |
| GPU 与 execution-plan 集成 | `codex/flywire-mnist` 与 `codex/execution-plan-research` 同为 `d8474dc0145d6b95834931bac12d45f08ce37743` | 后续应用集成快照；具体 GPU 交付证据以各自基线为准 |
| GPU 固定范围交付 | 生产实现基线 `7495df43a`；收尾提交 `a97a92edd` | 固定矩阵已完成，后续提交仍需说明改动范围 |
| MPI | `codex/mpi-cpu`，本次读取 HEAD `e4aa75f7354c8ee2fb1a8efe306d214ac86a7104` | MPI 与最新 GPU 分支独立演进；共同祖先 `5c34cb2dcd3032ffe838b8b066399cc64b6f184e` |
| MPI 内 GPU 卸载 | 实现提交 `740f5ad5e` | 本次讨论中已核对 11 个交付源码哈希与 MPI 工作树一致 |

MPI 与 GPU 共享 B2IR/前端设计基础，但当前不是同一个完整集成发行版。Methods 应列出实验快照，不能把所有数字标成一个未经共同验证的新版本。

## 2. 核心证据索引

### v2 增补 — 执行计划选择与 GPU 校准

- 正文 §2.4、§3.6、§4.5、§5.6 与 Table 3，补充材料 S9。
- 静态分析：completed effects、schedule eligibility、CPU 工作量启发式、目标所有权、canonical fallback、plan identity 与 explain/runtime binding。
- 动态选择：Metal/CUDA 中默认关闭的逐 activation 校准；完整结果指纹约束，三轮候选样本与保守收益门槛，仅最终验证重放发布到 Brian。
- 证据：[计划选择数据](data/plan_selection.json)、[候选和缓存完整表](data/plan_selection_tables.md)，以及六份按归档哈希核对的原始 benchmark report JSON。
- 本次复核：六份报告的字节数和哈希、十二个主机/案例摘要、记录中的策略选择条件、重放指纹、数值门槛、缓存 hit/miss 和中位数。没有重跑全部原始数组审计或硬件测试。
- 保留：M3 样本波动与默认策略保留、quiet 零脉冲、wide 的 f64 不一致、校准开销及缓存实验的编译/分配复用混杂因素。
- 不推出：跨 CPU/GPU/MPI 自动选型、全局最优、通用成本模型、改变输入后的性能决策迁移、独立归因的 planner/cache 加速比。

### E01 — 执行核心独立性

- 论点：前端保留，仿真执行由新引擎接管。
- 来源：[Device 实现](/private/tmp/brian2-flywire-mnist/brian2-rust/python/brian2_rust/device.py:69)、[lowering 实现](/private/tmp/brian2-flywire-mnist/brian2-rust/python/brian2_rust/export.py:482)。
- 已核实：Device 直接继承基础 Device；整网执行路径、前端 state-updater 展开及初始化 NumPy CodeObject 有明确边界。
- 可写：替换仿真执行基础设施，保留 Brian2 建模与前端编译能力。
- 尚需：为 Fig. 1 列出完整调用链，确认每个前端依赖与运行时依赖的归属；引用已有禁止 RuntimeDevice 委托测试。
- 不推出：完全不依赖 Brian2、全部现有模型零修改兼容。

### E02 — B2IR、验证与执行计划

- 来源：[B2IR](/private/tmp/brian2-flywire-mnist/brian2-rust/B2IR.md)、[ExecutionPlan](/private/tmp/brian2-flywire-mnist/brian2-rust/EXECUTION_PLAN.md)、[plan.py](/private/tmp/brian2-flywire-mnist/brian2-rust/python/brian2_rust/plan.py)、[distributed.py](/private/tmp/brian2-mpi-cpu/brian2-rust/python/brian2_rust/distributed.py)。
- 已有：三层身份、dtype/单位/effects/时序、独立 Rust 验证、CPU/GPU 计划与 DistributedPlan。
- 可写：显式语义与可检查策略；在支持范围内进行独立验证和差分测试。
- 尚需：选定 Fig. 2 的调度反例，抽取实际计划；明确 shared frontend/reference 的验证盲点。
- 不推出：形式化证明、IR 首创、跨后端自动最优调度或所有模型完整语义覆盖。

### E03 — 单机 Metal/CUDA 功能交付

- 来源：[GPU_DELIVERY](/private/tmp/brian2-flywire-mnist/brian2-rust/GPU_DELIVERY.md)、[m1-current 审计](/private/tmp/brian2-flywire-mnist/brian2-rust/execution-plan-evidence/m1-current/README.md)。
- 状态：固定范围交付完成。M1/M3/L4/A100 真机结果分阶段归档。
- 本次讨论已执行：离线收尾审计通过，943 个证据引用、60 个幂函数用例/704 字段、84 个 benchmark 快照/672 字段；JSON 规范化后与保存报告一致。
- 可写：各声明功能与数值契约有对应真机验证；历史六项 M1 过严 libm 逐位断言已按原有契约闭环。
- 尚需：将重叠测试按身份去重后再制作 Table 1；完整原始审计报告作为补充材料。
- 不推出：所有 GPU/全部功能组合已测试，或者 float32 等同 reference-f64。

### E04 — GPU 性能与外部对照

- 来源：[population-scale](/private/tmp/brian2-flywire-mnist/brian2-rust/execution-plan-evidence/population-scale/README.md)、[bitset-comparison](/private/tmp/brian2-flywire-mnist/brian2-rust/execution-plan-evidence/bitset-comparison/README.md)、[precompiled recurrent](/private/tmp/brian2-flywire-mnist/brian2-rust/execution-plan-evidence/precompiled-comparison/README.md)。
- 已有：ring STDP 两个规模、随机及较密图、低活动率等；预热后五轮随机顺序重放；Brian2CUDA、Brian2GeNN、GeNN 对照。
- 可用于选图的例子：16,384-neuron ring 的 M1 Metal 44.40 ms，M3 Metal 67.19 ms，L4 CUDA 34.82 ms，A100 CUDA 32.58 ms。每个数属于不同主机，分别比较该主机的基线。
- 计时：reset/初始化到完整主机结果，包含各适配器实际生命周期成本，排除编译与导出。不是 kernel 时间。
- 重要负例：precompiled recurrent CUBA 中 CUDA 慢于 direct GeNN 和同机 Rust CPU；ring 的部分原版外部对照未过数值门。全部保留。
- 尚需：从 raw report 生成统一作图表，分别标 stock/modified/excluded、CPU 控制组性质与精度；公开适配器修改。
- 不推出：GPU 比优化多核 CPU 普遍更快，或巨额重放差异就是 kernel 优势。

### E05 — CPU 吞吐及大模型

- 来源：[CPU 阶段报告](../history/brian2-rust/PROJECT_STAGE_REPORT_20260906.md)、[CPU 回归](../history/brian2-rust/REGRESSION_20260907.md)、[FlyWire 跨机器结果](../history/brian2-rust/FLYWIRE_CROSS_HOST_RESULTS.md)、[PD14/NEST](../history/brian2-rust/PD14_NEST_COMPARISON.md)。
- 已有：多种语义/模型、串行与多线程比较、degree-balanced 与事件路径消融。
- 可写：具体工作负载、线程与编译环境下的收益及自身扩展。
- 尚需：选定一种正式 CPU cohort；查原始报告与样本是否仍可访问。不能混用不同日期的最佳数值拼出总表。
- 保留：旧 PD14 全机比较中 EPYC NEST 最佳配置优于当时 Rust；最新 MPI 模型不是该 PD14 测试的直接更新值。

### E06 — MPI 正确性与 rank-local 存储

- 来源：[MPI](/private/tmp/brian2-mpi-cpu/brian2-rust/MPI.md)、[双节点验证](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/linux-two-node/README.md)、[分区优化](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/rank-local/README.md)。
- 已有：FlyWire EI 完整图四条件、1/2/4 ranks，共 33 次对照/验收，完整 results/events 与独立 reference 逐字节一致。
- 代表值：分片后 1/2/4 ranks 最大单进程 RSS 为 210.0/123.0/76.5 MiB；仿真＋记录＋最终收集中位数为 4.2326/4.4974/4.0587 s。
- 可写：该版本的存储分片降低每 rank 内存，且保持被比较的输出。
- 不推出：近线性扩展。4-rank 对 1-rank 中位数改善约 4.3%，小于 4-rank 样本跨度 7.54%；也不能把单进程 RSS 当作整机总内存。
- 尚需：重新计算 Fig. 3 的数据与标签，按协议说明只读数据/monitor 占位仍有复制。

### E07 — MPI 全规模容量与长记录

- 来源：[100.5 s primary 完成记录](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/primary-run/README.md)、[完整归档](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/primary-archive-v2/README.md)。
- 已有：32 脑区、254 populations、8,344 projections；4,129,924 neurons、24,126,516,728 recurrent synapses；四节点 32 ranks，100.5 s 模型时间，seed 1729。
- 完成记录：所有 rank/guard 成功终止；原始输出审计及修正后的归档完成。该历史运行 wall 为 39,301.456712 s；实际 cgroup CPU 为 196.64248015 core-hours。
- 可写：该网络/配置的工程容量与完成结果。
- 尚需：从最终 terminal/raw metadata 再确认正文数字；区分各节点独立峰值与同步总内存。
- 不推出：生物学论文复现、最优性能、最便宜、实时运行、首次达到该规模；MPI checkpoint 亦未由此得到验证。

### E08 — 最新 MPI Rust/NEST 同口径性能比较

- 来源目录：[performance-runs-v1](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/performance-runs-v1)、[Rust 科学统计复用准备](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/rust-science-reuse-preparation-v2/README.md)。
- 盘点时状态：本次讨论检查的 `rust32-target-v2` 最近快照是 2026-09-11 17:54:09 UTC，32 ranks 已启动，尚无 done 回执；未找到此目标的顶层完成文件。该状态仅来自已有本地快照，不是新的远端实时查询。
- 写作处理：结果槽位保留，等完成、完整 raw audit 与输出/资源口径复核后填值。
- 可替代：v1 先用 E07 证明容量，以既有适配器/小模型对照证明对应语义；不需要为写作另行启动一个全规模运行。
- 不推出：现阶段 MPI 无实现，或已经在该新基准中胜过 NEST。历史运行墙钟因输出/调优/资源不同不直接相除。

### E09 — MPI 内 CPU/GPU 混用

- 来源：[MPI_GPU](/private/tmp/brian2-mpi-cpu/brian2-rust/MPI_GPU.md)、[verification.json](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/heterogeneous-20260911/verification.json)。
- 已核实：118 项通过、12 项环境跳过；测试 XML 与汇总吻合，11 个交付源码哈希匹配。示例 82 spikes，两个 ranks 的实际 GPU dispatch 为 `[0,32]`。
- 实测：本地真实 MPI＋Apple Metal。CPU rank f64、GPU state update f32，必须显式 mixed-f32。
- 可写：按 rank 选择执行设备的集成路径及数值边界。
- 保留边界：threshold/reset/synapses/queue/communication 仍由 CPU 执行；CUDA 代码路径已实现但无此混合路径的 NVIDIA 硬件验收；未验证跨主机混合运行；无加速比结论。
- 论文安排：机制小节或补充示例；不承担主性能贡献。

### E10 — 全规模可塑性与恢复

- 来源：[LK 结果](../history/brian2-rust/LITWIN_KUMAR_RESULTS.md)、[模型定义](../history/brian2-rust/LITWIN_KUMAR.md)、[后续优化记录](../history/brian2-rust/NO_REGRESSION_20260907.md)。
- 已有：4,000 E/1,000 I、约五百万显式连接的 triplet-plasticity 变体；训练与自发阶段、控制实验、敏感性与新进程恢复。其标记为 `paper_reproduction: false`。
- 代表恢复证据：三个新进程重放完整 1,000 s spontaneous phase，41 个最终字段及 11 段记录精确一致。
- 重要边界：native RSS 优势不必转化为完整进程树内存优势；较早一个完整记录 cohort 在 1,000 s 时正有这种反例。恢复准备耗时须与 compiled throughput 分开。
- 尚需：查可访问原始 artifact inventory；分别标注记录内存、最终并行优化及后续未提交冷启动 cohort。
- 不推出：复现原论文全部规则、稳定吸引子、生物学显著性或所有场景端到端更快。

### E11 — 小内存机器上的 PD14

- 来源：[16 GB PD14](../history/brian2-rust/PD14_LOCAL_16GB.md)。
- 已有：77,169 neurons、298,880,968 synapses，Rust 在 16 GB M3 完成 10 s；C++ 前端物化阶段超过控制内存预算，严重 swapping 后停止。
- 可写：此 workload/configuration 下程序化原生构图的可用性差异。
- 不写：macOS 实际发生 OOM kill。报告明确没有观察到内核 OOM kill；也不是完整精度/拓扑位流相同的性能实验。
- 尚需：保留原始系统测量与停止记录；相同逻辑网络、随机建图语义与监控条件准确注明。

### E12 — 多脑区科学统计比较与局限

- 来源：[population statistics](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/primary-cell-comparison-v1/README.md)、[inter-area](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/primary-interarea-v1/README.md)、[LvR 来源诊断](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/primary-lvr-provenance-v1/README.md)。
- 已有：Rust/NEST/原始资料的 rate、LvR、correlation、FC/lag 等描述性对照，后续对分母和历史脚本进行来源诊断。
- 可写：大模型适配的统计行为及尚未闭合的比较；可作为补充科学校验。
- 保留：单一探索 seed；population/pair 不是独立 seed；LvR 历史定义和统计差异有不确定性，高相关不证明等价。
- 不作为 preprint 开写的前置条件：新增独立确认 seed 与完整科学复现，除非论文决定主张该结论。

### E13 — 通用 WASM 运行时、模型导入与浏览器建模

- 来源：[WASM 引擎](/private/tmp/brian2-flywire-mnist/brian2-rust/WASM.md)、[模型导入](/private/tmp/brian2-flywire-mnist/brian2-rust/WASM_IMPORT.md)、[Equation Lab](/private/tmp/brian2-flywire-mnist/brian2-rust/EQUATION_LAB.md)、[wasm.py](/private/tmp/brian2-flywire-mnist/brian2-rust/python/brian2_rust/wasm.py)、[共享执行器](/private/tmp/brian2-flywire-mnist/brian2-rust/src/executor.rs)。
- 已有实现：WasmPlan、浏览器独立 B2IR/plan 验证、共享 native reference 执行核心、Worker 分批执行/取消/隔离、版本化结果、bundle 导入、受限方程编写与结果导出。访问者运行时不需要 Python、Rust 安装或仿真服务。
- 支持范围：共享 reference 已实现的可移植 B2IR，包括多 clocks、pre/post delay、突触状态、summed、custom events、linked variables、typed monitors、portable Functions、RNG、inline/procedural topology。native-only Functions 与 filesystem-backed binary CSR 在通用路径拒绝。
- 既有验证：[浏览器验收索引](/private/tmp/brian2-flywire-mnist/brian2-rust/execution-plan-evidence/wasm/webgpu-20260909/README.md)记录 40 项 WASM 回归与五个独立 committed-HEAD native 对照；后续 Equation Lab 报告另有 51 项 pytest（含 11 项独立 Brian NumPy 模型比较）。这是不同、重叠 cohort，不合计成唯一测试数。
- 数值边界：reference-f64；跨 libm 可有末位误差；同 WASM 执行不同 batch 大小要求逐字节一致。WASM32 内存/索引预算、导出 run 边界及无 checkpoint export 单独说明。
- 可写：从建模/模型导出到浏览器独立执行的完整工作流；浏览器受限 authoring 也是另一个 B2IR producer，进一步展示前端与执行器分离。
- 尚需：按浏览器版本、WASM 二进制与具体 cohort 汇总 Fig. 7；本次仅审阅代码/报告，没有重跑浏览器或硬件测试。

### E14 — 浏览器 WebGPU

- 来源：[WEBGPU](/private/tmp/brian2-flywire-mnist/brian2-rust/WEBGPU.md)、[浏览器结果](/private/tmp/brian2-flywire-mnist/brian2-rust/execution-plan-evidence/wasm/webgpu-20260909/browser-report.json)。
- 已有：经 WASM 语义验证后生成 WGSL，Worker 中运行独立细胞 f32 kernel，显式选择后端、分块提交、稳定 spike bitmap 读回及资源限制。支持指定 built-in 模型和受限 Equation Lab。
- 实测边界：不支持突触、延迟队列、STDP、多 clocks 等通用网络能力；不可继承单机 Metal/CUDA 的功能声明。Izhikevich 等模型已有明确 f32/f64 spike/count/时间差异。
- 可写：浏览器 GPU 路径的存在、适用范围及观测到的精度/运行权衡。
- 不推出：WebGPU 可运行全图 FlyWire、任意 Brian2 模型或与 f64 普遍逐脉冲一致。官方 profile 标为实验性，应保留该状态。
- 尚需：保留同机配对样本、最终输出 guard 与早期计时版本之间的关系；将 smoke 时长与重复性能测量分列。

### E15 — 全图 FlyWire 的模型专用 WASM AOT 与离线分发

- 来源：[浏览器全图路径](/private/tmp/brian2-flywire-mnist/brian2-rust/validation/FLYWIRE_MNIST_BROWSER.md)、[wasm-check.json](/private/tmp/brian2-flywire-mnist/brian2-rust/validation/evidence/browser-local-20260910/wasm-check.json)、[browser-check.json](/private/tmp/brian2-flywire-mnist/brian2-rust/validation/evidence/browser-local-20260910/browser-check.json)、[分发清单](/private/tmp/brian2-flywire-mnist/brian2-rust/validation/evidence/browser-local-20260910/distribution.json)。
- 已有：把冻结的原生模型专用 AOT Rust 源码编译为 WASM，通过 memory-only host 替代文件/时钟/环境访问；保留生成方程、schedule、RNG、event queues、f64 与序列化。
- 容量：完整 139,255-neuron、15,091,983-edge 模型在本地 Worker 中运行；静态包包括模型/读出及相关资源。模型资源约 54 MiB；报告的 WASM linear memory 约 534 MiB，不包含 JS、渲染及加载临时副本。
- 检查：13 次新鲜仿真（0–9 与 A/B/A），输入/活动哈希及预测匹配 CPU oracle，score 最大误差小于 8e-14；另有 26 个图像预处理 fixtures。真实浏览器测试含停止静态服务器后从缓存重载并继续本地识别。
- 可写：完整模型作为静态制品分发，在访问者浏览器内执行与离线使用；这是 Fig. 7 的主要应用证据。
- 边界：该 AOT profile 是模型专用实验路径，不代表通用 BrowserExecutor 新增任意全图 CSR 能力；缓存可被浏览器清除；不是手机内存验收。没有在 WASM 重跑全量 10,000 张准确率评估，CPU 准确率不能挪用。
- 本次核查：读取保存的 JSON 与实现/文档；未重新编译或执行 13 次仿真。

## 3. 补证优先级

| 优先级 | 工作 | 触发/完成条件 |
|---|---|---|
| P0，写作基础 | Fig. 1 调用边界与 Table 1 能力矩阵 | 每项绑定当前源文件/测试；原生 CPU/GPU、通用 WASM、WebGPU、模型专用 WASM AOT、MPI 分列 |
| P0，结果可复核 | 每张主图建立数据清单 | 源码 hash、输入 hash、环境、原始样本、计时定义、资格门与图表脚本齐备 |
| P0，分布式主结果 | 冻结 E06–E07 对应 Fig. 3 的容量、分片内存及资源数据 | 明确构图/执行/输出范围，绑定多脑区适配器与模型身份；容量、扩展效率、相对性能分别陈述 |
| P0，语义解释 | 选已有调度反例形成 Fig. 2 | 独立期望结果、reference、实际 plan 和优化前后结果一致 |
| P0，主张收口 | 完成逐项 related-work 对照 | 能说清 B2IR 与 NIR、计划与既有 codegen、MPI 与既有分布式系统的具体差异 |
| P1，条件性结果 | 跟进 E08 已有作业的完成证据 | 顶层完成、全 raw/资源/计时审核通过后再写最新速度比较；不是新启动授权 |
| P1，发布可用性 | 选择统一集成快照或透明的分支制品集 | 能独立构建与运行最小 CPU/GPU/WASM/MPI 示例；若合并，测试被改变的共用边界 |
| P1，机制归因 | 检查所选消融的版本和输入一致性 | 现有数据能支持即可；缺失且影响核心主张才补测 |
| 后续研究 | 混合 MPI-CUDA 真机、更多科学 seeds、完整强/弱扩展 | 只有提升对应论文主张时才扩展，不当作已有功能交付失败 |

## 4. 每张图的最小数据清单模板

```yaml
figure_id: F4
panel: a
claim_id: C2
evidence_id: E05
source_revision: REQUIRED
source_manifest_sha256: REQUIRED
model_definition_sha256: REQUIRED
instance_sha256: REQUIRED
run_identity: REQUIRED
host_and_runtime_manifest: REQUIRED
numeric_profile: REQUIRED
baseline_variant: REQUIRED  # stock / disclosed-modified / numerical-control
timing_scope: REQUIRED
required_outputs: REQUIRED
warmups: REQUIRED
repeats: REQUIRED
execution_order: REQUIRED
raw_report: REQUIRED
correctness_gate: REQUIRED
memory_scope: REQUIRED
plot_script: REQUIRED
exclusions_and_failures: REQUIRED
```

此模板是图表元数据要求，不是本轮已经采集的数据。未知字段应保留为未知并说明影响，不能用估计值伪装成观测值。


## v3 已发表模型追加

| 论点 | 来源 | 边界 |
|---|---|---|
| 显式 NMDA 状态与性能 | [机器记录索引](data/published_models/evidence.json)、Table 4 / Figure 8 | 原科学方程；相同输入诊断与性能输入分开；最大规模没有完整确定性门 |
| Onasch 指定结果复现 | [Fig. 2/S1 ensemble 与全文矩阵](data/published_models/evidence.json)、S14 | 原模型源环境扫描与 Atlas 配对执行不同；不声称整篇复现 |
| 用户协调负担降低 | Device 引擎选择、B2IR/plan/result 路径；官方目标项目入口；S15 | 可检查的架构集成与设计目标；没有用户时间或操作数测量 |

## v3 兼容性附录

| 论点 | 来源 | 边界 |
|---|---|---|
| 各目标功能兼容范围 | Appendix A / Table 6；S16；[契约来源索引](data/compatibility/evidence.json) | B 为受限实现契约，X 为明确排除，NR 为本稿未建立全面资格；不将源码集成或共用 IR 当作全后端联合验收 |

## 异构 MPI 组合资格

| 论点 | 来源 | 边界 |
|---|---|---|
| 按 rank 选择 CPU/Metal/CUDA | §3.5 / §5.9 / Figure 9；[证据索引](data/mpi_heterogeneous/evidence.json) | CPU+Metal 仿真实测；CUDA 仿真实现没有该路径的硬件资格；CUDA MPI 训练为单 L4 上另一计划族；没有跨主机多厂商组合或速度结论 |

### Online circuit activity and B2IR inspection

Section 2.4 / S18 / Supplementary Figure S1: user-selected saved FlyWire DM1 WASM run (240 neurons, 300 ms, 2,796 full-run spikes, replay at 270 ms), paired with the online model's default-parameter B2IR draft (one execution population, 6,660 synapses, seven scheduled operations). The draft is not the saved execution artifact; the current catalogue supplies the network layout. Retained template SHA-256 identifies the downloaded draft input, not the run or Python source. See data/b2ir_visualization/capture.json. Interface illustration only; no physical optimization-path, usability, performance or scientific-validation claim.

### Connected synthetic capacity

Section 4.8 / 5.10 / Figure 10 / S19: complete-output acceptance at 86M neurons/86B edges and 128M neurons/128B edges, each for 100 ms on 30 populated ranks and 30 physical hosts, one worker core per host, reference-f64. The shared 24k-neuron pilot passes 303 exact independent-reference checks; the large runs pass engineering audits and terminal resource gates, not independent full large-graph reference execution or anatomical scientific validation. Isolated source/default-limit distinctions, initial-value-budget rejection and other incomplete configurations are retained. See data/capacity/evidence.json. These are single capacity observations, not maximum capacity, strong/weak scaling, matched simulator speed or full-scale 86B-neuron feasibility evidence.
