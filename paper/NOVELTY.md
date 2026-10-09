# 核心创新性定位与审稿问题

> 写作阶段记录：以下内容按各节记录日期保留，不是当前实现或验收状态。当前稿件、图表数量和验证入口见[论文索引](README.md)。

核查日期：2026-09-11。本文修正“以执行目标覆盖面作为标题中心”的前一版定位；保留浏览器、CPU/GPU 和 MPI 的全部范围。对照基于下列官方文档和已审阅源码，完整发表比较仍须锁定各外部项目 revision。

## 1. 已确定题目与核心主张

> **A Unified Intermediate Representation and Execution Architecture for Heterogeneous and Distributed Neural Simulation**
>
> **面向异构与分布式神经仿真的统一中间表示与执行架构**

“From Browsers to Clusters”可用于架构图标题或引言中的应用跨度描述。Brian2 前端作为系统边界在摘要首段明确，不必让主标题依赖 Brian2 的某个目标扩展名称。

核心主张：

> We present a unified intermediate representation and execution architecture for heterogeneous and distributed neural simulation, retaining Brian2 as a modeling frontend. B2IR makes model execution semantics explicit and supports rebuilt CPU, GPU, and browser execution within declared capability and numerical contracts. Distributed topology construction, partitioned state and synaptic storage, and ordered event exchange extend execution across nodes.

写作时分别论证统一 IR 的语义内容、重写后端的平台与资源收益，以及分布式实现的规模与效率。分布式部分重点解释模型构建和存储如何随分区分摊、跨 rank 执行如何落实时序与随机身份，以及实际资源代价。

新引擎是项目新实现；独立实现、研究新颖性、性能优势是需要不同证据的三个判断。代码量、重写语言和支持目标数量不能单独证明研究新颖性。

## 2. 已有 Brian 扩展的准确定位

| 系统 | 已核实的结构事实 | 对本文定位的影响 |
|---|---|---|
| Brian2 核心 | 抽象更新语句经 CodeGenerator/CodeObject/templates 进入编译与运行，Device 提供扩展点。[官方开发文档](https://brian2.readthedocs.io/en/stable/developer/codegen.html) | 已有抽象代码和可扩展执行体系；不能声称首次分离前端、首次中间表示或首次 code generation |
| Brian2CUDA | 官方源码中 `CUDAStandaloneDevice` 继承 `CPPStandaloneDevice`，实现 CUDA 生成及执行。[官方源码](https://brian2cuda.readthedocs.io/en/latest/_modules/brian2cuda/device.html) | GPU 支持与独立原生执行不能作为本工作的独有贡献；比较应落到具体语义表示、计划和运行结构 |
| Brian2GeNN | 文档将其定义为连接 Brian 与 GeNN 的 middleware，复用 cpp_standalone 代码并生成 GeNN 模型/代码；GeNN 本身有自己的生成与仿真系统。[官方架构](https://brian2genn.readthedocs.io/en/1.6/technical/how_it_works.html)、[源码](https://raw.githubusercontent.com/brian-team/brian2genn/master/brian2genn/device.py) | “保留 Brian2 前端＋独立执行器”也已有先例。必须解释统一的全模型 IR、验证/计划与多目标实现如何区别于这条管线 |
| Brian2Wasm | 官方仓库说明将 Brian 模型经 Emscripten 编译为 WebAssembly/JavaScript，输出可独立运行的网页目录。[官方仓库](https://github.com/brian-team/brian2wasm) | 浏览器执行与自包含网页分发已有先例；目前本次未完成其内部源码审计，不能推断它没有某类内部 IR 或模型验证 |

上述继承关系是实现边界事实，不是质量或先进程度排名。不能把 GeNN 描述成 Brian 的模板扩展，也不能把所有旧系统概括为“没有 IR”。

## 3. 本文“统一”的可检查定义

| 统一层次 | 本文具体含义 | 证据 |
|---|---|---|
| 模型执行语义 | B2IR 显式记录 schedule/clocks、typed state/units/domains、effects、events/delays、RNG identities | B2IR schema、独立验证器、正反例；E01–E02 |
| 计划的语义基础 | CPU/GPU、WasmPlan、DistributedPlan 从共同模型语义及 logical dependencies 出发 | 实际 plan 结构、lowering 和 plan 校验；E02、E06、E13 |
| 执行身份与比较 | definition/instance/run 身份和数值 profile 绑定模型、计划与运行输出 | hash 与验证链、差分检查、各目标 runtime binding；E02–E04、E06、E09、E13–E15 |
| 运行实现 | native reference 与通用 WASM 共享 Rust 核心；CPU AOT、Metal/CUDA 和 MPI 使用各自实现落实声明的契约 | 源码边界、同 IR/同 profile 的交叉执行和目标特定测试 |

“统一”不要求所有目标使用相同 kernel、物理计划类或数值位宽。各目标能力和 numeric profile 必须显式；当前分支集成状态也须公开。WebGPU 独立细胞 profile 和模型专用 WASM AOT 路径分别说明，不能拿它们代替完整通用 IR 消费路径。

## 4. 最值得回答的审稿问题

| 审稿问题 | 回答策略 | 应给出的证据 |
|---|---|---|
| 已有 Brian2CUDA/GeNN/Wasm，为什么需要这个系统？ | 将完整模型执行语义作为独立契约，解释它怎样约束多目标运行与全局变换 | Fig. 1 的四条架构对照；一个模型从 IR 到不同计划的具体例子 |
| 分布式神经仿真已有成熟系统，本文贡献是什么？ | 保留 Brian2 建模，并将显式执行语义落实到分布式构图、局部存储和跨 rank 事件处理；逐项比较实现与代价 | E06–E08；四节点规模完成证据、分区内存对照、同口径 NEST 对照；不能以规模相同推断效率相同 |
| B2IR 比 Brian 抽象代码增加了什么？ | 展示完整网络的时序、依赖、状态/索引域、事件、随机与运行身份 | 字段到可观察行为的映射；一个跨对象依赖反例 |
| 是否只是常见 compiler IR 架构在神经仿真的应用？ | 承认 IR/计划分层的先行工作，给出神经仿真特有的约束、算法和实验价值 | event/reset/refractory、延迟到达顺序、随机身份、monitor 可见性及分区实现 |
| “统一”是否只是共同 JSON 包装？ | 对同一份 IR 的共同可执行子集，在多个消费者上比较行为；解释后端差异 | 全部状态/事件/时间及配置身份；WASM/reference 共享实现不能独立排除共享 bug |
| 从头重写为何是研究贡献？ | 展示重建后能实施并验证哪些以前管线中不同的决策 | 依赖约束下的融合/owner routing、分片构图/内存、模型移植；进行针对性消融 |
| 验证是否构成正确性证明？ | 分清静态校验、差分、独立期望与科学复现 | malformed IR、plan 篡改、数值/事件正例及共享依赖盲点；不宣称形式化证明 |

## 5. 对现有大纲的调整

- Introduction 以沿用 Brian2 建模的跨节点仿真需求引出 B2IR 及分布式执行架构；浏览器到集群作为覆盖范围和检验场景。
- C1 为统一 IR 与可检查语义；C2 为重写的异构后端、Metal 平台覆盖及性能/内存收益；C3 为分布式构图、存储和执行，以及与成熟系统的规模和效率比较。
- Fig. 1 增加既有管线与新系统的边界对照，所有外部项目按已核实版本描绘。避免以“旧系统没有 IR”概括。
- Fig. 2 给出同一 B2IR 的多个计划及语义不变量；合法性和性能收益分开。
- Results 开头先证明声明范围的语义一致性，随后呈现分布式容量与资源结果，再展示 CPU/GPU 和浏览器场景。
- Related Work 加入 Brian2Wasm；NIR/SNN-MLIR 等仍需比较，不能因区别于 Brian 扩展就得出领域首次结论。

可优先复用已有数据组成共同子集验证表。若现有各目标用了不同模型实例，应明确标出，并仅为支撑“同 IR 多执行器”的核心主张补充一个小型对照；不需要为了命名统一重跑全部大型实验。

## 6. 分布式主张的证据层级

“让 Brian2 建模进入可与 HPC 神经模拟器共同评估的问题规模”可作为应用价值表述。论文的量化结论应落在当前完成的四节点、32 ranks、4,129,924 神经元、24,126,516,728 递归突触和 100.5 s 模型时间上（E07）。这属于集群规模执行证据，不能外推成数千节点扩展效率。

- **容量**：已完成网络、每节点资源和分片内存数据支撑，可作为 preprint v1 主结果。
- **扩展效率**：需要固定问题的 strong scaling 或明确增长规则的 weak scaling，以及通信/计算分解；现有小规模 rank 时间不能替代。
- **竞争性效率**：需要模型、数值、输出、资源和计时口径一致的 NEST 等对照。E08 的历史读取状态不能作为当前完成证据。

NEST 与 NEST GPU 的既有分布式研究应正面引用（R5–R6）；全规模多脑区模型已有先行工作。本文须以 Brian2 前端到分布式执行的具体系统设计及可复核结果建立贡献，避免领域首次或已经胜过成熟 HPC 系统的推断。
