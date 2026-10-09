# 相关工作与引用计划

> 写作阶段记录：以下内容按各节记录日期保留，不是当前实现或验收状态。当前稿件、图表数量和验证入口见[论文索引](README.md)。

检索日期：2026-09-11。以下是开写所需的核心文献清单与论证用途，已核对题目/来源；不代表穷尽检索或完成逐篇算法审计。引文键供 MANUSCRIPT.md 使用，正式全文阶段从出版方导出完整书目信息。

## 1. 核心文献

| 键 | 文献及原始来源 | 正文用途 | 应逐项比较的问题 |
|---|---|---|---|
| R1 | Stimberg, Brette & Goodman (2019), [Brian 2, an intuitive and efficient neural simulator](https://elifesciences.org/articles/47314), eLife. DOI: 10.7554/eLife.47314 | 前端来源与设计基础 | 哪些建模、数值展开和实验协议能力被保留？执行生命周期改变了什么？ |
| R2 | Yavuz, Turner & Nowotny (2016), [GeNN: a code generation framework for accelerated brain simulations](https://pmc.ncbi.nlm.nih.gov/articles/PMC4703976/), Scientific Reports. DOI: 10.1038/srep18854 | 代码生成与 GPU 系统 | 代码生成阶段、模型/执行边界、连接与事件布局；对照时锁定实际 GeNN 版本 |
| R3 | [Brian2GeNN: accelerating spiking neural network simulations with graphics hardware](https://www.nature.com/articles/s41598-019-54957-7) (2020), Scientific Reports. DOI: 10.1038/s41598-019-54957-7 | Brian 前端与另一执行器的连接 | 中间适配、生命周期和支持范围；本项目修正适配器与原版须分别标注 |
| R4 | [Brian2CUDA: Flexible and Efficient Simulation of Spiking Neural Network Models on GPUs](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.883700/full) (2022), Frontiers in Neuroinformatics. DOI: 10.3389/fninf.2022.883700 | GPU 调度、事件、延迟和基准 | 比较同模型/同数值模式的执行策略；区别 kernel 时间与完整 standalone replay |
| R5 | Jordan et al. (2018), [Extremely Scalable Spiking Neuronal Network Simulation Code: From Laptops to Exascale Computers](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00002/full), Frontiers in Neuroinformatics. DOI: 10.3389/fninf.2018.00002 | NEST 的分布式数据结构与扩展 | rank 所有权、通信、连接存储及强/弱扩展口径；引用算法细节前检查勘误 |
| R6 | Tiddia et al. (2022), [Fast Simulation of a Multi-Area Spiking Network Model of Macaque Cortex on an MPI-GPU Cluster](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.883333/full), Frontiers in Neuroinformatics. DOI: 10.3389/fninf.2022.883333 | 同类大规模案例与 MPI-GPU 先行工作 | 其已研究约 4.1 million neurons/24 billion synapses；本工作以执行架构与前端保持为论点，不能以规模本身声称首次 |
| R7 | Pedersen et al. (2024), [Neuromorphic intermediate representation: A unified instruction set for interoperable brain-inspired computing](https://www.nature.com/articles/s41467-024-52259-9), Nature Communications. DOI: 10.1038/s41467-024-52259-9 | IR 抽象层与跨平台表达 | NIR 连续时间模型原语与 B2IR 的离散 schedule、state-updater 语句、effects、运行身份分别表达什么？这是基于现有资料的比较方向，须精读后定论 |
| R8 | [SNN-MLIR: An MLIR Dialect for Compiling Neuromorphic SNNs from NIR to Bare-Metal C](https://arxiv.org/abs/2606.09213) (2026), arXiv:2606.09213 | 最新 SNN 编译/IR 相关预印本 | 原语 lowering、类型和目标代码；明确其预印本状态，不推断其没有尚未审阅的功能 |
| R9 | Potjans & Diesmann (2014), [The Cell-Type Specific Cortical Microcircuit: Relating Structure and Activity in a Full-Scale Spiking Network Model](https://publications.rwth-aachen.de/record/231390/), Cerebral Cortex. DOI: 10.1093/cercor/bhs358 | PD14 模型来源 | 本项目 DC/Poisson 输入、拓扑抽样、时间映射和记录窗口与对应参考配置的关系 |
| R10 | Dorkenwald et al. (2024), [Neuronal wiring diagram of an adult brain](https://www.nature.com/articles/s41586-024-07558-y), Nature. DOI: 10.1038/s41586-024-07558-y | FlyWire 数据/连接组来源 | 实际导入版本、加权边与生物接触数量；引擎 workload 的动力学假设单独说明 |
| R11 | Schmidt et al. (2018), [A multi-scale layer-resolved spiking network model of resting-state dynamics in macaque visual cortical areas](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006359), PLOS Computational Biology. DOI: 10.1371/journal.pcbi.1006359 | 多脑区模型与科学观测量来源 | 参数来源、时间/不应期/输入映射、统计定义、模型运行与科学复现的区别 |

**R12 — Brian2Wasm**：官方软件项目，[仓库与架构入口](https://github.com/brian-team/brian2wasm)。官方说明通过 Emscripten 将 Brian 模型编译为 WebAssembly/JavaScript 并分发为网页目录。用于浏览器运行/模型分发先行工作比较；本次未完成其内部源码审计，不推断不存在内部 IR。详见 [创新性比较](NOVELTY.md)。

## 2. Related Work 的四段结构

1. **建模前端与代码生成**：Brian2 的原有贡献 → GeNN/Brian2GeNN/Brian2CUDA 的执行路径 → 本文将语义验证、计划与运行时分离的具体边界。需要代码/论文两层证据。
2. **模型表示与执行表示**：NIR 及其他模型交换工具 → 本文对离散执行时序、effects、RNG 与运行身份的表达 → 这种更具体的契约带来的适用范围与限制。
3. **分布式神经仿真**：NEST 的内存/通信设计及 NEST GPU 多脑区案例 → 本文的前端保留、分布式构图和 owner-local 执行 → 完整网络容量与扩展性能分别比较。
4. **现代编译基础设施**：SNN-MLIR 与相关编译工作 → 本文使用的实际计划机制 → 尚未提供的形式化证明或跨目标自动优化能力。

## 3. 投稿前补充检索

| 主题 | 需要补充的材料 | 用途 |
|---|---|---|
| PyNN / NeuroML / LEMS | 原始论文与规范 | 补足前端/API、模型交换、动力学语义之间的层次比较；当前尚未完成该项精读 |
| ANNarchy / Arbor / NEURON | 与本文范围直接相关的原始论文 | 防止把通用 codegen、分片或并行概念误写为独有贡献 |
| MLIR / XLA / TVM | 原始论文/官方编译文档 | 为 logical/physical plan、合法性与成本分离提供计算机系统背景 |
| Litwin-Kumar–Doiron 及其变体规则 | 模型原文与本项目源审计 | 为 Fig. 6 区分原模型、triplet 变体、生命周期实验和科学结论 |
| 可复现性与数值误差 | 与 SNN 阈值、随机流、累加顺序直接相关的原始研究 | 支撑精度与语义验证方法，而不是以常识代替论证 |
| 浏览器科学计算与模型分发 | WebAssembly 原始论文/规范、WebGPU 规范，以及直接相关的浏览器神经仿真系统原始研究 | 比较模型移植、验证、运行依赖、Worker 生命周期与交互工作流；不把使用 WASM 本身当作创新 |

上述条目是后续检索任务，不作为已经核实的文献结论使用。当前清单足以开始架构写作；创新性最终措辞等逐项比较完成后收紧。


## v3 新增模型来源

**R14 — Contextual dendritic gating model software**：Onasch、Miehl 等作者的[原始模型仓库](https://github.com/computational-neural-circuits/contextual-dendritic-gating/tree/dbb77525f2662199544f5a0d3dcc9c18b0e1c853)，冻结 commit dbb77525f2662199544f5a0d3dcc9c18b0e1c853。本版引用模型源码作为 Table 4 工作负载来源。公开仓库说明已核对；本文不以新增选定工作负载验收声称整个原研究的科学复现。


**R15 — Skaar, Haug & Plesser (2025)**：[A simplified model of NMDA-receptor-mediated dynamics in leaky integrate-and-fire neurons](https://link.springer.com/article/10.1007/s10827-025-00911-8), Journal of Computational Neuroscience 53, 475–487。原显式/general Brian2 基准作为 Atlas 目标；科学近似方法的 NEST 成本优势与 Atlas 引擎加速分别引用和解释。
