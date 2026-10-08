# 新引擎 preprint 写作材料包

> 2026-10-09 更新：当前实现固定到 Atlas [`bf1cf30af55a4a14ae42d0d75534728385b62d06`](https://github.com/RockLi/brian2-atlas/tree/bf1cf30af55a4a14ae42d0d75534728385b62d06)。中英文正文同步更新源码引用及709项Modal CUDA验收；PD14在该提交上重新构建和运行。历史性能测量保留实际来源版本。

> 双仓库迁移说明：下文保留写作阶段与历史版本记录；当前图表数量和检查范围以 [构建报告](validation/build_report.json) 为准。HTML 重建使用本仓库保留的数据，PDF 导出见 [PDF 构建说明](PDF.md)。历史结果继续对应各自的源码身份。

更新日期：2026-10-07。阶段：默认阅读稿已采用正式稿件样式，九张图、六张表（含附录兼容矩阵）、15 项引用及 S1–S18 补充材料。本版基于本地 next-dev（含未提交开发）更新集成状态、MPI 可塑性与边界恢复、兼容范围、多 realization 分布式证据、原生训练和树突模型基准；保留执行计划研究与 Figure 7 的真实 Neural Lab 界面。尚未公开发布或投稿。

**论文定位：保留 Brian2 建模与前端编译能力，围绕统一 IR、重写的异构执行后端和分布式实现三项贡献，呈现模型语义、平台覆盖、性能、内存效率与跨节点容量。**

已确定题目：

> **A Unified Intermediate Representation and Execution Architecture for Heterogeneous and Distributed Neural Simulation**
>
> **面向异构与分布式神经仿真的统一中间表示与执行架构**

作者署名已于 2026-10-06 确定为 **Xinjun Li**，身份/单位行使用 **Independent researcher**。项目名称已定稿为 **brian2-atlas**，论文题目保持不变。源码目录 `brian2-rust` 与 Python 模块 `brian2_rust` 是当前实现路径和标识，复现材料按实际名称保留。当前实现通过可选 Device 接入，不代表 Brian 团队的官方替代版本。

## 阅读入口

| 文件 | 用途 |
|---|---|
| [英文完整初稿 · 阅读版](MANUSCRIPT.html) | 内嵌九张图的 HTML，可离线阅读 |
| [论文 PDF 构建方法](PDF.md) | 正式稿件样式，含系统能力、实验结果与兼容附录 |
| [英文完整初稿 · 编辑版](MANUSCRIPT.md) | 摘要、引言、方法、结果、讨论、结论及 15 项引用 |
| [补充方法与证据](SUPPLEMENTARY.md) | 源码边界、实验配置、原始报告定位、证据层级及复现命令 |
| [计划选择与调优数据](data/plan_selection.json) | 六份归档报告、十二个主机/案例摘要、来源哈希 |
| [候选与缓存完整表](data/plan_selection_tables.md) | 全部候选的样本范围，以及另一批缓存实验的完整成本 |
| [版本说明](CHANGELOG.md) | v3 与历史版本的结构与证据更新 |
| [图表数据](data/figure_data.json) | MPI/GPU 记录样本、CPU/LK 报告汇总、来源哈希和浏览器检查 |
| [Neural Lab 截图留档](data/neural_lab_capture.json) | 界面构建、截图与导出实验哈希，及单次演示的范围 |
| [外部 GPU 比较全表](data/gpu_comparisons.md) | 全部策略、修改适配器及被排除的资格失败项 |
| [创新性与审稿问题](NOVELTY.md) | 与 Brian2CUDA/GeNN/Wasm 的架构区别、统一的定义及论证证据 |
| [论文大纲](OUTLINE.md) | 核心问题、三项贡献、章节论证、九组主图和写作顺序 |
| [原始写作骨架](MANUSCRIPT_SCAFFOLD.md) | 保存完整正文写作前的结构，供历史对照 |
| [证据与实验矩阵](EVIDENCE_MATRIX.md) | 论点到源码/报告的对应、数值口径、缺口及补证优先级 |
| [相关工作与引用计划](REFERENCES.md) | 已核实的关键文献、比较维度及仍需精读的问题 |

## 已确定的写作原则

- “替换”指执行架构的独立性。Brian2 前端仍负责对象、方程、单位、符号解析及数值方法的抽象语句展开；新引擎负责其后的编译、执行与结果管理。
- 兼容范围按后端列出。CPU、单机 GPU、通用 WASM、WebGPU、模型专用 WASM AOT、MPI 和 MPI 内 GPU 卸载的能力不能相互代替。
- 算法和系统解释围绕同一语义契约展开；Rust 作为实现选择介绍。
- 已有的 GPU/MPI 与 WASM 执行能力进入正文。进一步性能调优或科学确认不自动成为整篇论文的前置条件。
- 每个实验图绑定确切版本和数据。主线代码已集成；历史分支与后续冻结版本的证据分别保留，不归为一个共同验收版本。
- 保留速度较慢、精度分歧及科学统计不一致的结果。测试通过、性能收益、科学模型复现是不同结论。

## 本材料包的检查范围

历史 v1/v2 材料依据主目录、`codex/flywire-mnist` / `codex/execution-plan-research` 和 `codex/mpi-cpu` 的相关实现与报告，包含当时的离线审计与语义回归。v3 对照当前 next-dev，记录已检查源码身份，保留 27 份新增小型来源记录，并核对交付 XML、终态摘要、归档哈希和数值算术。本轮工作为论文修订与排版，没有启动硬件基准、大型仿真或远端作业。

WASM 作为可移植运行时进入正文：覆盖浏览器独立验证、Worker 内执行、模型导入/编写与分发；WebGPU 和全图 FlyWire WASM AOT 分别注明独立范围。主标题突出统一 IR 以及异构与分布式执行架构；“从浏览器到集群”用于概括部署跨度，具体执行目标在摘要和架构图中展开。

v3 已包括 Fig. 1–9、Table 1–6 和测量方法。四节点 32-rank、24,126,516,728 递归突触的完成记录支撑集群规模执行能力；最新 NEST 对比没有被写成已完成的速度结论。

## 历史 v2 核验与公开前事项

- 重跑跨 population 的 summed/linked 语义回归：reference、AOT、Brian NumPy 对照通过。
- 从现有 MPI/GPU JSON 提取样本，检查中位数、资格失败、内存单位及来源哈希。MPI 33 次记录明确分为 6 次预热、18 次静息测量和 9 次其他条件检查。
- v2 核对六份归档调优/缓存报告的字节数和 SHA-256，复核十二个主机/案例摘要、全部候选的选择门槛、记录的结果指纹与缓存命中序列；没有重跑硬件仿真或全部原始数组审计。
- 七张图已生成 SVG/PNG；检查图像布局、数值和参考文献链接。CPU/LK 部分明确为报告汇总，未虚构原始样本或误差条。
- [构建核验报告](validation/build_report.json)记录检查范围。图表重建脚本为 [build_manuscript.py](scripts/build_manuscript.py)，不会启动硬件基准。

署名与身份/单位行已确定。作者贡献、无外部资助及无利益冲突声明已根据作者确认写入。公开前仍需补齐公共源码/数据归档地址；定位 CPU/LK 对应原始制品并核对公开版本；审阅方法与结果主张。正文写作已经完成，这些是从内部初稿转为可公开 preprint 的收尾工作。


## v3 新增材料与版本边界

- [v3 证据索引](data/v3/evidence.json)：归档报告、验收终态、三个新增 CPU 工作负载与独立训练阶段的范围。
- [本地源码检查快照](data/v3/review_snapshot.json)：next-dev 基础 commit、未提交状态及已读源码哈希；不是干净 release 或整树验收证明。
- [历史v3证据收集脚本](scripts/collect_v3_evidence.py)：用于原始v3归档收集，不用于覆盖已加入统一源码复测的最终证据索引。当前稿件从保留的索引重建。
- [保留的 v2 阅读版](MANUSCRIPT_V2.html)、[v2 编辑版](MANUSCRIPT_V2.md)、[v2 补充材料](SUPPLEMENTARY_V2.md)、[v2 PDF 归档索引](../archives/README.md)。

v3 保留原 CPU/GPU/MPI 基准的冻结身份，增加三组树突模型 Cython 对照及较慢的网络案例。新增 MPI 恢复属于显式/binary 小模型验收；不将它等同于大型 procedural 网络恢复。三个 Rust realization 与 NEST reference 只支持描述性比较。原生训练按版本和硬件列出资格，不宣称全 Brian 可微、最新 CUDA 全部通过或跨主机多 GPU 扩展性。

本版沿用已确认标题、项目名称 brian2-atlas 和 Xinjun Li / Independent researcher 署名。发布前仍需补公共源码/数据地址并完成作者对全文的最终审阅。


## v3 已发表模型与生态统一补充

- §5.8 现在覆盖 Skaar 显式 NMDA 与 Onasch 树突门控两项工作流；新增 Figure 8 / Table 4，原树突性能表成为 Table 5。
- [已发表模型证据](data/published_models/evidence.json)保留 15 份小型来源记录；NMDA 计时与状态验收、Onasch 原模型扫描与 Atlas 配对执行分别表述。
- Supplement S14 提供范围矩阵；S15 说明统一 Device/B2IR/计划/结果接口与独立浏览器交付，保留各目标限制。
- 摘要、引言及讨论加入减少多项目协调与模型迁移负担的用户价值；不将这一设计目标写成已测量的易用性优势。
- [初始 v3 PDF 归档索引](../archives/README.md)、[初始 v3 源稿](MANUSCRIPT_V3_INITIAL.md)、[初始 v3 阅读版](MANUSCRIPT_V3_INITIAL.html)已保留；当前默认文件为补充后的 v3。

## v3 兼容性附录

Appendix A / Table 6 按 CPU、CUDA/Metal、通用 WASM 和 MPI 列出功能边界；S16 展开限制和版本来源。B 为受限实现契约，X 为该目标明确排除，NR 为本稿证据未建立全面验收。NR 不等于未实现，也不将共用 IR 等同于各目标全功能支持。

## v3 作者声明

2026-10-07 根据作者确认替换占位文字：无外部资助，计算及论文准备费用自行承担；无需披露的利益关系。作者贡献与软件/模型/数据致谢分别列出。工具使用说明按作者要求留待后续人工定稿。

## 正式稿件样式

默认源稿和阅读版移除封面版本/日期/内部状态、页脚工作稿标识、写作过程占位文字及补充材料发布清单。PDF 仅保留页码，默认输出为 `output/pdf/brian2-atlas-preprint.pdf`（仓库根目录；见 [构建说明](PDF.md)）。科研验证边界与来源版本继续保留；本轮为稿件内容和版式整理，未公开发布。此前 v3 PDF 与 V3_PRE_FINAL 源稿保留用于内部对照。

## 异构 MPI 图与资格矩阵

§3.5/§5.9、Figure 9 与 S17/Table S1 说明按 rank 选择 CPU/Metal/CUDA 的实现与实际验收边界；区分 CPU+Metal 仿真、未验收的 CUDA 仿真卸载和单 L4 CUDA MPI 训练。未将它们合并为任意厂商/跨平台集群已验证或速度优势。

§2.4 与 S18/Supplementary Figure S1 使用用户指定的线上 FlyWire DM1 保存运行结果，并配合模型页面的 B2IR 草稿。三面板展示实际活动回放、逻辑执行周期及突触投递操作；明确默认参数草稿与历史执行产物的区别。截图与模板哈希保留，补充图直接合并到默认论文 PDF。

默认合并 PDF 包含正文、兼容性附录、S21 条件资源分析和三页 Supplementary Figure S1；当前页数与页面检查记录见 `validation/full_scale_resource_pdf_qa.json`。

## 连接网络容量实验

§4.8/§5.10、Figure 10 与 S20/Table S3 纳入已验收的五档弱扩展：3/6/12/24/30 台机器、每台八个物理工作核，对应 8600 万至 8.6 亿神经元、860 亿至 8600 亿连接，均为 100 ms 模型时间。每档只有一次计时观测。数值小模型、完整输出、资源退出与清理证据见 [capacity/weak-evidence.json](data/capacity/weak-evidence.json)。此前固定 30 台的 86M/128M 与 256M 容量观测保留在 S19，不并入这条弱扩展曲线。比例仅为相对 86B 参考神经元数量的归一化，不代表人脑区域复现、最大容量或全脑可行性验证。

## 全参考计数的条件资源分析

S21/Table S4 与 Supplementary Figure S2 按冻结源码建立可审计的数组、通信、记录和投影报告模型。86B 神经元、平均入度 1000 对应 86 万亿连接；保持本地工作量的条件分区映射是 3000 台、24000 个物理工作核。该映射不代表当前版本实际所需机器数或已验证可行，且不外推运行时间。生成器已经按 signed-i32 上限切分脉冲批次；约 641 GiB/进程对应全局源 CSR，而非全规模单个接收缓冲区。详细条件与瓶颈见 [full_scale/analysis.json](data/full_scale/analysis.json)，重建命令为 `python scripts/analyze_full_scale_resources.py`。分析不启动仿真或远程作业；S21 与图表直接合并到默认论文 PDF。

## 统一源码迁移基准与树突复测

最新next-dev（含未提交改动）的前端、CPU/GPU/MPI执行代码、训练、WASM、模型驱动与测试已按实际字节冻结；共1,695个文件，身份为`cc67a82bfeed8ce850c264b4ace7924b9b56be116b4132bb3e4eed9e2b020112`。源码与归档身份见[data/common_source/freeze.json](data/common_source/freeze.json)，逐项迁移状态见[data/common_source/migration.json](data/common_source/migration.json)。新测量必须绑定该源码身份，并另行记录输入、编译器、数值模式、设备与数值门槛。

历史结果保留自己的源码身份。此次迁移先覆盖图3完整拓扑100 ms的重新导出、配对计时和缓存消融；其余CPU、GPU、MPI、WASM及训练实验逐项迁移，不能直接改标签视为统一版本已验收。执行核心加入节点内端点指数缓存，保持原节点与边累加顺序；合法性、内存载荷和生命周期纳入物理计划。

最终正文表5报告默认实现的结果；同源码关闭缓存的消融对照见S13表S5。已被当前默认实现替代的旧版性能记录保留于开发实验档案，不作为最终论文的性能结论。S13／表S5同步翻译并合并进中英文默认PDF。
