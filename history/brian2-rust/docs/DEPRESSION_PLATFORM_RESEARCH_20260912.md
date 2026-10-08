# 抑郁症仿真研究现状与 Brian2-Rust 平台建议

检索与本地代码审阅日期：2026-09-12。本文是定向文献与可行性研究，不是系统综述。本次未运行抑郁症模型、未下载患者数据；本地性能数字引用既有报告，不代表此次复测。以下分别标注论文发现、仓库事实和拟议研究。

## 立项判断

建议建设面向研究人员的抑郁症机制与干预实验平台。近期交付应是可复现模型、机制对照、参数不确定性与外部验证工作流。临床个体疗效预测是需要独立证据的后续目标。

最适合当前执行器的入口是前额叶—扣带回点神经元网络。最有科学价值的后续问题是：多个机制产生相似静息信号时，哪些任务或扰动能将它们区分，以及干预效果能否跨参数、随机种子、模型结构和患者数据保持。

速度和内存优势可以扩大实验覆盖，但不能单独证明疾病机制或疗效。仿真恢复发放率、节律或信息处理能力，也不等于患者抑郁症状恢复。

## 已核实的研究路线

| 路线 | 代表性工作 | 已达到的程度 | 证据边界 |
| --- | --- | --- | --- |
| 双脑区网络 | Ramirez-Mahaluf et al., Cerebral Cortex 2017 | vACC–dlPFC 生物物理网络；用谷氨酸动力学异常解释切换受损，并模拟 SSRI、DBS 干预 | 对特定病理假说的模型检验，不是完整病因或个体处方模型 |
| 人类微回路与 EEG | Yao et al. 2022；Mazza et al., PLOS Computational Biology 2023 | 以人类细胞及突触数据约束模型，研究 SST 抑制下降，生成 EEG 并与 PV 抑制变化对照 | 机制到可观测信号的预测；不能由单个 EEG 特征直接确诊某细胞缺陷 |
| 计算药理 | Guet-McCreight et al., Communications Biology 2024 | 在人类微回路模型中模拟 α5-GABA-A 正向变构调节，研究功能与 EEG 恢复 | 药理机制的计算预测；模型强度不直接等于人体给药剂量 |
| 人类药物实验与模型反演 | Translational Psychiatry 2024，氯胺酮 EEG/DCM | 对 27 名纳入分析的 MDD 参与者，用皮层—丘脑模型解释给药相关连接与受体动力学变化 | 直接使用患者实验数据，但仍是有模型假设的参数推断 |
| 个体全脑与 DBS | An et al., NeuroImage 2022 | 用个体纤维束及高分辨率虚拟脑模拟刺激传播，并与患者诱发反应对照 | 原文报告需个体调参、刺激位置之间预测精度不同；不是已验证的普遍疗效推荐器 |
| 最新网络现象解释 | Cunha et al., Frontiers in Computational Neuroscience，2026-08-27 | 对照 30 名 MDD 与 28 名健康者 EEG，用 QIF-E 网络探索多尺度熵特征 | 作者明确限定为现象对应，不将局部电耦合认定为抑郁症病因 |
| 外侧缰核放电机制 | Fedorov et al., Journal of Physiology，2026-04-02 | 离体实验与计算模型研究 LHb 的多种簇发形态 | 与动物抑郁样行为相关的机制背景；不是人类疗效验证 |

来源：[2017 原论文](https://pubmed.ncbi.nlm.nih.gov/26514163/)、[2023 EEG 模型](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010986)、[2024 计算药理](https://www.nature.com/articles/s42003-024-05907-1)、[2024 患者 EEG/DCM](https://www.nature.com/articles/s41398-024-02738-w)、[2022 个体 DBS 模型](https://www.sciencedirect.com/science/article/pii/S1053811921011198)、[2026 QIF-E](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2026.1899952/full)、[2026 LHb](https://doi.org/10.1113/JP289617)。

另有 2025 年 PLOS 氯胺酮/PV/SST 建模工作及公开代码，但其 MEG 实验对象是 **12 名健康志愿者**，不能写成抑郁症患者疗效验证。代码公开，临床与 MEG 数据未公开。[论文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013118)、[代码](https://github.com/jessierademacher/Ketamine_E-I)。

## ModelDB 与 EBRAINS 的实际用法

### 模型资产

ModelDB 本次读取的 MDD 专题页列出两个条目；这个标签的数量不代表整个领域只有两个模型。搜索 depression 还会命中突触抑制/可塑性意义的 depression，需要与临床 MDD 区分。[MDD 专题](https://modeldb.science/ModelList/240955)

| 资产 | 原始执行环境 | 本项目处理建议 |
| --- | --- | --- |
| ModelDB 240954，vACC–dlPFC | 旧版 Brian，代码为 `from brian import *` | 第一优先；保留原始参考后迁移到 Brian2，再对照 Rust |
| ModelDB 267595，人类 L2/3 微回路 | NEURON + LFPy，多区室形态与离子通道 | 保留详细参考模型；独立研究降阶到点神经元后能保留哪些功能 |
| α5-PAM 2024 代码 | 详细人类微回路模型的计算药理扩展 | 第二阶段作为受体特异性扰动和观测标定参考 |

来源：[240954](https://modeldb.science/240954)、[原始 Python](https://github.com/ModelDBRepository/240954/blob/main/MDD_spiking_model.py)、[267595](https://modeldb.science/267595)、[α5-PAM 仓库](https://github.com/agmccrei/HumanL23Circuit_a5PAM_AGM2023)。

240954 源码设每区 800 个兴奋性和 200 个抑制性神经元，两个区共 2,000 个回路神经元，另有外部输入源。它含旧式 Connection/IdentityConnection，并通过 `network_operation(when='start')` 和矩阵乘法计算 NMDA 总输入。因此迁移不只是修改 import；必须验证求和、时间步顺序及输入实现。源码中的“轻中重度”参数是该模型的定义，不能直接映射为患者临床严重度。

原库提供代码能帮助复现，但运行环境、随机性、依赖、参考图和许可证仍需逐项记录。不能承诺下载任意模型就得到与论文逐点相同的虚拟脑区。

### EBRAINS 集成

EBRAINS 可提供图谱、结构与功能数据处理、TVB 全脑网络模拟，以及多尺度共仿真工具。其官方现成路线包括 TVB–NEST；不能据此假定已有 Brian2-Rust 即插即用适配。[全脑工具](https://ebrains.eu/data-tools-services/modelling-simulation/whole-brain-simulation)、[TVB–NEST](https://ebrains.eu/data-tools-services/tools/multi-scale-brain-simulation-with-tvb-nest)

拟议集成顺序：先使用数据/图谱约束区域及连接；再做离线微回路响应标定；最后考虑双向共仿真。最后一步要验证平均场到输入的转换、脉冲到群体活动的聚合、单位、延迟、耦合步长和数值稳定性。原官方工具也明确指出技术联通之后，生物物理合理性的耦合仍在发展。

EBRAINS 的 Virtual Brain Twin 项目公开展示的重点包括精神分裂症个体化建模，不能直接视为已经完成的抑郁症数字孪生产品。[2025 项目说明](https://ebrains.eu/news-and-events/events/2025/ebrains-at-the-brain-innovation-days-advancing-mental-health-and)

## 当前 Brian2-Rust 能提供什么

代码与文档审阅依据：[README](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/README.md)、[兼容矩阵](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/COMPATIBILITY.md)、[导出器](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/python/brian2_rust/export.py:859)。现有工作区有其他未提交开发，本报告未修改其实现。

可利用的能力包括多 NeuronGroup、conductance/HH 方程、延迟、随机输入、突触状态、部分可塑性、分段执行及受限 checkpoint，足以支持一个点神经元机制平台的初始版本。每个模型仍要用 capability report 和差分实验确认；CPU/AOT、CUDA、Metal 的支持范围与精度不能互相外推。

本地既有报告记录：16 GB M3 以 Rust 运行 77,169 个神经元、298,880,968 条突触的 PD14 网络，模拟 10 秒生物时间，主体约 104 秒，子进程峰值约 6.31 GB；该资源测量关闭记录。这是特定皮层基准的容量证据，不是前额叶疾病生物学验证，也不是详细树突模型的容量承诺。[16 GB 报告](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/PD14_LOCAL_16GB.md)

M1 Ultra 既有比较记录 Rust 主体约快 1.98 倍，并有更低内存占用；这是该模型、各自所测设置下的结果，不能承诺抑郁症模型同样加速。报告本身说明长运行每后端只有一次，统计检查也并非全部处于参考区间。[比较报告](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/PD14_COMPARISON.md)

关键缺口：

1. 导出器显式允许的对象类型不含 SpatialNeuron；NEURON/LFPy 详细树突回路不能直接交给当前 Rust 后端。
2. 原始双脑区模型的 Python 每步 callback 不在支持范围。需要以可验证的声明式求和/调度表达重写，或补齐必要能力。
3. EEG/MEG 前向观测模型尚不是已验证的本项目能力。群体发放率、平均膜电位和头皮 EEG 是不同物理量。
4. 大批量实验还需要参数/种子注册、失败恢复、资源估计、结果索引与统计工具。原生执行可复用不代表通用 ensemble 工作流已经完成。
5. 秒级网络动力学不能自动解释数周治疗过程。长期适应需单独校准可塑性、稳态调节及药代/药效时间尺度。

## 拟议平台：Depression Circuit Lab

主要用户是计算神经科学实验室、精神病学研究组及神经调控/药物机制团队。产品的基础交付是研究方案与可复现实验结果。

```text
论文模型 + 生理约束 + 获准使用的患者数据
                    ↓
模型注册：版本、物种、脑区、机制、许可证、参考结果
                    ↓
Brian2 建模 → Rust 批量执行
NEURON/LFPy 详细模型 → 参考与观测标定
                    ↓
统一观测：spikes / 突触电流 / 任务表现 / 经验证的 EEG
                    ↓
参数推断、机制比较、扰动设计、不确定性
                    ↓
独立验证 → 报告、可重复脚本、后续实验预测
```

平台中需要分别展示“原始论文复现”“新机制假说”“仅计算验证”“外部实验验证”的状态。可视化优先呈现参数相图、指标分布和证据链；3D 脑图用于定位及结果展示。

### 首个实验

拟议问题：**哪些机制能让 vACC–dlPFC 在情绪样与认知样输入之间切换失败，哪些扰动能恢复切换，同时保留任务信号处理？**

先只复现原模型的健康/异常动态。随后逐项引入谷氨酸衰减、抑制强度、区域连接、背景输入等候选参数；每一项必须有来源和合理范围，不能同时任意调节所有参数。

主要指标为状态驻留时间、切换成功率、干扰后恢复时间和输入检测表现。发放率、节律功率及同步性是辅助指标，不能用“整体静默”冒充功能恢复。反刍只作为待验证的心理现象关联，回路持续活动本身不是反刍测量。

第二模型加入 Pyr/SST/PV/VIP 细胞类型，检验 SST 下降、PV 下降及兴奋增益等不同机制是否在静息条件下近似，而在任务或扰动下分离。若简化模型丢失树突靶向抑制的关键特性，应退回详细参考模型，不能继续沿用原论文的细胞机制解释。

### 候选突破及验收

| 拟议贡献 | 成果验收 | 何时应停止或转向 |
| --- | --- | --- |
| 可复现的抑郁症模型基准 | 原实现→Brian2→Rust 的关键现象及统计分布复现，环境与成本公开 | 实现差异导致机制结论不稳定时先解决复现 |
| 多机制的辨别实验 | 用拟合之外的扰动/任务区分候选机制；跨种子和结构变体保持 | 静息 EEG 无法识别时应报告等价机制集合，增加观测 |
| 稳健干预搜索 | 改善主要功能指标，对参数后验的大部分样本有效，保留正常活动 | 仅单一参数点或单一随机种子有效不能升级为结论 |
| 人类数据的增量预测 | 受试者/站点隔离的外部验证优于临床特征、EEG 特征及简单动力学基线，并有不确定性评估 | 不优于简单基线时保留机制研究定位，停止疗效预测宣传 |

本次检索并不能证明这些方向从未有人做过。具体论文的创新性需要在确定问题后继续做近邻文献与预注册设计检查。

模拟数据训练的推断器要先做参数恢复与模型恢复检验，再做真实数据外推。不能把同一套假设生成的标签和信号之间的高准确率当作临床有效性。

药物扰动首先用受体作用或电导变化表达；临床剂量需要药代/药效约束。TMS/DBS 若先以简化输入表达，应标明假设；个体刺激预测还需电场或纤维募集模型。观察性疗效相关也不能直接回答哪种治疗对同一患者更好。

## 数据与分阶段交付

TDBRAIN 是现实候选：论文描述 1,274 人的异质数据库，并非全部为 MDD；包含 rTMS 相关样本和盲测机制。数据访问需要账户/协议，部分标签保留用于独立验证。药物使用记录及治疗混杂限制需随样本一起审查。[数据论文](https://www.nature.com/articles/s41597-022-01409-z)、[现行盲测入口](https://brainclinics.com/resources/tdbrain-dataset/introduction/tdbrain-challenge)

EMBARC 可作为抗抑郁药研究候选，NIMH Data Archive 已列出 EEG 及治疗信息。需另核当前访问要求、具体队列、重复测量和结局可用性，本次仅核实目录，未取得数据。[NDA 数据记录](https://nda.nih.gov/study.html?id=2742)

| 阶段 | 建议时间预算，非承诺 | 具体交付与放行条件 |
| --- | --- | --- |
| A：原模型复现 | 1–2 周 | 固定 240954 原始版本与环境，恢复关键图/指标；记录旧 Brian 依赖和参考输出 |
| B：Brian2/Rust 迁移 | 2–4 周 | 解决 NMDA callback/调度；受控输入差分、多种子分布、步长敏感性；实测性能 |
| C：机制平台 MVP | 4–8 周 | 参数集合、干预协议、结果注册、机制/参数恢复检验；第二模型开始加入 |
| D：人类数据试验 | 数据获批后 2–4 个月 | 锁定终点与划分，对照简单基线；有条件开展跨站点/独立队列验证 |

以上是小型工程团队有计算神经科学协作者时的粗估，阶段部分可重叠；详细微回路降阶失败、旧依赖恢复、患者数据获取可能延长工期。初版从约 2,000 个回路神经元起步，先测每次实验成本，再确定样本数量和算力，暂不依据 PD14 推算万次疾病仿真的完成时间。

首个可公开研究成果应包含：固定模型、复现命令、已验证与未验证指标、迁移差异、参数相图、正反例和一组可供实验检验的新预测。最终能否形成疾病研究突破，取决于这些预测能否得到独立数据支持。
