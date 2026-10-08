# 精神疾病仿真研究的五年文献筛查与选题判断

## 结论

在不自行招募患者、不采集新 EEG 的条件下，脑回路仿真仍有产生重要成果的机会。近年的成功研究已经包括：从分子变化推导学习异常并用独立遗传—脑电数据检验；从回路仿真提出新的神经编码预测，再用既有动物记录验证；建立能把受体变化连接到全脑状态的方法。由此，研究成果可以是疾病机制、具有疾病意义的计算原理，或经过真实数据验证的通用方法，不必预先限定为发现一种新的病理实体。[^12][^21][^28][^36]

现阶段最合理的决策是保留三个候选进行小规模预研究：

| 候选 | 核心问题 | 适配优势 | 最关键的未解决条件 | 建议 |
| --- | --- | --- | --- | --- |
| 幻听中的预测信号失配 | 是预测内容错误、到达时间错误，还是学习过程把错误预测固定下来？ | 已核实患者分段 EEG 文件；适合有传导延迟、内容选择性和可塑性的回路 | 现有任务是否足以区分这些解释；独立验证队列仍缺 | 原始 EEG 范围内，优先启动可区分性预研究 |
| 精神分裂症中的学习修复边界 | 恢复神经活动幅度，何时仍不能恢复正确学习？ | 分子可塑性模型公开；适合大量机制组合、长学习序列和跨尺度比较 | 原始模型并非 Brian2；独立遗传—可塑性 EEG 尚未取得 | 若合作方具备分子/可塑性专业能力，优先作为机制主线考察 |
| 抑郁症中干预方向的条件 | 增强抑制与暂时解除抑制，何时分别有利于信息处理和学习？ | 有不同受体干预的实验依据；可做状态、靶点、时间的系统比较 | 需扩大到既有动物/细胞数据；不能仅靠普通静息 EEG 验证 | 科学问题值得深入，但立项前必须证明超出已有局部模型 |

显著性网络扩张、抑郁生物分型、DBS 恢复轨迹、分布式价值编码均值得保留为储备。它们暂时没有同时满足“充分新增量、与执行器相配、验证数据可用”三个条件。建立通用疾病平台可以作为研究产物，尚不足以单独支撑疾病领域的重要发现。

以下排序是选题判断，不是文献已经证明的新发现，也不是期刊录用概率。Nature Neuroscience、Nature Medicine、Nature Communications、Nature Computational Science 和 Molecular Psychiatry 的定位不同，不能用一个“Nature 级”标签替代对贡献和证据的判断。

## 范围与筛查方法

时间窗口为 **2021-09-12 至 2026-09-12**，以首次正式在线发表日期优先；在线年与卷期年不同时单独注明。检索围绕 depression/MDD、schizophrenia/psychosis、biophysical/spiking/computational model、EEG/MEG、plasticity、NMDA/GABA、hallucination、reward、brain network 与公开代码/数据交叉展开。先定位期刊原始文章及 PubMed/PMC 记录，再沿引用和相关论文扩展，并针对入围方向检查近邻竞争研究及数据声明。

文献表整理 **36 篇窗口内的研究、方法或数据论文**；另列 3 项预印本/观点材料以及窗口外的重要先行工作。此数字指实际列入比较表的独立论文，不代表搜索引擎总命中数，也不代表穷尽了五年全部文献。它是面向选题的结构化范围筛查，没有实施独立双人筛选、预注册系统综述或荟萃分析。

优先标准包括：问题是否重要；最近论文已经做到了哪一步；新增模拟能否提出与现有解释不同的可检验预测；数据是否足够约束模型；Brian2-Rust 的能力是否对问题有实际帮助。单纯诊断分类准确率、常规频谱组间差异、无实验约束的巨大网络，不作为优先方向。

数据状态分四层：**文件已核实、论文声明开放、需要申请/联系、未核实**。论文开放获取不等于原始数据开放；绘图源数据不等于连续 EEG；公开代码不等于可直接在当前执行器运行。本次只检查文献、代码说明和仓库元数据，没有读取患者原始信号、重做统计分析或运行疾病模型。

## 文献已经完成了什么

### 已有成果超出了单个模拟例子

Pathak 等的皮层—纹状体模型同时联系了神经活动、学习和行为，提出可预测错误行为的神经元编码特征，并在已有猕猴记录中检验。模型和预处理后的脉冲/LFP 数据有公开入口。论文首次在线发表于 2025-12-29，卷期为 2026 年。这是“模型提出新现象，既有实验数据检验”的直接参照；它不是精神疾病疗效研究。[^28]

Mäki-Marttunen 等把突触内生化过程与精神分裂症相关表达变化结合，随后在 286 名健康对照的遗传—视觉诱发电位数据中检验相关预测。这里的 EEG 验证样本是健康人，不是 286 名患者；表达差异与风险相关也不能直接等同于疾病因果链已经证明。[^12]

Sacha 等已经建立受体、点神经元、平均场与全脑活动之间的连接，并以麻醉相关现象验证。由此，“把 NMDA/GABA 参数接入全脑仿真”已有很近的先例。Wang 等 2026 年预测加工研究则明确没有采集新实验数据，说明理论与既有证据也能形成高水平成果。[^21][^36]

### 若干看起来新颖的题目已有直接竞争者

| 初步构想 | 最近先行工作 | 必须新增的内容 |
| --- | --- | --- |
| 降低 SST 抑制，模拟抑郁 EEG，再用药恢复 | 2022–2024 年已有完整研究链 | 改变既有解释的适用边界，或获得新的独立干预预测 |
| 用 EEG 为虚拟患者估计 α5 药物剂量 | 2024 年已有模拟剂量预测 | 真实药理验证、模型错误下的可靠性，不能只扩大虚拟样本 |
| 用 MMN/ASSR 比较精神分裂症机制 | 已有多范式 DCM；2026 年已有 Brian2 多协议 MMN 模型 | 实际区分以前混淆的解释，而非更多参数扫描 |
| 用脑网络把抑郁分成若干类型 | 2024 年六类回路分型；2025 年连续因素模型 | 类型是否对应不同且可验证的回路反应 |
| 将抑郁解释为状态被困住 | 2026 年已有能量景观研究 | 新的转变规律及真实时间预测；“困住”比喻本身不够 |
| 建立大规模脑仿真平台 | 2024 年已有极大规模数字脑；2025 年已有受体到全脑方法 | 一种新的计算能力确实解决过去无法回答的问题 |

这些判断分别依据文献 2、4、9、10、14、19、21、31、34，以及窗口外 Adams 等多范式工作。[^2][^4][^9][^10][^14][^19][^21][^31][^34][^40]

## 候选一：幻听中的预测信号失配

### 通俗问题

人在准备说话时，大脑会提前影响听觉系统。可以把它理解成两种通知：“接下来的声音可能来自自己”，以及“具体将会是什么声音”。如果通知内容错了，或到达时间不对，听觉反应可能看起来相似。长期学习还可能把暂时的预测错误变成稳定偏差。

值得研究的问题是：**现有 EEG 能否区分内容、时间和学习三种故障，并找到它们各自失败的条件？** 这比简单问“幻听是否因为抑制不足”更具体。但三个故障可能共存，不能强行规定互斥类别。

### 文献基础与竞争边界

Yang 等 2024 年已用延迟发音 EEG 和两层网络解释不同患者的抑制及内容选择性异常。研究包括 20 名有幻听患者、20 名无幻听患者；患者在用药，健康比较使用既有任务数据。其两参数模型是必须比较的基线，不能把“抑制失灵加内容预测不精确”重新当成独创。[^11]

Okimura 等 2023 年已提出预测信号延迟解释异常主体感，不过任务和模型层次不同。2025 年皮层—纹状体学习预印本把局部多巴胺与感觉预期学习相连，亦已有行为和动物信号检验；本次未确认其正式期刊版本。跨论文连接本身不是新发现，必须形成新增预测。[^7][^37]

### 可开展的预研究

先复现已发表的小模型，再建立尽量匹配自由度的候选：内容选择性改变、传导延迟/抖动改变、学习历史改变。使用具有内容选择性的运动—听觉回路；只有当跨试次历史确实能提供额外信息时才加入学习模块，不应一开始堆入完整纹状体。

校准使用一部分已有条件，预先锁定其余条件和整段反应时程。比较指标应涵盖响应潜伏期、时间宽度、准备内容一致/不一致差异，以及实际记录中可恢复的试次历史。不能仅让更大模型拟合更多数据后比较误差；要约束复杂度，并在真实噪声水平下做机制恢复测试。

假设层面的突破是：被既有模型归为同一种异常的 EEG，实际上在某些条件下必须由不同动力学解释；这些解释对未参与拟合的真实任务响应给出分离预测。若进一步发现一个共同的时间关系能够统一解释多种现象，贡献将强于单个队列的再拟合。当前尚未证明这些数据具有这种区分力。

### 已核实的数据与硬限制

OSF `rsnu4` 的公开 API 显示，两组患者各有 **20 个个人 `-epo.fif` 文件及 1 个汇总 `.h5` 文件**，另有反应时、人口学和诱发电位结果目录。健康组目录目前核实到的是人口学及 N1/P2 汇总表，未核实到健康人的逐试次 EEG。分段 FIF 是预处理后的 epochs，不能称为完整连续原始记录。事件编码、分段范围、伪迹处理及是否仍保留完整试次顺序尚未读取核查。[^11][^41]

因此可以启动患者两组之间的模型比较，但不能声称已经具备完整三组原始 EEG 或独立复制队列。对已知论文的条件留出属于新的验证设计，不是事后变成真正未知的新实验结果；最终需要另一套独立数据或更一般的理论结果支撑更高目标。

**停止条件：** 若实际事件和时间分辨率不能区分内容失配与时间失配；若只有重新自由调参才能解释留出条件；若增加脉冲细节未产生小模型没有的新预测，则不以高影响疾病机制主线继续投入。

## 候选二：精神分裂症中的学习修复边界

### 通俗问题

一个回路可能“能响应声音、发放率也恢复正常”，却依然不能正确学习。类似设备恢复供电，但保存信息的过程仍然出错。研究目标是查清：**什么情况下恢复活动就能恢复学习，什么情况下必须直接修复学习规则？**

### 已有基础与需要跨过的门槛

2024 年 PNAS 已模拟 LTP、LTD、STDP 和视觉刺激协议，也做过多种分子参数变化。因此“把风险基因放进可塑性模型”“扫描刺激频率”“解释 VEP 可塑性下降”均已有先例。其公开仓库包含突触生化模型、详细 L2/3 锥体细胞和遗传分析，原始实现依赖 Python/NEURON。[^12][^42]

可考察的新增层次是：把这些受约束的突触变化放入有竞争表征和反馈的网络，研究局部修复是否足以恢复网络学习。区分至少两类恢复操作：调整兴奋性，使基线活动与诱发幅度恢复；调整可塑性过程，使经验能正确改变连接。两者有时可能等效，不能预设必然分离。

CogLinks 已把皮层—纹状体学习和丘脑—皮层控制结合，并讨论精神分裂症推理偏差。新的工作必须说明分子约束带来了什么原模型不能预测的差异，例如对刺激历史的特异性依赖，而不能仅重复“错误学习导致异常行为”。[^27]

### 具体突破尝试

以原始生化模型作为参考，构造在既有实验范围内保持可塑性预测的简化实现，再连接较大回路。先寻找“活动已恢复而学习未恢复”的参数区域，并用替代模型检验是否依赖某一个任意假设。随后从中推导能在真实记录里观察的特征，而不是只报告内部突触变量。

一个有价值的结果可能是：某一类受到遗传证据约束的缺陷，在普通诱发反应正常时仍造成特定学习历史效应；这个效应在独立数据中存在，且既有单突触/纯增益模型预测错误。此处所有具体方向都应由模型推出，不应先选定“必然失败”的结果。

计算规模的作用主要在于多机制、多随机网络、多学习历史及不确定性验证。是否需要更多神经元，取决于网络竞争、稀疏表征或传播过程是否改变结论。仅把相同突触复制一百万次，不构成新增科学内容。

### 数据与实施门槛

公开代码能够支持参考模型研究，但不能据此认定患者表达、个人基因型及 EEG 都可直接下载。本次未取得独立遗传—可塑性 EEG，也没有核实到满足该联合验证要求的匿名公开队列。普通静息 EEG 和 MMN 数据不能自动替代专门的长期可塑性协议。

当前执行器尚不能直接承接原仓库所有详细树突和生化求解。需要先保留 NEURON 参考，再核对方程刚性、求解误差、钙信号和简化后的响应。可将跨尺度简化作为方法贡献的一部分，但不能假定这一步已经成立。

**停止条件：** 没有独立可塑性数据；新现象完全由已知单突触曲线解释；简化模型无法保留干预预测；或所谓修复边界只在狭窄的人工参数区域出现。

## 候选三：抑郁症中干预方向的条件

### 通俗问题

抑郁相关研究中，有工作尝试增强特定抑制，也有工作通过暂时解除某些抑制来恢复学习。重要问题不是简单判定“应该让脑更兴奋还是更安静”，而是：**作用于哪类细胞、哪个位置、持续多久，会恢复哪一种功能？**

2024 年人体皮层微回路研究用 α5-GABA 受体调节增强抑制，恢复模拟中的信号检测。2025 年 GluN2D 研究则在小鼠海马及重复应激模型中，通过针对中间神经元的干预改善可塑性。两者的脑区、物种、初始状态和终点不同，不能把它们直接写成互相推翻的结果。[^9][^25]

### 可争取的新增量

将“抑制”拆分为有实验依据的具体作用位置和时间过程，比较背景噪声抑制、输入选择和学习窗口。使用不同局部回路作为模型集合，研究是否存在一个可测的状态边界，决定同一类调节在何时改善信息处理、何时损害学习。

成立的贡献应是一个能够预测干预方向变化的定量关系，并在未用于校准的真实实验条件中得到支持。仅把两个原有模型放在一个界面，分别复现两篇论文，不足以回答这个问题；仅说“不同脑区不同作用”也没有足够新增量。

现实意义在于约束候选靶点的使用条件，减少把任何“正常化的 EEG”误当作功能恢复的风险。它最多首先提供前临床机制解释和待验证干预假说，不能换算成患者治疗建议。

### 数据实际能支持什么

GluN2D 论文给出 Figshare 数据入口。本次 API 核查到 **40 个文件，共约 5.36 MB**，主要为 GraphPad Prism 文件和一份 Excel，包含绘图统计数据；不能称为完整原始膜片钳或动物连续记录。可以据此拟合和检验部分剂量—响应及可塑性曲线，逐试次神经动力学的验证仍有限。[^25][^43]

另一个有明确实验分歧的储备切口是 LHb 爆发放电。2023 年 Nature 研究显示 ketamine 的受体滞留与持续效应有关，并已操纵活动时间改变效应持续；因此“药物加刺激时机”不是空白。2026 年研究发现其条件下阻断 NMDAR 不消除爆发，但使用较短应激、不同膜电位范围，且没有直接测试 ketamine。可研究状态依赖的爆发机制边界，不能据此宣称前者错误。[^5][^30]

此方向需要接受既有动物和细胞数据为主要验证来源。头皮 EEG 难以直接识别深部 LHb 的离子机制，普通公开 EEG 不能包办此验证。

**停止条件：** 模型只是通过为不同论文指定不同初始状态来解释一切；无法预先预测留出条件；没有超出已有剂量曲线或树突抑制理论的关系；目标必须完全依靠新实验而现有合作条件无法提供。

## 储备方向与暂缓理由

### 显著性网络拓扑与抑郁易感性

2024 年 Nature 将比较稳定的功能网络范围与随症状变化的连接状态区分开，为“为什么有些人的网络更容易进入某种状态”提供依据。可以用空间网络模型研究范围变化、连接变化和局部增益变化能否产生不同响应。[^13]

但功能分区扩大不等于增加了同等比例的神经元，也不是已经证实物理连线被重新分配。该论文关键患者密集纵向记录在发表时未公开；健康公开参考数据不能替代患者验证。本次没有核实到这些患者记录后来全部释放。需要扩展到 MRI，并取得足够长的个体记录。故暂不作为可立即兑现的主线。

### 快感缺失与分布式价值编码

2025 年观点预印本已经提出“抑郁是分布式价值编码障碍”；同年 Nature Communications 已连接基础多巴胺与价值学习偏差。将“悲观神经元多、乐观神经元少”写进 SNN，本身不再是充分创新。[^22][^38]

比较有价值的是区分奖励感受、更新学习、行为表达三个阶段的故障，推导在相同平均奖励反应下不同的选择分布。已有临床 Reward Positivity MEG 研究可以提供约束，但 MEG 需要明确扩展范围，而且不同队列的行为和脑信号不能拼接成同一个人的完整病理链。[^23]

### DBS 的快速作用与缓慢恢复

2023 年 Nature 提供了随恢复变化的颅内信号，并声明 DABI 数据开放。全队列 10 人，但慢性 LFP 分析可用 6 人；信号在短暂停止刺激时记录，不能直接当作持续 DBS 同时的无伪迹信号。其公开程度目前核到论文与入口，尚未核到完整文件及协议。[^6]

可问快速动力学和慢可塑性怎样共同决定恢复，但六人的观测数据不足以直接证明所有慢性恢复机制。将仿真中的秒级时间任意换成数周病程没有依据。Scangos 等的 2021 年闭环病例则说明临床转化可行，却不提供普适的患者规律。[^1]

### 药物响应与模型简化

EMBARC 的随机治疗及重复 EEG 很有价值，但 2026 年已有 beta 短暂事件预测舍曲林反应，2020 年已有 EEG 治疗特异性预测。新增模型必须改善机制区分或真正的治疗相关预测，不能只换一种特征。[^33][^44]

如果最后发现不同尺度模型对同一真实药理操作给出相反预测，可以考虑独立的方法课题：哪些简化保留干预结果，哪些简化会改变结论。这与执行器比较贴近，但 2024 年参数推断、2025 年多尺度平均场已有先行方法。需要通用准则、多个模型及真实干预数据，单个模型的速度比较不足以支撑高目标。[^17][^21]

## Brian2-Rust 的实际结合点

本地 README 记录完整 PD14 网络通过普通 Brian2 Network 路径运行，规模为 77,169 个神经元、298,880,968 条突触。此处引用已有工程记录，没有重新运行性能测试。导出器当前对象白名单包含 NeuronGroup 等对象，未包含 SpatialNeuron；复杂 NEURON/HNN 树突模型不能直接作为兼容代码导入。[^45]

| 能力与任务 | 科学用途 | 不能据此推出 |
| --- | --- | --- |
| 点神经元、受体电导、延迟及多群体网络 | 内容选择、传导失配、局部回路的竞争解释 | 已经具备完整的人脑或疾病模型 |
| 多后端与参数化执行 | 增加不同机制、连接结构和随机种子的验证覆盖 | 在所有新模型上都加速同一倍数 |
| 可塑性及较长运行的工程基础 | 比较活动恢复与学习恢复，研究试次历史 | 可以不经验证模拟数月临床病程 |
| Rust 参考实现与不同规模模型对照 | 检查数值、简化或规模改变是否影响科学结论 | 更大规模天然比平均场更真实 |

大规模的价值需要通过一个具体问题证明：删掉网络传播、异质性、稀疏竞争或学习历史后，是否会失去关键预测？如果答案是否，应选择更简单的模型。节省的计算资源仍可用于更全面地排除竞争解释。

观测模型是所有 EEG 方向的共同门槛。平均发放率、脉冲计数、膜电位、局部场电位和头皮 EEG 是不同的量。用于定量脑电比较时，应明确电流源、空间混合及滤波；若只是响应代理指标，须限定结论，不把其单位和形状当作真实 EEG。[^16]

## 立项前的最小判据

建议先给一个候选安排约四至六周的可行性阶段；这是工作规划，不包含数据审批时间，不是已承诺能够完成疾病发现的周期。

1. **数据核查：** 文件、事件、试次数、症状、用药、重复测量和留出条件的实际交集。没有这些，样本总人数没有决定意义。
2. **参考复现：** 先复现最接近论文的核心结果及其简单模型。确定新增模型并非依靠额外自由度获胜。
3. **区分实验：** 在合成数据中加入真实噪声与观测混合，检查既有协议能否区分候选解释。不能区分时，先缩减问题。
4. **冻结预测：** 在查看保留部分之前锁定目标、允许变化的参数、对照与统计规则。已发表结果可以用于后验验证设计，但不可标成真正前瞻预测。
5. **继续门槛：** 存在重要且新的预测差异；真实数据能裁决；结果不依赖单个脆弱模型；执行器支持的计算量对结论有实质帮助。

严格保持 EEG 范围时，优先做幻听候选的数据和区分能力核查。若合作方具备突触机制研究能力并能获得已有可塑性数据，学习修复边界更适合发展成机制主线。若允许动物/细胞公开数据，干预方向的条件值得竞争性预研究。平台建设应围绕胜出的科学问题积累模型与验证接口，不先以覆盖两个疾病为交付范围。

## 36 篇文献比较表

“适配”表示可供研究借鉴或构建对照，不表示直接兼容当前执行器。下表日期在能核实到日时记日，否则只列正式发表年。标题有少量缩写，完整出处见文末。

| 编号 | 年份、期刊与论文 | 已完成的部分 | 资源/适配与选题裁决 |
| --- | --- | --- | --- |
| 1 | 2021，Nature Medicine：Closed-loop neuromodulation…[^1] | 单个难治性抑郁患者闭环干预 | 有源数据/代码；原始神经信号按需申请；转化参照 |
| 2 | 2022，Cell Reports：Reduced inhibition in depression…[^2] | 详细人皮层模型中 SST 抑制减少与信号检测 | 代码有入口；NEURON 类模型；重复题应排除 |
| 3 | 2022，Scientific Data：TDBRAIN[^3] | 跨诊断临床 EEG 数据资源 | 需遵守访问和使用条件；不是随机药理验证 |
| 4 | 2023，PLOS Computational Biology：In-silico EEG biomarkers…[^4] | SST 改变对应模拟 EEG 特征 | 疾病 EEG 正向模型的直接先行工作 |
| 5 | 2023-10-18，Nature：NMDAR trapping in the LHb[^5] | 药物滞留、活动依赖及持续效应 | 动物实验；源数据，非已核实完整连续记录；状态依赖储备 |
| 6 | 2023，Nature：Cingulate dynamics track depression recovery…[^6] | DBS 恢复的纵向颅内指标 | 声明 DABI 开放；可用慢性 LFP 6 人；扩展数据模态 |
| 7 | 2023-10-16，Schizophrenia：Aberrant sense of agency…[^7] | 预测延迟与主体感异常模型 | 数据/训练代码按需申请；幻听候选竞争解释 |
| 8 | 2023，Translational Psychiatry：Functional connectivity signatures of NMDAR dysfunction…[^8] | 药理影像与遗传影像联系 | 统计多模态约束；不是已完成的细胞因果验证 |
| 9 | 2024-02-23，Communications Biology：In-silico testing of new pharmacology…[^9] | 人体细胞约束的 α5-PAM 模拟恢复 | 有药理约束；详细形态迁移有门槛 |
| 10 | 2024，PLOS Computational Biology：Therapeutic dose prediction…[^10] | 虚拟疾病程度到剂量预测 | 模拟剂量不等于患者剂量；排除同类重复 |
| 11 | 2024-10-03，PLOS Biology：Impaired motor-to-sensory transformation…[^11] | 患者任务 EEG 与两参数模型 | 核实两组患者 epochs；优先预研究底座 |
| 12 | 2024-08-14，PNAS：Genetic mechanisms for impaired synaptic plasticity…[^12] | 分子可塑性、表达、独立健康人遗传—VEP 验证 | ModelDB 267741；详细生化/NEURON；机制候选底座 |
| 13 | 2024-09-04，Nature：Frontostriatal salience network expansion…[^13] | 稳定网络范围和变化的症状相关连接 | 关键患者记录开放尚未核实；MRI 储备 |
| 14 | 2024，Nature Medicine：Personalized brain circuit scores…[^14] | 抑郁/焦虑回路生物分型与治疗相关分析 | 重做聚类的增量弱；需机制验证才能推进 |
| 15 | 2024：40 Hz steady-state response…GABAergic inhibition[^15] | 健康人药理 MEG 约束 ASSR | 真实药理参照；不能把某剂量下无效等同受体无关 |
| 16 | 2024-02-19，Nature Communications：A neurophysiological basis for aperiodic EEG…[^16] | 多因素共同塑造频谱背景 | 模型/代码公开；禁止把单个斜率直接读成 E/I |
| 17 | 2024，PLOS Computational Biology：Methods and considerations…SBI[^17] | 详细神经模型参数估计及其困难 | 方法基线；“用 SBI 拟合脑电”已有先例 |
| 18 | 2024-12-19，Nature Computational Science：Simulation and assimilation of the digital human brain[^18] | 极大规模脑模拟及同化 | 单纯规模纪录竞争强；不作为当前选题卖点 |
| 19 | 2025-03-11，Nature Communications：MDD on a neuromorphic continuum[^19] | 结构异常以连续因素表示 | 队列按需申请；不能把 MRI 因素直接映射成细胞病因 |
| 20 | 2025，Science Advances：Large-scale maps of altered cortical dynamics…[^20] | 药理、受体空间图与早期精神病联系 | 原始健康药理 MEG 有公共入口；空间机制储备 |
| 21 | 2025-05-28，Nature Computational Science：Molecular mechanisms impact large-scale brain activity[^21] | 受体—点神经元—平均场—全脑 | 多尺度直接竞争；可作缩减模型基线 |
| 22 | 2025-08-13，Nature Communications：Tonic dopamine and biases in value learning…[^22] | 多巴胺与价值学习偏差模型 | 分布式价值题的近邻竞争研究 |
| 23 | 2025，Biological Psychiatry: CNNI：Hypoactivation of ventromedial frontal cortex…[^23] | MDD 奖励阳性相关 MEG | OpenNeuro ds005356；需要扩展到 MEG |
| 24 | 2025，Schizophrenia：AMP SCZ EEG protocol, reliability and stability[^24] | 多范式 EEG 可靠性及重复测量 | NDA 受控共享；协议论文不是疾病机制发现 |
| 25 | 2025-11-26，Nature Communications：GluN2D…rapid antidepressant action[^25] | 细胞靶向、药理和可塑性动物证据 | 40 个绘图/统计文件已核；干预方向候选依据 |
| 26 | 2025，Nature Mental Health：Stratified precision medicine…α2A agonism[^26] | 回路分层后的小样本开放标签治疗研究 | 非随机对照；证明分型转化已有人推进 |
| 27 | 2025-10-16，Nature Communications：Uncertainty processing in hierarchical decision making[^27] | CogLinks 学习/执行控制及精分相关推理 | 公开模型数据；认知桥接已有竞争者 |
| 28 | 在线 2025-12-29／卷期 2026，Nature Communications：Corticostriatal micro-assemblies discovers a neural code[^28] | 模型提出新编码并在既有猕猴数据验证 | Julia 模型与预处理 LFP/脉冲入口；重要成功参照 |
| 29 | 2026-01-03，Nature Communications：Distinct antidepressant therapies act on a common brain network[^29] | 多种治疗关联到共同网络并作进一步验证 | 原始 MRI 不开放；图谱/源数据可用；“共同治疗网络”已有先例 |
| 30 | 2026，Frontiers in Psychiatry：NMDARs are not necessary for LHb burst firing[^30] | 特定小鼠切片条件下的 NMDAR 阻断结果 | 与既有条件不同，未测试 ketamine；分歧储备 |
| 31 | 2026-03-20，European Journal of Neuroscience：Novelty detection in MMN…[^31] | 多协议 MMN 仿真与机制扫描 | ModelDB 2019882 主脚本 Brian2；直接迁移最接近 |
| 32 | 2026，Biological Psychiatry 接受稿：E/I balance and conversion to psychosis…[^32] | CHR 基线 MMN/P300 与后续转归模型 | 直接限制“损伤与代偿”创新空间；CHR 不等于精分 |
| 33 | 在线 2026-03-11，Journal of Psychiatric Research：Transient frontal spectral events…[^33] | EMBARC beta 事件与舍曲林反应 | 重复 beta 预测已不足；药理映射仍需独立约束 |
| 34 | 2026-04-23，Nature Communications：Spatiotemporal asymmetries…system entrapment[^34] | 抑郁严重度与状态景观联系 | fMRI/连接组；其控制能量不等于代谢能量 |
| 35 | 2026-03-03，Imaging Neuroscience：The Virtual Brain links TMS evoked potentials…in MDD[^35] | 用健康数据约束模型产生 MDD 相关预测 | 健康数据验证与真实患者验证须区分 |
| 36 | 2026-03-13，Nature Communications：Desegregation of neuronal predictive processing[^36] | 分布式预测加工理论及既有证据解释 | 声明无新实验数据；理论路线成功参照 |

预印本及观点材料另计：2025 年皮层—纹状体多巴胺/幻听研究；2025 年分布式价值编码抑郁观点；2026 年 ASSR 前端/后端模型。前者含原始研究，第二项为理论观点，第三项为计算研究，均不在上述正式论文数量内。未检索到正式版本不代表其一定尚未发表。[^37][^38][^39]

窗口外先行研究不能忽略：Adams 等虽然卷期为 2022 年，首次在线日期是 **2021-08-10**，位于严格五年窗口之前，因此不计入 36 篇；其多范式结果仍是重要竞争基线。Wu 等 2020 年 EEG 药物反应预测同理。[^40][^44]

## 来源

[^1]: Scangos et al. *Closed-loop neuromodulation in an individual with treatment-resistant depression*. Nature Medicine (2021). [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11219029/)，[DOI](https://doi.org/10.1038/s41591-021-01480-w)。
[^2]: Yao et al. *Reduced inhibition in depression impairs stimulus processing in human cortical microcircuits*. Cell Reports 38, 110232 (2022). [原文](https://www.sciencedirect.com/science/article/pii/S2211124721017411)。
[^3]: *The Two Decades Brainclinics Research Archive for Insights in Neurophysiology (TDBRAIN) database*. Scientific Data (2022). [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9198070/)。
[^4]: Mazza et al. *In-silico EEG biomarkers of reduced inhibition in human cortical microcircuits in depression*. PLOS Computational Biology (2023). [原文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010986)。
[^5]: Ma et al. *Sustained antidepressant effect of ketamine through NMDAR trapping in the LHb*. Nature 622, 802–809 (2023). [原文及更正入口](https://www.nature.com/articles/s41586-023-06624-1)。
[^6]: *Cingulate dynamics track depression recovery with deep brain stimulation*. Nature 622, 130–138 (2023). [原文](https://www.nature.com/articles/s41586-023-06541-3)，[DABI 入口](https://dabi.loni.usc.edu/dsi/1UH3NS103550/UXUF7822Z3JL)。
[^7]: Okimura et al. *Aberrant sense of agency induced by delayed prediction signals in schizophrenia: a computational modeling study*. Schizophrenia 9, 72 (2023). [原文](https://www.nature.com/articles/s41537-023-00403-7)。
[^8]: *Functional connectivity signatures of NMDAR dysfunction in schizophrenia—integrating findings from imaging genetics and pharmaco-fMRI*. Translational Psychiatry (2023). [原文](https://www.nature.com/articles/s41398-023-02344-2)。
[^9]: Guet-McCreight et al. *In-silico testing of new pharmacology for restoring inhibition and human cortical function in depression*. Communications Biology 7, 225 (2024). [原文](https://www.nature.com/articles/s42003-024-05907-1)。
[^10]: *Therapeutic dose prediction of α5-GABA receptor modulation from simulated EEG of depression severity*. PLOS Computational Biology (2024). [原文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012693)。
[^11]: Yang et al. *Impaired motor-to-sensory transformation mediates auditory hallucinations*. PLOS Biology 22, e3002836 (2024). [原文](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002836)。
[^12]: Mäki-Marttunen et al. *Genetic mechanisms for impaired synaptic plasticity in schizophrenia revealed by computational modeling*. PNAS 121, e2312511121 (2024). [原文](https://doi.org/10.1073/pnas.2312511121)，[出版信息](https://pubmed.ncbi.nlm.nih.gov/39141354/)。
[^13]: Lynch et al. *Frontostriatal salience network expansion in individuals in depression*. Nature 633, 624–633 (2024). [全文及数据声明](https://pmc.ncbi.nlm.nih.gov/articles/PMC11410656/)。
[^14]: Tozzi et al. *Personalized brain circuit scores identify clinically distinct biotypes in depression and anxiety*. Nature Medicine 30, 2076–2087 (2024). [原文](https://www.nature.com/articles/s41591-024-03057-9)。
[^15]: *40 Hz Steady-State Response in Human Auditory Cortex Is Shaped by Gabaergic Neuronal Inhibition* (2024). [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11170946/)。
[^16]: Brake et al. *A neurophysiological basis for aperiodic EEG and the background spectral trend*. Nature Communications 15, 1514 (2024). [原文](https://www.nature.com/articles/s41467-024-45922-8)。
[^17]: Tolley et al. *Methods and considerations for estimating parameters in biophysically detailed neural models with simulation based inference*. PLOS Computational Biology (2024). [原文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011108)。
[^18]: *Simulation and assimilation of the digital human brain*. Nature Computational Science (2024). [原文](https://www.nature.com/articles/s43588-024-00731-3)。
[^19]: Li et al. *Major depressive disorder on a neuromorphic continuum*. Nature Communications 16, 2405 (2025). [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11897166/)。
[^20]: Arazi et al. *Large-scale maps of altered cortical dynamics in early-stage psychosis are related to GABAergic and glutamatergic neurotransmission*. Science Advances (2025). [DOI](https://doi.org/10.1126/sciadv.ads0400)，[数据记录](https://www.fdr.uni-hamburg.de/record/17545)。
[^21]: Sacha, Tesler, Cofre & Destexhe. *A computational approach to evaluate how molecular mechanisms impact large-scale brain activity*. Nature Computational Science 5, 405–417 (2025). [原文](https://www.nature.com/articles/s43588-025-00796-8)。
[^22]: *Tonic dopamine and biases in value learning linked through a biologically inspired reinforcement learning model*. Nature Communications (2025). [原文](https://www.nature.com/articles/s41467-025-62280-1)。
[^23]: Pirrung et al. *Hypoactivation of ventromedial frontal cortex in major depressive disorder: an MEG study of the Reward Positivity*. Biological Psychiatry: Cognitive Neuroscience and Neuroimaging (2025). [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12084424/)，[数据说明](https://github.com/OpenNeuroDatasets/ds005356)。
[^24]: AMP SCZ EEG protocol reliability/stability paper. Schizophrenia (2025). [原文](https://www.nature.com/articles/s41537-025-00622-0)，[NDA Release 4](https://nda.nih.gov/study.html?id=3272)，[2026-09-03 手册记录](https://zenodo.org/records/22287306)。
[^25]: Vestring et al. *The NMDA receptor subunit GluN2D is a potential target for rapid antidepressant action*. Nature Communications 16, 10613 (2025). [原文](https://www.nature.com/articles/s41467-025-66774-w)。
[^26]: Hack et al. *A stratified precision medicine trial targeting α2A-adrenergic receptor agonism as a treatment for the cognitive biotype of depression*. Nature Mental Health (2025). [原文](https://www.nature.com/articles/s44220-025-00510-7)。
[^27]: Wang, Lynch & Halassa. *The neural basis for uncertainty processing in hierarchical decision making*. Nature Communications 16, 9096 (2025). [原文](https://www.nature.com/articles/s41467-025-63994-y)。
[^28]: Pathak et al. *Biomimetic model of corticostriatal micro-assemblies discovers a neural code*. Nature Communications 17, 390 (online 2025; volume 2026). [原文](https://www.nature.com/articles/s41467-025-67076-x)，[代码和数据入口](https://github.com/Neuroblox/Neuroblox.jl/blob/main/RESOURCES.md)。
[^29]: Ji et al. *Distinct antidepressant therapies act on a common brain network*. Nature Communications 17, 1176 (2026). [原文](https://www.nature.com/articles/s41467-025-67945-5)。
[^30]: *NMDA receptors are not necessary for burst firing of lateral habenula neurons in mice*. Frontiers in Psychiatry (2026). [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13294388/)。
[^31]: Eissa et al. *Computational Modelling of Novelty Detection in the Mismatch Negativity Protocols and Its Impairments in Schizophrenia*. European Journal of Neuroscience 63, e70453 (2026). [DOI](https://doi.org/10.1111/ejn.70453)，[ModelDB](https://modeldb.science/2019882)。
[^32]: Rodriguez-Sanchez et al. *Biophysical modeling of excitation/inhibition balance and conversion to psychosis in the clinical high risk syndrome*. Biological Psychiatry (2026; accepted manuscript). [机构记录](https://discovery.ucl.ac.uk/id/eprint/10225747/)，[DOI](https://doi.org/10.1016/j.biopsych.2026.04.007)。
[^33]: Waller, Carpenter & Jones. *Transient frontal spectral events from EEG predict antidepressant response to sertraline in depression*. Journal of Psychiatric Research (2026). [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13221100/)。
[^34]: *Spatiotemporal asymmetries on brain energy landscape uncover system entrapment related to depression severity*. Nature Communications (2026). [原文](https://www.nature.com/articles/s41467-026-71961-4)。
[^35]: Hofsähs et al. *The Virtual Brain links transcranial magnetic stimulation evoked potentials and inhibitory neurotransmitter changes in major depressive disorder*. Imaging Neuroscience (2026). [PubMed](https://pubmed.ncbi.nlm.nih.gov/41799679/)，[模型仓库](https://github.com/virtual-twin/TMS_MDD)。
[^36]: Wang et al. *Desegregation of neuronal predictive processing*. Nature Communications (2026). [原文](https://www.nature.com/articles/s41467-026-70347-w)。
[^37]: Lakshminarasimhan et al. *A corticostriatal learning mechanism linking excess striatal dopamine and auditory hallucinations*. bioRxiv preprint (2025). [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11956939/)，[标明预印本的 PubMed 记录](https://pubmed.ncbi.nlm.nih.gov/40166304/)。
[^38]: Botvinick et al. *Depression as a disorder of distributional coding*. Perspective preprint (2025). [arXiv](https://arxiv.org/abs/2507.16598)。
[^39]: Xia, Xu & Zhang. *Front-end and Back-end Computational Modeling of 40-Hz Auditory Steady-State Response Abnormalities in Schizophrenia*. Preprint (2026-08-29). [arXiv](https://arxiv.org/abs/2608.29104)。
[^40]: Adams et al. *Computational Modeling of EEG and fMRI Paradigms Indicates a Consistent Loss of Pyramidal Cell Synaptic Gain in Schizophrenia*. Biological Psychiatry (online 2021-08-10; issue 2022). [日期核验](https://pubmed.ncbi.nlm.nih.gov/34598786/)，[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8654393/)。
[^41]: [Yang et al. OSF 仓库](https://osf.io/rsnu4/)。2026-09-12 API 元数据核查：患者 epochs 每组 20 个 FIF 加 1 个 H5；本地审计附件位于同目录 `psychiatry_literature_audit_20260912/`。没有下载或执行患者数据文件。
[^42]: [ModelDB 267741](https://modeldb.science/267741)，[源代码 README](https://github.com/ModelDBRepository/267741)，[突触模型目录](https://github.com/ModelDBRepository/267741/tree/main/syn)。
[^43]: [GluN2D 图形源数据记录](https://doi.org/10.6084/m9.figshare.30436471)。API 清单保存为 `psychiatry_literature_audit_20260912/glun2d.json`；40 个文件，5,363,410 字节。元数据日期并非论文正式发表日期。
[^44]: Wu et al. EEG antidepressant-response prediction, Nature Biotechnology (2020), outside screening window. [原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7145761/)。
[^45]: 工程能力依据本地 [README](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/README.md:21) 与 [对象导出限制](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/python/brian2_rust/export.py:859)。工程文档记录不替代新疾病模型的数值与性能验证。
