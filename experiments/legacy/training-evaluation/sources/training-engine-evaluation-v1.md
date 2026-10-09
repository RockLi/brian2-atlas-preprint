# Training 引擎评估 v1：冻结候选协议

日期：2026-10-04。状态：`freeze_candidate`。本文件规定首轮问题、工作负载、计量规则及决策条件；**没有执行 benchmark，没有锁定硬件、运行环境、数据或适配器，也没有产生性能排名。**

配套：[竞品分析](training-competitive-analysis-2026-10-04.md)、[机器可读计划](validation/training-engine-evaluation-v1.plan.json)。二者与本协议使用相同工作负载 ID。若参数存在冲突，应在执行冻结前修正并重新计算协议哈希；不得在看到性能结果后选择有利口径。

## 1. 评估目标与结果的解释范围

首轮回答四个问题：

1. 同数值模型、同学习规则、同精度和同硬件下，Atlas 完整训练步骤的执行成本如何？
2. 对真实事件任务，各框架采用合法的合适算法时，达到相同质量的时间、内存和成功率如何？
3. 更宽、更长、更稀疏或更多状态的模型，在哪个资源限制上失败？增加设备是否改善吞吐或容量？
4. 复用已有 Brian2 模型需要多少改写，原机制和训练后行为是否能保留？

本评估不合成一个全局冠军分数。速度、容量、训练质量、兼容范围和迁移投入分别呈现。只完成仿真不能记作训练完成；能执行 `backward` 不能替代收敛；增加总资源不能直接记作单设备效率提升。

## 2. 四条赛道与迁移面板

| 赛道 | 固定的内容 | 比较对象及定位 | 主要结果 |
| --- | --- | --- | --- |
| **E：同语义 SG-BPTT** | 前向方程、调度、替代梯度、reset 求导、损失、优化器、输入、初始参数、精度 | Atlas、snnTorch、SpikingJelly、Spyx、BrainPy state/brainstate；Norse 交叉参考；Sinabs+EXODUS 对应前馈 CUDA 子集，Lava-DL SLAYER 对应动力学子集 | 正确性合格后的完整 step 成本、内存、全 BPTT 容量 |
| **L：原 Brian 局部学习** | 原方程、事件顺序、突触规则、输入、连接、精度 | 原版 Brian2、Atlas；GPU 加 Brian2CUDA/GeNN、BindsNET，分布式加 NEST；各组合先过语义资格 | 原模型前向与局部权重更新的正确性、运行及构图成本 |
| **A：同任务的最佳合法方法** | 数据划分、可见信息、评价指标、预算、质量目标 | E 组及 mlGeNN e-prop/EventProp、BrainTrace；NEST e-prop 按适配情况纳入 | 时间到质量、成功率、资源成本；不同算法不称同梯度加速 |
| **F：能力边界与参数拟合** | 明确原模型和目标；复杂模型可按阶段仅做支持性核验 | Jaxley/braincell、brian2modelfitting、深度 SNN 栈及 Atlas | 真实支持边界；可比较子集上的时间到拟合误差 |
| **M1：迁移面板** | 事先分层抽取的公开 Brian 模型集合与完整分母 | 各适配路径 | 改写时间、代码改动、删减机制、语义/梯度/参数回灌验证 |

原版 Brian2 没有本轮需要的通用 SG-BPTT 接口，因此不以“不支持”当作其输掉 E 赛道。Brian2CUDA/Brian2GeNN 的仿真或 STDP 不能自动升级为监督训练支持。GeNN 底层、Brian2GeNN 适配层、mlGeNN 训练编译器分别记录。

竞品可以使用自己的兼容依赖环境，不要求所有框架安装相同 Torch/JAX 版本。稳定发行与前沿版本单独标识。文档没有确认的组合记作待资格核验，不先判定不支持。

### 2.1 强基线覆盖与专项对手

- **Spyx 必须参加核心 SG-BPTT 资格**，重点为普通 LIF、SHD 和长 T 的 JAX/XLA 整步编译路径。其 associative scan `PSU_LIF` 去掉 reset 等改变不等价于普通 reset LIF，应独立标模型；当前 NNX 与旧 Haiku 示例不能混版本。[官方仓库](https://github.com/kmheckel/spyx)、[并行神经元边界](https://spyx.readthedocs.io/en/latest/explanation/parallel-spiking-neurons/)
- **Sinabs+EXODUS 是兼容前馈 LIF/Conv CUDA cell 的必要专项对手**。不得仅运行普通 Sinabs 后宣称已比较最强路径；EXODUS 的 BPTT/surrogate 规则与当前 strict fixture 是否匹配先资格判断。[EXODUS](https://github.com/synsense/sinabs-exodus)
- **Lava-DL SLAYER** 纳入可学习动力学、ALIF、延迟等相关子集；不能从 Lava 多硬件进程推断 SLAYER 已有分布式反向训练。若方程不匹配 E4/E5，保留其能力证据并在 A/专项原生视图比较。[SLAYER](https://lava-nc.org/lava-lib-dl/slayer/slayer.html)
- **BindsNET** 作为局部 STDP/奖励可塑性候选，其规则需与 L1/L2 对齐，不将旧前向速度图当梯度训练结果。[官方规范](https://bindsnet-docs.readthedocs.io/models_spec.html)
- **Rockpool** 保留为参数约束学习与硬件部署链的外围专项；纯 LIF CUDA 先通过直接 EXODUS 控制重复评估成本。这不是其“不支持训练”的判断。[约束学习](https://rockpool.ai/in-depth/howto-constrained-opt.html)

全部直接对手先做适用小型资格，再决定完整扩展曲线覆盖；排除理由必须是模型、语义、硬件或明确预算，不能先删除潜在强对手再宣称领先。数据驻留 GPU 的 compute-only 与 streaming 数据加载分表，所有框架采用相同驻留条件。

## 3. 公共数学契约与资格入口 Q0

### 3.1 专用同步 LIF fixture

以下是用于严格比较的**专用同步调度候选**，不是所有库的默认 LIF，也不是所有 Brian2 模型的代表。必须先确认 Atlas 原生解析器和每个适配器执行此契约，再锁定；若无法实现，应修复适配器或重标为不同语义，不能放宽容差让它通过。

对每个 tick，所有层首先同时执行：

```text
u_l[t] = beta * v_l[t-1]
s_l[t] = 1 if u_l[t] > theta else 0
```

全部层完成 threshold 后，再进行突触输入与重置：

```text
feed_l[t] = x[t]                      # 第一层
feed_l[t] = s_(l-1)[t]                # 后续层
v_l[t] = u_l[t] + W_l * feed_l[t]
         + R_l * s_l[t]               # 仅循环 fixture 有 R
         - theta * stop_gradient(s_l[t])
```

循环连接按同 tick 的已计算 spikes 传播，其影响只能在后续 threshold 生效。不得将它悄悄改成先输入、后 threshold，也不得用已更新的前层状态改变下层同 tick 的 threshold。

| 项目 | 候选契约 |
| --- | --- |
| 时间步与衰减 | `dt = 1 ms`，`tau = 20 ms`，Euler 对应 `beta = 0.95`；不能替换为 `exp(-dt/tau)` |
| 阈值及初始状态 | `theta = 1`，严格 `>`，`v[0] = 0` |
| 替代梯度 | margin `m = u - theta`；`ds/du = g(m) = 1 / (1 + 5 * abs(m))^2` |
| reset 求导 | `theta * stop_gradient(s)`，不通过该 reset spike 路径求导 |
| 输出及损失 | `logits = 5 * mean_t(output_spikes)`；交叉熵、batch mean |
| 更新 | Adam：学习率 `0.001`，betas `(0.9, 0.999)`，epsilon `1e-8`；epsilon 位置、bias correction 与 step 计数纳入契约 |
| 其他 | 无 bias、无噪声、无 refractory、无 dropout、无梯度裁剪；特殊工作负载另列增量契约 |

无噪声规范只针对此 strict fixture；如果另开随机模型 fixture，必须预生成同一噪声数组，不能依赖各后端使用相同 seed 来获得相同随机数。

Norse 默认突触电流引入额外状态，不能直接将其包装成上述单状态模型。可以编写精确适配器参加 E，同时在原生优化视图中保留其自然模型；两个成绩不可混合。

### 3.2 Q0：进入性能比较前的最小资格

`Q0` 固定为输入—隐藏—输出 `2–4–2`、`T=32`、`B=2`。建立可逐元素检查的输入、权重、状态和标签，并加入独立的阈值边界、reset、循环及共享权重变体。全部公共数组作为制品保存并计算 SHA256；**共同数组才是同输入证据，共同 seed 不是。**

Q0 至少验证：前向所有状态和 spikes、loss、独立 VJP、所有参数梯度、单步 SGD、一个及多个 Adam 步的参数/一二阶矩、batch 缩放、样本之间不携带状态、状态 reset、负输入、延迟、循环时序、共享权重梯度累加、可学习动力学参数的链式求导。平台自动数据编码与预处理不参与这个入口。

独立 oracle 应直接实现数学契约与 VJP，不能调用 Atlas 的梯度实现，也不能把所有适配器共同依赖的一段反向代码当独立参考。有限差分只用于平滑子图及其参数变换；hard spike 的有限差分通常为零或不连续，**不得以它验证 surrogate gradient 是否正确**。

候选默认逐元素容差采用 `abs(a-b) <= atol + rtol*abs(reference)`：

| 数值情况 | state/loss/参数更新 `atol / rtol` | 梯度/VJP `atol / rtol` |
| --- | --- | --- |
| CPU FP64 | `1e-10 / 1e-8` | `1e-10 / 1e-8` |
| GPU FP32 相关路径 | `1e-5 / 1e-4` | `1e-4 / 1e-3` |

这些是待 Q0 资格运行冻结的候选值，不是已证明所有模型都可满足的误差界。量纲缩放与复杂模型需要独立声明。需要调整时必须在性能测试前解释数值原因、修订整个对应 profile 并留下历史，不能针对某个失败后端单独放宽。

阈值有足够 margin 的 case 要求 spikes 一致。接近边界及恰等于阈值的 case 单独报告比较符、数值误差和首次分歧，不能用平均 state 误差掩盖 spike 不一致，也不能把 near-boundary 容忍自动应用于所有 case。

## 4. E 赛道：由简单到复杂的固定工作负载

E 系列使用合成事件输入隔离计算规模；输入、连接、初始值、标签、活动分布和参数 mask 的生成脚本及数组哈希尚待冻结。一个主扫描一次只改变一个维度，禁止把更宽、更稀疏、更短的组合混作同规模。

| ID | 固定设计 | 扫描与必须披露的内容 |
| --- | --- | --- |
| **E1** | dense LIF 小：`[128,128,10]`、`T128 B16`；大：`[512,1024,1024,20]`、`T128 B32`。列表包含输入与输出宽度 | 先运行 Q0 的同语义前馈模型；两个尺寸均报告冷启动与稳态，不只选大尺寸 |
| **E2** | tied-weight conv：`1×28×28 → Conv16(k3,s2,p1) → Conv32(k3,s2,p1) → flatten → 128 → 10`；空间尺寸依次为 `14×14`、`7×7` | `T100 B16`；共享 kernel 参数只存一份，反向必须正确累加。展开连接与原生 conv 可各自优化，但标明逻辑连接数与独立参数数；不能把 untied 大矩阵视为等价 |
| **E3** | 循环网络：input128、hidden `N=128/512/2048`、output20，`B16 T512`；循环密度 `p=.01/.1/1` | 主锚点 `N512,p.1`；宽度扫描固定 p，密度扫描固定 N，其余组合作为预声明补充。每个图保存 edge list，不将 masked dense 宣传为稀疏存储 |
| **E4** | 两状态适应神经元，hidden `N256`、`B16`，`T512/2048`，input128、output20 | 同一双状态方程分为参数固定与参数可学习两栏；参数名称、约束变换、初值及 trainable mask 必须在执行前锁定。ALIF、AdEx 等不同适应方程不得互相替代 |
| **E5** | AdEx、Izhikevich 各自独立 fixture，`dt=.1 ms`；再加入一个明确写出方程的四状态 fixture | 具体网络形状/B/T、准确方程、单位、积分器、事件/reset、参数边界及四状态模型原件 SHA256 待锁。三者分别资格判定；四状态数量本身不能证明是 HH |
| **E6** | 循环长序列：input128、hidden512、output20，`p.1 B16` | `T=128/512/2048/8192`；全 BPTT 为主；checkpoint、TBPTT 单独分类；同时保存实际 spike count/发放率，防止时间长度变化混入活动变化 |
| **E7** | 循环宽度：input128、hidden `N=128/512/2048/8192`、output20，`p.1 B16 T512` | 固定 p 意味循环候选边数约按 N² 增长，必须报告实际边数；不能把这条宽度扫描称为线性负载扩展 |

E4 的具体双状态契约及 E5 的模型原件尚未完成，因此不得仅凭 `N/T/B` 齐全称它们为可执行冻结工作负载。E3 的完整补充网格与各 fixture 活动生成参数也须与 JSON 计划核对后封存。

### 4.1 两种实现视图

- **同语义适配器视图**：实现上述精确契约，通过相同 oracle。允许官方融合 kernel、编译和等价稀疏优化；不能为了代码统一强迫所有对手执行慢速手写 Python 时间循环。
- **原生优化视图**：每个框架采用其推荐实现、原生 neuron、编译/内存优化。若仍严格满足契约，可标明资格后加入同语义表；若改变 reset、更新顺序、积分器或梯度，作为独立模型/算法成绩，转入 A 或单列。

`torch.compile`、JAX JIT、SpikingJelly CuPy/Triton/memopt、稀疏 kernel、Atlas 原生 kernel 都记录具体开关和适用范围。前沿实现不能冒充稳定发行；未启用官方合理优化必须说明原因。

### 4.2 长序列与内存策略

完整 BPTT 与保持同一梯度的 activation checkpoint/recompute 可以同算法分组，分别报告重算成本。TBPTT 会改变梯度覆盖范围，单独列 `algorithm_variant`、截断长度及质量结果，不能以其吞吐或内存对比全 BPTT 后声称同算法优势。

当前 Atlas TBPTT 仍可能保留完整 tape；配置了截断不代表实现了内存压缩。实际释放时间、峰值内存和 tape 结构以锁定实现及测量为准。

## 5. L 赛道：局部学习及网络分区

| ID | 固定设计 | 核验重点 |
| --- | --- | --- |
| **L1** | STDP 事件 fixture；预设 pre/post spike 时间，包含前后相隔、同 tick、重复输入以及不同传输延迟 | 分别核验事件到达时刻、同 tick 的 pre/post 顺序、trace 更新与使用顺序、weight clipping、边界条件。固定权重增量 oracle；不得仅比较最终分布 |
| **L2** | sparse E/I，`N=1k/4k/16k`，E:I=80:20；固定 indegree100，`dt=.1 ms`，生物时间 `1s/10s` | 候选每个目标神经元 E 入度80、I入度20；edge list 冻结，是否允许自连接/多重边明确记录；主要规模负载按 N 增长。静态、同一 STDP 规则分表；发放率和通信事件数必须保存 |

L2 的 neuron/synapse 方程、哪些连接可塑、delay 分布和驱动输入须在执行冻结时确定；同时记录 fixed-indegree 实际边数。禁止用固定 p 导致的 N² 网络冒充固定 indegree 线性扩展。

原版 Brian2 至少包括 Cython runtime、C++ standalone 单线程与可用的 OpenMP 配置；NumPy fallback 可展示，但不作为唯一 CPU 对手。[官方执行后端](https://brian2.readthedocs.io/en/stable/user/computation.html)

跨框架“STDP”名称相同并不代表同规则。NEST、Brian、GeNN 的窗口、延迟、同 tick、nearest-neighbour/all-to-all 与积分语义必须对齐，否则转为不同规则的负载描述。Brian2GeNN 文档支持范围不能从现代原生 GeNN 功能推导。[适配限制](https://brian2genn.readthedocs.io/en/stable/introduction/exclusions.html)

固定发放率或 ignore-and-fire 可以作为附加的可塑性吞吐控制负载，必须标明不衡量任务学习，不能替换 L2 自洽网络行为，更不能替换 A 的训练质量。[NEST e-prop 扩展测试的相关边界](https://arxiv.org/html/2511.21674v1#S5.SS8)

## 6. A 赛道：真正训练到同一质量

| ID | 固定任务 | 数据与预算规则 |
| --- | --- | --- |
| **A1** | MNIST：`784→128→10`，`T100`，10 epochs | 官方 train 划为55k/5k，test10k；候选 B32。训练/验证/测试 ID、编码与像素归一化锁定；主要用于流程检查，不作为长时记忆证据 |
| **A2** | SHD：`700→256 recurrent→20`，`B32`，`dt1 ms` | T 由全数据不读取标签的最大事件时间确定，覆盖最后事件，不截断；train-only speaker-aware validation，50 epochs 与2小时运行上限在执行前固定 |
| **A3** | 与 A2 同一 SHD 数据、划分、观测时间窗和读出目标 | 比较 e-prop、EventProp、BrainTrace 等合法方法；使用相同隐藏容量作为主表，算法所需状态/连接差异披露。允许原生最佳实现，按质量/成本评价，不能记作 E 赛道的同梯度速度 |

### 6.1 SHD 编码及划分

使用原始 SHD 官方 train/test 边界。官方没有 validation；validation 仅从 train 划分，优先采用整 speaker 分组，保存准确 speaker ID、样本 ID 与每类计数，并说明样本量。具体留出 speaker 与划分算法在任何调参前固定；如果数据组成不允许预定分组，先修订划分协议再训练，不能转而使用 test 选模型。[数据出处](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)

`T = floor(max_event_time / dt) + 1`，以原始事件时间单位转换后的全数据最大值确定；扫描时不读取类别标签。此操作只确定统一输入形状，属于预声明无标签元数据使用。不得使用 test 事件分布选择超参数、归一化或性能目标。

时间 bin 采用左闭右开区间，保存每通道每 bin 的 **event count**。同通道同 bin 多次事件不能静默裁成一次。如果某实现只支持 binary spike，需采用保持事件计数贡献的确定性适配并通过 Q0 对应多事件测试；若无法保持原输入，就作为单独编码版本，不与 count-bin 主结果混排。

EventProp 使用原始时间戳与 binning 模型可能存在数值差异。A3 可分别保存“共同离散输入”和“原生事件输入”子表，明确时间精度，不把两者当完全相同前向。spike buffer 溢出必须报错或统计并判不合格；不能静默丢事件换取速度和内存。

### 6.2 调参、质量与停止条件

默认训练种子为 **11、23、37、51、71**。相同算法比较使用共同初始数组；不同算法无法共享全部状态时，保留可共享部分，并记录初始化规则和实际哈希。

每个框架/算法获得同等预声明的调参试验数与计算预算。A3 候选为每对手8个配置×2个调参种子101/103，共16个 run；每 run 最多10 epochs或1800秒、相同设备预算。按两种子的验证准确率平均选配置，失败 run 计0且保留错误；均分相同先选总调参 wall time 更短者，再以配置 ID 确定顺序，不能挑 best seed。完整搜索空间与实际资源仍待冻结，不以“各用默认值”代替公平调参。搜索仅使用 train/validation。正式五种子运行不追加调参，不丢弃失败种子。

质量门槛必须在正式性能运行前写入冻结清单。A2/A3 候选验证准确率为50%/70%/80%，分别报告达到每个门槛的时间；不能只挑 Atlas 有利的一条。可由独立 CPU 参考的资格/试运行核验门槛合理性，但一旦查看正式结果就不能移动门槛。若未指定或正式锁定质量门槛，仅可展示完整学习曲线与固定预算终点，不能发布“时间到同质量”比值。

A1 最多10 epochs；A2/A3 最多50 epochs或2小时，先到者停止。2小时主上限从完整训练调用开始、包含首次编译与训练所需加载/编码；冷启动准备时间同时独立计量。离线数据下载不计入训练调用，但环境/数据准备时间另报。长冷编译已耗尽预算的实现也要如实记录，不替对手删去编译成本；补充 warm-budget 表应对所有实现同口径。

验证频率统一为每个完整 epoch，保留各次 wall time。主时间到质量定义为首次达到预先冻结验证门槛的评价时间；稳定性另报随后连续评价是否维持门槛。测试集仅对按 validation 预声明规则选出的 checkpoint 做最终评价；不用于停止、超参搜索或挑选种子。

停止但未达质量的运行保留完整曲线，标 `quality_not_reached`；到时仍运行则同时保留 `termination_reason=wall_time_cap`。不同算法的 memory/compute 交换、batch、动力学和正则化变化全部披露。

## 7. F 赛道与 M1 迁移面板

### 7.1 F1–F3 的边界

| ID | 任务 | 首轮深度 |
| --- | --- | --- |
| **F1** | 单室 Hodgkin–Huxley 参数拟合；训练刺激与未见刺激验证分开 | 先核验模型支持、单位、积分和拟合目标；在可比较子集运行。Jaxley/braincell 与 brian2modelfitting 各自算法分栏 |
| **F2** | 多室形态、电缆方程及生物物理参数的拟合/训练 | 能力与小规模正确性边界优先；原形态、离子通道、积分器、参数集合、原件 hash 待锁；不为增加覆盖把多室换成单室 |
| **F3** | 深层卷积 SNN / Spikformer | 支持范围和最小构图、前向、反向、一步更新资格；本轮不预设全规模训练或完整超参搜索 |

参数拟合按相同目标轨迹误差、模拟预算、参数边界和未见刺激误差比较。相同外层优化器可隔离前向引擎成本；每框架最佳原生方法另表。轨迹拟合好、参数可辨识和未见刺激预测是三个不同结论。SBI 的神经密度估计器用 GPU 不代表原 SNN 模拟或 BPTT 用 GPU。[brian2modelfitting 功能](https://brian2modelfitting.readthedocs.io/en/latest/features/index.html)

F 系列没有锁定原始模型、数据和资源预算，现阶段只能冻结任务身份与判定方法；不能将占位模型记作 completed benchmark。

### 7.2 M1：对“已有 Brian 模型能否接入学习”的证据

在改写开始前，按点神经元类型、状态数、突触机制、延迟、噪声、多时钟、空间/多室模型分层预选官方 Brian 样例。记录所有被选模型的来源、版本与 hash，以及选入/排除理由。样本分母在适配前公开，不能只报告成功模型。

每个模型分别报告：

| 阶段 | 输出 |
| --- | --- |
| 原模型运行 | 原版 Brian 可运行性、前向参考制品 |
| 支持判断 | 原样支持/有文档的不支持/尚未资格；具体缺失机制 |
| 改写 | 实际主动工程时间与总历时、代码行增删、配置修改、删除或替代的机制 |
| 语义 | threshold/reset/delay/schedule/状态轨迹与 spike 核验 |
| 梯度与学习 | 哪些参数可学习、什么梯度规则、是否通过 oracle、任务是否达标 |
| 回灌与重放 | 训练参数回灌原 Brian 模型，使用相同及未见刺激重放，记录行为差异 |

“梯度路径有近似”与“前向模型被修改”必须分开描述。不先承诺“大部分 Brian2 模型”；只对冻结样本集、支持子集和实际通过阶段给出分子/分母。

## 8. 硬件梯度与分布式语义

以下是资源类型候选，**机器型号、可用设备和资源预算尚未锁定**。不存在的硬件标 `unqualified` 并说明，不虚构运行结果，也不自动启用付费资源。

| ID | 候选拓扑 | 比较规则 |
| --- | --- | --- |
| **H0** | Mac ARM CPU；Apple Metal 单设备 | CPU 与 Metal 分表；用于本地使用场景，不与 Linux NVIDIA 直接算纯引擎倍数 |
| **H1** | Linux CPU：1/8/32物理核 | 仅在有对应物理核时运行；固定绑核与 NUMA、禁用意外超订阅；报告 SMT 和 BLAS/OpenMP 线程 |
| **H2** | Linux 单张 NVIDIA GPU | 同型号、同张设备/等价隔离资源比较；CPU 型号与供给线程也锁定 |
| **H3** | 单机2/4张同型 GPU | 真实物理卡；固定 GPU UUID、PCIe/NVLink 拓扑，核验 rank→device 映射 |
| **H4** | 双物理节点；CUDA 子赛道为1+1 / 2+2 GPU，CPU 子赛道另锁核数/节点布局 | GPU 分别与单机2/4卡、相同总卡数比较；CPU 与相同总物理核和资源预算的单节点布局比较。记录网络、NCCL/MPI transport、设备及 CPU/内存配置；CPU 与 GPU 不同榜 |

H4 的 CPU-only NEST/Atlas 模型分区使用独立拓扑清单和预算，不能从 GPU 子赛道结果推断。H3 成功不自动说明 H4 成功。Apple Metal、CUDA、HIP 分别标记后端；底层支持 HIP 不说明某训练编译器或分布式路径支持 HIP。本轮硬件执行不包含 TPU、AMD HIP 性能或神经形态芯片，能力证据可以保留。

### 8.1 三种并行方式独立计量

1. **实验并行**：独立模型/种子/超参任务并发，报告总工作完成时间；不称作单模型分布式训练。
2. **数据并行**：复制模型、按样本切 batch、同步参数更新。强扩展固定 global batch 与总样本/更新数，测1→2→4卡；固定 per-device batch 的吞吐扩展另表，质量和有效 batch 变化披露。
3. **模型分区**：单一网络状态、边、参数或 tape 真正分布到设备/节点，报告每 rank 持有量和通信。权重分区、状态分区、tape 分区分别证实；不得仅因用了 MPI 就声称容量扩展。

当前 Atlas MPI 分担部分数值工作，但复制模型、输入和 tape；它不是已证明的 tape 内存分片。正式评估既记录每 rank，也记录全 job 总内存，不能将复制后的总容量误说为可训练单模型容量。[当前本地证据与边界](training-competitive-analysis-2026-10-04.md#atlas-必须分开看的三层)

mlGeNN 的 global batch 按 ranks 分割与 NCCL reduction 是数据并行。NEST MPI 通常分区同一个网络，属于不同并行语义。[mlGeNN 源码](https://github.com/genn-team/ml_genn/blob/fca7e2c70df181014c0d0effddadb09ef3b9a91a/ml_genn/ml_genn/compilers/compiler.py)、[NEST MPI](https://github.com/nest/nest-simulator/blob/main/doc/htmldoc/hpc/parallel_computing.rst)

两卡/跨机运行先验证真实 GPU UUID、所有 rank 的参与及更新一致性。单卡启动多个 ranks 只能归入单卡多进程调度测试。checkpoint/restart 需验证模型、optimizer、RNG、数据位置与 rank 状态恢复，不能只验证权重文件可读。

## 9. 数值精度、公平实现与计时

### 9.1 数值 profile

每条记录分别写出：输入/参数存储、前向、累加、反向、optimizer、主参数、通信归约的 dtype，以及 TF32、AMP、fast-math、denormal 等开关。仅写“FP32”不足以比较。

当前 Atlas GPU 候选是前向FP32 / 反向FP32 / 主机 optimizer FP64，CPU 候选为FP64。已检查的旧版本 GPU 路径还涉及每步主参数FP64转FP32、每样本梯度FP32转FP64后的顺序 batch 累加、loss从FP32转FP64求均值；这些细节必须对最终版本重新核验，不能只匹配三个 dtype 标签。对手需要匹配该分阶段 profile 才能进入严格倍数表；其他原生配置另列效率/质量结果。**CPU64 与 GPU32 的时间比不能称纯设备或纯引擎加速。** 混合精度必须独立过资格和质量门槛。

在可用框架中建立 matched-profile 组；无法保持同 profile 的组合标明差异，不人为将所有对手降至不合理路径来制造主排名。

严格 profile 匹配的是数值与归约契约，不强制 optimizer 的执行位置。对手若能在 GPU 上以 FP64完成等价更新并通过 oracle，就是合法优化；placement 单独记录。Atlas 实际主机 optimizer 引起的往返传输仍计入其公开 API时间，不能要求对手复制该开销。

### 9.2 主计时边界

主成绩为公开引擎 API 的端到端时间，包括执行该 API 所必需的输入/输出编码、复制、同步、子进程和分布式启动、参数与结果回传。网页/UI、人工排队和数据下载另报。

对当前 Atlas，**每次 `trainer.step` 的 JSON、文件读写、子进程启动以及该调用中的 MPI launch 均计入主 step 时间**。不能拿 Atlas kernel-only 对比对手公开 API；若未来改为常驻进程，记录为新的实现版本并重新资格。

| 指标 | 定义 |
| --- | --- |
| 环境准备 | 安装、下载、构建系统依赖；单独记录，不混入每 step |
| cold time | 新进程、空项目编译缓存，从构建模型/训练接口到完成第一完整更新；预先准备的数据读取策略一致 |
| warm public API | 缓存可用、明确同步后的完整 forward+loss+backward+optimizer 与必要桥接开销 |
| 内部分解 | forward、backward、optimizer、数据转换、通信、I/O、启动；难以无扰动分解时保留 unknown |
| kernel-only | 独立诊断表，注明 events/同步方法及遗漏项；不替代主成绩 |
| 真实训练任务 | 从统一任务调用到质量门槛或预算终点，包含按协议发生的编译、数据准备、验证与 checkpoint |

异步 GPU 要在边界同步；分布式主计时采用 **coordinator 侧完整公开调用的 wall time**，涵盖调用前后的 JSON、文件、`mpirun` 启动和结果回收。各 rank elapsed 的最大值是内部诊断指标，不能代替主成绩，否则可能漏掉启动和桥接。不能只读 rank0 局部 kernel 时间。分解相加若有重叠不得强行当总时间。

### 9.3 采样与运行次序

标准微基准：每个进程 **10个 warmup steps + 50个 measured steps**，共 **5个独立进程**，与五个固定 seed 一对一，不是5 seeds×5 processes共25个进程。输入在计时前锁定，完整训练更新按预声明方式连续推进，不能一方只 forward、另一方不断更新。

50个 step 不是50个独立实验；统计单位为进程/种子。每进程保存原始50次延迟和其中位数，主表给五进程中位数、范围与配对比值；置信区间注明样本量和方法，避免以250个相关 step 伪造精度。

微基准单 case 候选 wall-time cap 为1800秒。长序列/复杂负载若预计无法使用标准计数，须在正式运行前锁定统一 step 数、warmup 和 wall-time cap，并给出原因。未锁定则保留标准方案并按 timeout 记录，不能运行中偷偷缩短 T、B、迭代数或只计较快的前几步。

运行顺序按预生成随机或轮换表交错框架，控制热状态与后台负载；记录 GPU 时钟/功耗限制和是否共享。发现外部干扰可按统一预声明规则重跑，原记录保留并注明无效原因。

## 10. 内存、容量与失败分类

内存同时记录：主机进程 RSS、包含子进程的 job/RSS、框架 allocator allocated/reserved、设备 driver 观测值、显存/统一内存环境、采样周期，以及所有 ranks 的峰值。设备峰值应报告逐卡与同一时刻 job 聚合；不把各 rank 在不同时刻的峰值相加伪称瞬时总量。

H0 统一内存与 CUDA 显存不是同一容量计量。CUDA caching allocator 的 reserved 不等于有效状态；driver 观测包括上下文/库；三者都保留。OOM 后记录最后可完成完整更新的规模、失败的分配阶段与设备可用内存。最大容量必须完成前向、反向和 optimizer，只有模型构建成功不算。

当前原生实现的 `max_tape` 上限1 GiB与 JSON输入64 MiB限制属于软件准入预算，不是已测量的 RSS/VRAM 上限。进入性能运行前计算精确序列化字节数和估算 tape 请求，并保存预检制品。SHD 的 dense `B32×T×700` 在浮点 JSON编码下、例如 T≥750时，仅输入就可能超过64 MiB；必须测量实际编码，不能把此拒绝标为 GPU OOM。

不能为绕过准入在运行中缩短 T 或改 B。可预先定义保持同 global batch 的 microbatch+梯度累加变体，默认关闭：每个 global batch 只做一次 Adam，保持 loss 缩放、梯度累加精度与顺序契约，并通过 Q0 后独立记录。未实现或未资格时仍按原负载报告拒绝，不能用变体覆盖基准失败。

容量测试显式设置并记录所选公开版本允许的 `max_tape_bytes` 上限；当前候选为1 GiB，不能用默认64 MiB准入结果代表物理容量。先运行固定粗网格，再在相邻成功/失败规模之间最多3次细化：取 `floor(sqrt(lower * upper))`，若没有严格位于两端之间的整数则停止。软件拒绝、物理 OOM和超时边界分开；最高网格点通过时只报告“至少达到该规模”，不能宣称已找到最大值。

固定状态枚举：

| 状态 | 含义 |
| --- | --- |
| `unsupported_documented` | 锁定版本有明确不支持证据，附来源与具体特性 |
| `not_applicable` | 在 `capability_status` 中表示工具/算法不属于对应赛道，例如原版 Brian2 不参加通用 SG-BPTT；不当作速度失败 |
| `unqualified` | 尚未运行/环境资源未锁定/尚未取得资格；不是零分或不支持 |
| `dependency_error` | 安装、编译、驱动、依赖或启动失败；保存原错误与锁定信息 |
| `semantic_mismatch` | 方程、事件、输入、梯度或更新不满足契约；不进入性能倍数表 |
| `budget_rejected` | 软件 tape/序列化输入等准入预算拒绝，保存估算量、精确输入字节与配置上限；不等同物理 OOM |
| `oom` | 内存耗尽，记录 host/device/rank 及阶段 |
| `timeout` | 在固定 wall-time cap 内未完成指定工作量 |
| `numerical_divergence` | NaN/Inf、非预期爆炸或冻结数值判据失败 |
| `completed` | 指定工作量完整完成并通过对应正确性门槛；不自动表示任务达标 |
| `quality_not_reached` | A/F 任务预算内完成或停止，但未达到冻结质量门槛 |

记录 `capability_status`、`execution_status`、`qualification_status`、`quality_status` 和 `termination_reason`，可区分“不适用”“支持但尚未运行”“执行到 cap 且未达质量”等情况。能力判断与执行状态分开；`not_applicable` 不进入性能排名。timeout/OOM 不可只保留最快成功子集；未支持点不得从兼容性分母消失。

## 11. 每赛道结果 schema

所有表共享以下字段；实现时可以 JSON/Parquet 保存，展示层不必展开全部列：

```text
protocol_id, protocol_sha256, freeze_id, run_id, timestamp
track, workload_id, workload_variant, framework, framework_version, source_commit
adapter_commit, adapter_sha256, model_sha256, data_manifest_sha256
environment_lock_sha256, container_digest, compiler, driver, runtime
hardware_id, host_ids, cpu_topology, gpu_uuids, interconnect, ranks, threads
parallelism_kind, global_batch, per_rank_batch, N, E_actual, parameters_unique, T, dt
numeric_profile, algorithm, memory_strategy, seed, initial_arrays_sha256
capability_status, qualification_status, execution_status, quality_status, termination_reason, error_artifact
raw_metrics_path, log_paths, output_sha256, resource_budget, elapsed_s
```

`E_actual` 是物理/逻辑有效边数；masked dense 另记录 allocated candidate edges。卷积必须同时保存独立 kernel 参数数、逻辑连接数和实际存储。时间步 T 与生物时间不可互换。

| 赛道 | 额外必需字段 |
| --- | --- |
| **Q0/E** | `semantic_contract_sha256`，state/spike/loss/VJP/gradient/update 最大误差及首次分歧；cold、每进程完整 step 原始时间、内部阶段；RSS/allocator/driver/all-rank 内存；checkpoint/TBPTT 长度；实际 spikes；合格后的配对速度比 |
| **L** | 原 Brian 源码 hash、pre/post/delay/schedule 契约、fixture weight deltas、STDP 规则；构图/初始化/运行/记录时间；生物时间、发放率、事件数、通信量；每 rank neuron/edge/state 分配 |
| **A** | split/encoding hash、搜索空间与预算、选定超参、每 epoch elapsed/loss/validation；预定 target、首次达标时间、终点和选定 checkpoint；最终 test、五 seed 成功率、总算力时间及模型状态量 |
| **F** | 原模型/形态/通道/刺激 hash、拟合参数边界、优化器、模拟调用次数、训练与未见刺激误差、数值解设置；supported/qualified/完整任务深度；只做边界时明确 `performance_run=false` |
| **M1** | 预选模型全集 manifest、层别、支持/改写/语义/梯度/学习/回灌各阶段状态、主动工时/总历时、LoC增删、移除机制、重放差异 |

费用只用冻结时记录的实际租赁或事先声明的计价规则，披露计价时间；没有可信价格时只报 CPU/GPU-hours。不能用理论 FLOPs、TFLOPs 或宣传能效替代实测任务成本。

## 12. 决策门槛：何时继续，何时收窄

下列是本项目建议的内部研发门槛，不是行业标准；执行前与预算一起冻结。

| 赛道/主张 | Go 条件 | No-go / 需要改写的结论 |
| --- | --- | --- |
| **E 执行优势** | 同语义、同 profile、同硬件资格全部通过；至少一个预声明重要模型族的两个相邻规模，完整 step 中位数≥2×加速，五配对结果完整披露 | 只赢 kernel、不同 dtype、未优化循环或只有一个特选点，不发布引擎领先 |
| **E 容量优势** | 同设备预算和同全 BPTT，最大完整训练 T 或有效模型规模≥2×；上下边界、OOM与重算代价披露 | 只增加总设备内存，或 TBPTT 对全 BPTT，不能记作同算法容量优势 |
| **L 原模型优势** | 原方程和局部更新通过，两个相邻负载获得明确端到端节省；有规模数据支撑 | 仅原版 NumPy fallback、删机制或改活动负载，不支持原模型加速主张 |
| **A 任务价值** | 同质量目标下五 seed 中至少4个达标，任务总时间/成本有稳定改善；非劣质量候选容限为1个百分点，配对种子置信区间方法事先锁定，过宽区间记证据不足 | 只比默认值、靠 test 调参、丢失败 seed 或没达到共同质量，不宣布更好训练 |
| **F 能力投入** | 明确用户任务在原模型上通过资格；拟合误差与未见刺激指标满足冻结门槛，投入与资源可接受 | 边界核验未通过则如实记不支持/未资格，不用简化模型填补 |
| **M1 迁移价值** | 在预选分层样本中证实可量化减少改写和验证成本，并保留原机制及回灌检查 | 分母不公开或只挑成功示例，不宣传“大部分可训练” |

分布式进入下一阶段的最低条件是多 rank 更新正确、资源真实、端到端没有系统性倒退且能解释通信/复制成本。建议作为产品卖点的候选 strong-scaling 门槛为2卡≥1.5×、4卡≥2.5×完整 step 加速，至少覆盖两个有足够工作量的负载；这不是通过 MPI API 的资格条件。跨机须同总卡数比较，单独报告网络带来的损失，不把“能启动”当作扩展收益。

若 Atlas 同模型没有速度/内存优势但 M1 明显省事，可将价值定位为模型接入与研究流程。若优势集中在稀疏循环、多状态或长序列，应围绕那个范围投入；不得外推到所有 SNN、所有 Brian2 模型或所有设备。

## 13. 分阶段执行与真正冻结所需制品

| 阶段 | 必做 | 进入下一阶段的条件 |
| --- | --- | --- |
| **P0 资格与冻结** | 版本/依赖候选、Q0、公共数组与独立 oracle；确定真实数据和模型原件、计量脚本、设备与预算 | 每个待测组合取得对应资格；所有 freeze 字段齐全并生成不可变 manifest |
| **P1 CPU / 单 GPU** | H0/H1/H2 可用资源；E1/E3/E6 为主，E2/E4/E5/E7 分层扩展；L1/L2；A1/A2 | 主计时与内存路径可解释，模型/质量无未解决分歧；完整公开失败点 |
| **P2 真多卡** | H3；区分数据并行与模型分区，固定 global batch，记录 GPU UUID 与所有 ranks | 多 rank 更新正确，工作量/状态分配确认，具备进一步扩展的负载 |
| **P3 跨机器** | H4 GPU 与相同卡数 H3 比较；H4 CPU 与同物理核资源的单节点比较；传输、恢复和通信分解 | 才能给出跨节点训练的实际结论 |

F/M1 按模型资格可与 P1 独立推进；不因单一复杂模型未支持阻止已合格 E/L/A 子集执行。完整竞争格局不意味着全部框架×全部模型×全部设备的笛卡尔积。

执行冻结清单必须包含：

- Atlas 原生源码快照与 bundle 身份；平台入口、部署 bundle、研发版本分别命名。
- 各竞品精确发行/commit、wheel SHA256、依赖锁、编译器/flags、驱动/runtime、容器 digest。
- 适配器与独立 oracle 源码 hash、Q0报告、支持矩阵及未合格项。
- 所有模型方程/调度/参数约束、初始数组、edge lists、数据原件/样本划分/编码 manifest 和 SHA256。
- 机器/CPU/GPU UUID、线程与rank分配、网络拓扑、计时/同步/内存采样方法。
- 质量目标、调参搜索空间和预算、wall-time/OOM/重试规则、长负载采样例外、正式运行次序。
- 结果 schema、统计方法、决策门槛、冻结时间、协议与 JSON 的 SHA256。

当前以上运行制品尚未齐备，所有未运行组合的默认状态为 `unqualified`。本文件完成的是**可执行实验的设计候选**；取得并封存资格、环境、数据和资源清单后，才能发布 `execution_frozen` 版本。任何影响结果的新变更生成新的 freeze ID，保留旧结果，不能覆盖原冻结记录。
