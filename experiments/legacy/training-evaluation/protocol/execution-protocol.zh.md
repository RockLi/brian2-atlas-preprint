# Atlas Training 首轮引擎评估：有限执行协议 r1

日期：2026-10-04（Asia/Singapore）。状态：`scope_frozen_execution_pending`。本文件完成范围、数学增补、资源请求规格与完成条件；不代表全轮环境/数据/资格已完成，也不包含新性能结果。可运行小切片分别封存 `execution_frozen` manifest，其他格保持 pending。主任务的运行证据与本文件分开保存。

执行顺序：立即做本地 CPU Q0、独立 Brian 前向、L1 和 E1-small；合格后完成 CPU/单卡有限矩阵和 SHD；有明确资源后运行真多卡与双机；F/M 能力边界同时推进。任何阶段预算耗尽都保留未完成格，不能把“计划写好”或“本机暂时不能运行”称作完成比较。

## 范围与完成含义

[覆盖矩阵](coverage-matrix.md) 包含原来的 23 候选 × 17 工作负载，共 391 格：57 必测、77 专项、257 本轮不适用。每格具体理由、独立能力/资格/执行状态见 [JSON](coverage-matrix.json) 和 [CSV](coverage-matrix.csv)。mandatory/specialized 表示评估义务，不是已验证能力；缺硬件不得改成不适用。专项必须给出版本与支持/语义资格证据，不能因性能可能不利而跳过。

核心 Atlas、snnTorch、SpikingJelly 前沿强优化、Spyx、BrainPy state/brainstate 全部进入 Q0、dense、稀疏循环、适应双状态、长序列、宽度和 A1/A2。Norse 是独立数值交叉参考；SpikingJelly 稳定版、BrainPy classic 保留少量版本交叉点。EXODUS 指定前馈 CUDA；SLAYER/Rockpool 指定动力学与 SHD 原生算法；BrainTrace、mlGeNN、NEST 指定在线/事件算法；原 Brian 及扩展指定原模型局部学习；Jaxley/braincell/brian2modelfitting 指定生物物理边界。边界证据和速度榜分开。

F1/F2 本轮义务为源模型、方程、积分器/单位、最小前向、可用梯度/一步更新和拟合接口的能力核验，`performance_run=false`；不预设完整拟合成绩。F3 固定四层卷积网络资格，不宣称它等价于 Spikformer。完整 HH 拟合或大型深 SNN 训练需另立数据、目标和优化预算。这是对源文件“先做能力边界”的具体化，不能将能力核验算作任务拟合完成。

## 来源与并发隔离

三份交接源文件逐字保存在 `sources/`，逐文件字节数/SHA256 见 [source-manifest.json](source-manifest.json)。它们是研究候选，不是本轮新成绩。主任务另存 Atlas commit、dirty patch、一致的源码副本、运行时与适配器 hash；本协议不引用变化中的工作树作为可复现运行依据。

只读核对的源码仍明确 JSON 上限 64 MiB、native tape 许可上限 1 GiB；原生文档仍将跨主机/多物理 GPU 标为暂缓。后者是研发任务约束记录，本评估不能推断自己获得云资源批准。历史两主机 CPU、小型单 L4 多 rank 正确性不赋予当前快照真多卡资格。任何新引擎优化/修复必须保留失败基线并换版本重新资格。

全部环境、编译缓存、TMPDIR、Brian standalone 输出和运行日志都进入本评估独立目录。不得修改活动源码、共享 venv 或共享构建输出。只有包含全部运行依赖的快照与哈希一致才能成为正式候选。

## 数学契约与有限形状

源协议的同步 LIF 不变：每 tick 先所有群体 drift，再所有群体 threshold，随后事件输入与 reset；严格 `>`；Euler `beta=.95`；surrogate `g(m)=1/(1+5|m|)^2`；reset spike gate detach；Adam、batch mean、输出时间均值和初始数组一致。对手可以采用 JIT、融合、合法稀疏与 checkpoint，不能为代码表面统一强迫慢 Python 循环。

独立 oracle 核验完整状态/spikes/loss、指定 VJP、全部梯度、SGD、一个及多个 Adam 更新、moments、batch 归约和恢复。硬阈值有限差分不验证 surrogate；只对平滑子图使用有限差分。共同数组必须落盘取 SHA，不能以“同 seed”代替。资格失败保留首次分歧，不在看过结果后只对失败方放宽容差。

[semantic-contracts.json](semantic-contracts.json) 补齐先前空缺：

- E4 是明确的双状态适应阈值 Euler 模型。`uv=(1-dt/tau_v)v`、`ua=(1-dt/tau_a)a`，阈值 `uv>1+ua`，电压 subtract reset gate detach，适应增量 `k*s` 保留 surrogate 路径。初值 `tau_v=20ms,tau_a=200ms,k=.1`；固定与可学习参数两栏；可学习物理参数以有界 sigmoid 映射，原始参数数组共用。它不自动等价于某库默认 ALIF。
- E5 分别是公开 Brian AdEx 方程、Izhikevich 方程和明确命名的 synthetic four-state 滤波模型。固定 `[128,128,20] B8 T512 dt=.1ms`；逐个锁单位、初值、Euler simultaneous flow、threshold 比较符、reset 与电压输入顺序、surrogate 尺度。四状态模型不称 HH。AdEx/Izh 的受控网络变化有记录，不能算原示例无改写迁移。
- L1 对齐实际适配器定义的八条一对一突触 fixture：`dt1ms,T16,tau_pre10ms,tau_post12ms`，pre trace `+.08`、post trace `-.09`，权重 clip `[0,1]`，同 tick pre 先于 post，含不同延迟/上下 clip/重复时序。完整时间数组见 JSON。
- L2 保留 CUBA 电流方程，明确 exact 线性流、20/5/10ms 常数、`El=-49mV,Vt=-50mV,Vr=-60mV`、5ms refractory。每目标恰好 80 个 E 与 20 个 I 源、无自连接/重复边；边排序后 delay 循环1/2/3ms。仅 E→E 使用 L1 同规则；无运行时随机驱动。与公开 CUBA 的连接方式变化单列。准入、refractory 和 event 顺序仍需独立资格。

这些增补目前是已具体写出的候选数学契约；oracle、适配器、实际数组/hash 尚未完整，不能标作性能可执行冻结。

[finite-engine-cases.json](finite-engine-cases.json) 固定 19 个独立 E 形状：E1 两个、E2 一个、E3/E6/E7 合计九个去重点、E4 四个、E5 三个。E3 只做五点单轴扫描，不做九点笛卡尔积。E3/E6/E7 重合点只在所有输入/状态/实现/硬件相同时复用结果并保留全部标签。容量最多各增加三个 T、三个宽度几何中点，不能无限追边界。

## 数据、调参与正式计时

五个正式种子为 `11,23,37,51,71`。每微基准是五个独立进程，每进程 10 warmup + 50 measured steps。源协议对“单 case 1800秒”存在粒度歧义，本执行候选明确为五进程总计1800秒，每进程等额360秒；看过正式结果后不再更改。任何 timeout 保留五种子计划与实际完成量，不缩 T/B/step 补成完成。主统计单位是进程中位数，报告五个值/配对比值及范围，不把250个相关 step 当独立样本。

主成绩是 coordinator 完整 public API wall time。Atlas JSON、临时文件、参数拷贝、子进程、MPI launch 与结果回收全部计入；各 rank 最大 elapsed 与 kernel-only 仅为诊断。异步设备在边界同步。cold 新进程/空编译缓存到首个更新，warm 仍包含每次必须发生的桥接。若用原生优化改变 dtype/模型，独立表报告。

CPU FP64、GPU前后向FP32/optimizerFP64、native-best 三种 profile 分开。参数存储与每次 cast、per-sample 梯度/损失精度、按样本/按 rank 归约次序、optimizer moments/bias correction/epsilon 位置、TF32/AMP/fast-math 均是资格字段。数值等价不要求对手复制 Atlas 的 host optimizer 位置。

A1 为 MNIST `[784,128,10] T100 B32` 流程检查，固定像素/255 常值电流编码，10epochs/1800秒，55k/5k 分层 train-only split，目标90%/95%，固定 Adam 不做 HPO。A2/A3 为 SHD 主任务，B32、700通道、hidden256、20类、50epochs/7200秒，目标50%/70%/80%全部报告。

SHD 只读取 test 时间戳 extent 决定 `T=floor(max_time/dt)+1`，不读 test 类别/成绩做选择；count-bin 保留所有重复事件。训练集 speaker 子集在保证两边所有类别覆盖的前提下选最接近20%样本量者，speaker ID 字典序打破并列；实际索引在任何拟合前封存。无法满足则改协议后再训练，不使用 test 调参。

每个 A2/A3 框架/算法八配置 × 调参种子101/103；每 run 10epochs/1800秒。SG-BPTT 搜索固定学习率 `{.0003,.001,.003,.01}` × 公共初始权重乘数 `{.5,1}`，按两种子验证均值选配置，失败计0并留记录，再以调参耗时、配置ID破并列。在线算法原生参数搜索尚需精确依赖/API后锁定，当前不能伪称已冻结。五个正式种子不追加调参。按验证最佳、同分最早 checkpoint，最终每种子 test 一次。

没有目标达标则保留学习曲线/预算终点，记录 `quality_not_reached`。非劣质量参考五个配对准确率差的95% t区间（df4），下限大于−1个百分点才允许非劣声明；所有失败 seed 同时公开。时间到质量包含 compile/loading/validation；不能删除未达标种子后给出唯一好看比值。

运行前精确序列化整个请求，保存 JSON 字节数和实际 tape 估算/上限。SHD dense input 超64MiB或tape超1GiB是 `budget_rejected`，不是 OOM。不静默缩T、丢事件或改batch。microbatch4/global32候选默认关闭，只能资格后另立版本，按固定精度累加梯度且每global batch只做一次Adam，原monolithic失败格保留。

## M1 公开分母与能力边界

[brian-corpus-manifest.json](brian-corpus-manifest.json) 在适配前固定12个官方示例，从本仓库 commit `81eb571d78292a0e24a8e79980ea0d5c26e8a48c` 用 `git show` 提取原件，放在 `corpus/originals/` 并逐文件 SHA256：

1. IF_curve_LIF：点LIF与refractory。
2. CUBA：循环稀疏电流网络。
3. adaptive_threshold：双状态与随机输入。
4. Brette_Gerstner_2005：AdEx。
5. Izhikevich_2003：异质参数、双路径延迟、run_regularly噪声。
6. COBAHH：电导与六状态HH网络。
7. STDP：事件驱动trace与局部可塑性。
8. jeffress：异质延迟、TimedArray和噪声。
9. stochastic_odes：乘性噪声与Milstein/Heun。
10. custom_events：自定义事件、布尔状态与多路径。
11. hodgkin_huxley_1952：1000室真实空间模型。
12. compare_GSL_to_conventional：多时钟与GSL可变步长。

这是按机制有意分层的有限语料，不是所有 Brian 模型的随机代表样本。每个原模型运行上限600秒，每模型主动适配上限120分钟。每项分别记录原运行、支持、改写、前向、梯度、学习、参数回灌、同刺激/未见刺激原Brian重放。无支持、依赖失败、超时、工程未完成全部留在12的分母；超时不等于不支持。移除机制或缩模型必须另命名变体，不把简化版本算作原模型成功。训练后参数未回灌与重放的模型不得宣称迁移全链完成。

## 资源预留、停止条件与完成清单

本机8CPU/16GiB macOS arm64；初期可用磁盘约992MiB，主任务后续观察约2.4GiB，必须运行前复查。sandbox 报 CUDA/MPS不可用不等于机器物理上无Metal。本地P0仅小CPU资格和有界制品，保留768MiB磁盘下限；不清理用户文件或共享缓存。正式CPU预留72小时与50GiB磁盘，当前未满足完整磁盘预算。

H1 规格为32物理核/128GiB/200GiBSSD；H2为单L4 24GiB、8CPU/64GiB/200GiBSSD；H3为四张同型L4、32CPU/256GiB；H4为两节点各2张L4、16CPU/128GiB、声明至少25Gb/s链路。CPU跨机另用16+16物理核与单机32核比较。型号是资源请求规格，不是已获得硬件；若换型号，先重新冻结。

预留上限：H0+H1 72小时、H2 400 GPU-hours、H3 96 GPU-hours、H4 GPU96 GPU-hours，CPU跨机12小时。总CUDA资源上限592GPU-hours，需实际SKU/地区/计价时间与存储/网络报价才可得到货币预算，本次未授权付费作业。理论最大微基准工作量为9个实现视图 × (19形状+最多6细化) ×0.5小时=每硬件112.5小时，另有专项；所以72小时CPU是阶段上限，不是假定全部case都在该时间内必然完成。到cap剩余格继续pending并调整资源预留，不挑点删除。

适配/资格工程估计40–80小时，F/M16–32小时；这是规划量，不是实测或完成承诺。硬件pending不阻塞独立CPU工作。多机/多卡不能以单卡多rank替代；Atlas复制模型/输入/tape，必须按rank记录状态持有量，不能提前声称模型容量分片。

明确完成条件见 [execution-plan.json](execution-plan.json) 的 `completion_checklist`：

- P0：隔离源码与运行时身份一致；公共数组、oracle、对应Q0/Brian/L1资格；可运行切片的依赖、硬件、数值profile和适配器hash封存。
- P1：所选CPU与单GPU格全部有完整结果或带原始证据的终态失败；强优化基线审计；A1/A2/A3数据、HPO、五seed与质量结果齐全。
- P2：真实1/2/4GPU身份与rank映射，固定global batch强扩展、固定local batch吞吐、真实模型分区分别报告。
- P3：真实两节点与同资源单机放置比较，CPU/CUDA分表，通信/恢复证据齐全。
- F/M：选定边界逐项判定；12模型完整阶段状态与投入，成功子集给出学习和回灌原模型重放。
- 报告：原始JSON/log、独立验证入口、版本/数据/设备清单、图表与中文结论，明确Atlas实测优势、不占优项和未执行硬件项。

资格、环境、数据、硬件、搜索空间尚未锁定的格不得进入性能倍数表。源协议的unsupported_documented、unqualified、dependency_error、semantic_mismatch、budget_rejected、OOM、timeout、numerical_divergence、completed、quality_not_reached语义继续生效；能力、执行和质量分别记录。没有完整五seed/适用对手/同语义证据就不宣布Atlas领先。
