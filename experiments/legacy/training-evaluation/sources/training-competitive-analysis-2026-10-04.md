# Training 竞品分析与首轮引擎评估建议

调研日期：2026 年 10 月 4 日。对象：Brian2 Atlas Training 的研发与产品决策。

**建议建立首轮引擎评估，但目前没有证据预先判定 Atlas 领先。** 核心对手应包括 snnTorch、SpikingJelly、Spyx 和 BrainPy 的新状态化训练栈；Brian2 原版是原模型执行和局部可塑性的基准，mlGeNN 是事件驱动监督训练的强对照，NEST 与 Jaxley 分别覆盖大规模在线学习和复杂生物物理模型。不同学习算法、数值模型和并行方式必须分开比较。

本报告完成官方文档、源码及已有本地证据的核查；没有安装竞品环境、重新运行训练、租用 GPU 或产生新的性能成绩。配套文件：[首轮评估协议](training-engine-evaluation-v1.md)、[机器可读计划](validation/training-engine-evaluation-v1.plan.json)。协议状态为冻结候选；运行环境、适配器和数据制品完成锁定后才成为可执行的冻结实验。

## 本次分析要回答的决策

用户采用 Training 的理由应落到研究任务：同样的模型和学习方法是否更省时间或内存；是否能训练原来受容量限制的模型；保留原 Brian2 模型能否显著减少转换与验证工作。编辑界面、可学习参数、支持 GPU 或 MPI 的功能列表不能替代这些证据。

首轮同时回答三个问题：

1. **执行效率**：同方程、同梯度规则、同数值精度和同硬件，Atlas 是否有优势？
2. **研究成本**：不同合法学习方法解决同一任务时，达到相同质量的时间、内存和算力成本分别是多少？
3. **采用成本**：原 Brian2 模型需要改多少代码、删去哪些机制，训练前后是否能验证原模型的行为？

第三项单独报告。它可以成为采用理由，但不能把少改代码换算成引擎速度。

## 竞争者的实际分工

下表中的“官方”表示存在官方文档或实现，不代表本次已在我们的硬件上复现。“待验证”不等于不支持。CUDA GPU、Apple Metal、数据并行、模型切分及跨节点均分别判定。

| 项目 | 主要学习与模型能力 | CPU 与 GPU | 分布式证据与边界 | 首轮角色 |
| --- | --- | --- | --- | --- |
| Brian2 原版 | 方程驱动仿真、用户定义突触可塑性；原版没有本轮所需的通用 SG-BPTT 训练接口 | CPU runtime、C++ standalone、OpenMP；GPU 要明确标为扩展后端 | 官方独立仿真多进程不是一个网络的跨节点分片 | 原方程前向及 STDP 基准，不放进没有同算法实现的 BPTT 速度榜 |
| brian2modelfitting | Brian 模型的 traces/spikes 参数辨识、无导数优化及 SBI 后验推断 | runtime/C++ standalone；SBI 密度估计器的 CUDA 训练不等于原模型 CUDA BPTT | 自动并行候选仿真；没有据此确认单模型多机分片 | 广义 Training 的参数拟合边界；比较达到误差目标的时间和模拟次数，不进入 SG-BPTT 核心速度榜 |
| Brian2CUDA | Brian2 方程和支持范围内的突触可塑性在 CUDA 执行 | NVIDIA CUDA；不是独立通用梯度训练框架 | 本次未核验通用单模型跨 GPU/节点训练 | 原模型 GPU 可塑性对照 |
| Brian2GeNN 与 GeNN | Brian2 适配层及代码生成仿真；GeNN 自身具有神经元、突触和可塑性机制 | GeNN 有不同后端，Brian2GeNN 的具体设备/模型兼容以锁定版本核验 | 不能从其他 GeNN 应用的多卡能力推断 Brian2GeNN 全部支持 | 原模型迁移与可塑性辅助对照 |
| snnTorch | PyTorch SG-BPTT、循环/卷积组合、多状态自定义神经元、可学习参数 | CPU/CUDA 官方；MPS/compile 的具体模型需运行核验 | PyTorch DDP 可作为适配路径；不能由 nn.Module 自动推断通用模型分片或跨机已验证 | 主要易用性与常规梯度训练对照 |
| SpikingJelly | SG-BPTT、STDP、深度 SNN；前沿版本有 Triton、FlexSN、memopt | Torch/CuPy/Triton 要分别锁版本和适用模型；MPS 不作未经验证的承诺 | latest 文档明确 Vision DDP/FSDP2/TP/PP；自定义模型需分片规则；公开实测为单机四卡 | 必测强性能、内存及多卡对照 |
| Norse | 显式状态的 PyTorch 神经动力学，循环网络、梯度学习、STDP | CPU/CUDA 官方；MPS 待验证 | 官方有 Lightning 多 GPU 示例；不是通用模型切分证明 | 数值语义与动力学的独立交叉基线 |
| Spyx | JAX/Flax NNX 的 SG-BPTT、LIF/ALIF/循环网络，整网及时间循环 JIT | CPU/CUDA；数据全驻留显存与 streaming 条件分别计量 | 本次未核验包级多节点训练；不从 JAX 通用能力推断 | 必测 JIT 效率、SHD 与长序列对照 |
| Sinabs + EXODUS | Torch 接口，EXODUS 提供 CUDA surrogate BPTT，支持指定 IAF/LIF/ExpLeak | Sinabs Torch；EXODUS 优化路径要求 CUDA | 本次未核验框架级模型分片/跨节点训练 | 匹配的前馈 LIF/Conv 强性能对照 |
| Lava-DL SLAYER | CuBa、ALIF、resonator、可学习动力学及延迟、循环与卷积 | CUDA JIT 训练路径 | Lava 分布式推理进程不等于 SLAYER 分布式反向训练 | 学习动力学/延迟专项对照 |
| Rockpool | Torch/JAX 参数学习、约束及硬件感知训练；有 LIFExodus | 按 Torch/JAX/EXODUS 后端核验 | 本次未核验框架级跨节点训练 | 动力学与部署专项；纯 LIF 内核可通过 EXODUS 减少重复 |
| BindsNET | STDP、奖励调制塑性、无监督和强化学习 | Torch CPU/GPU | 本次未核验单网络分布式学习 | 局部学习辅助对照，不能硬放进 SG-BPTT 榜 |
| BrainPy state 与 brainstate | 点神经元、状态化 JAX 变换、grad/jit/vmap、自定义动力学 | CPU/CUDA/TPU 生态；本轮 TPU 排除 | pmap2/shard_map 有官方接口；本轮真实 SNN 多节点吞吐尚未复现 | 多状态、循环网络及长序列主要对照 |
| BrainTrace | eligibility trace、D-RTRL、pp-prop、e-prop 等在线方法 | BrainX/JAX 栈 | 不从批处理或 JAX 通用能力推断已验证跨机训练 | 不同算法达到同质量的独立比较 |
| mlGeNN | GeNN 上的 EventProp/e-prop 等监督学习 | 重点是 CUDA 执行 | 当前源码有 global batch 按 rank 划分及 NCCL 归约，属于数据并行；不是单模型容量分片 | 事件驱动监督学习与多卡重要对照 |
| NEST | 大规模脉冲网络、局部可塑性；已有 e-prop 监督学习示例 | 重点为 CPU/OpenMP/MPI | NEST 网络分布式成熟；某一 e-prop 教程是否跨 rank 正确仍需单独验证 | 局部/在线学习与分布式复杂网络对照 |
| NEST GPU | GPU 神经元与 STDP 模型；不默认等同于 CPU NEST 的 e-prop 功能 | NVIDIA GPU | 官方多 GPU MPI 模型分区，通常一 rank 一 GPU，显式 RemoteCreate/RemoteConnect；具体可塑性跨 rank 组合需核验 | GPU 分区网络与局部可塑性补充对照，不作为未经确认的通用 BPTT 引擎 |
| Jaxley | 单室/多室 HH、离子通道、形态和突触参数的可微训练，梯度 checkpointing | CPU/GPU/TPU；本轮只取 CPU/CUDA | vmap 是批处理；本次未找到足以确认包级通用多节点单模型训练的证据 | 复杂生物物理能力边界，不能默认参与 LIF SG-BPTT |

来源分别为 [Brian2 计算与 OpenMP](https://brian2.readthedocs.io/en/stable/user/computation.html)、[独立仿真并行](https://brian2.readthedocs.io/en/stable/examples/standalone.standalone_multiple_processes.html)、[brian2modelfitting](https://brian2modelfitting.readthedocs.io/en/latest/)、[Brian2CUDA](https://brian2cuda.readthedocs.io/)、[Brian2GeNN](https://brian2genn.readthedocs.io/)、[GeNN](https://github.com/genn-team/genn)、[snnTorch](https://github.com/jeshraghian/snntorch)、[SpikingJelly 分布式](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/distributed_training.html)、[Norse](https://github.com/norse/norse)、[BrainState 并行接口](https://brainstate.readthedocs.io/tutorials/transformations/04_advanced_batching.html)、[BrainTrace 算法](https://brainx.chaobrain.com/braintrace/apis/algorithms.html)、[mlGeNN](https://github.com/genn-team/ml_genn)、[NEST e-prop](https://nest-simulator.readthedocs.io/en/stable/auto_examples/eprop_plasticity/index.html)、[NEST GPU 多卡分区](https://nest-gpu.readthedocs.io/en/latest/guides/multigpu_simulations.html)、[NEST GPU STDP](https://nest-gpu.readthedocs.io/en/latest/models/stdp.html)、[Jaxley 模型范围](https://jaxley.readthedocs.io/en/stable/faq/question_03.html)。

## 不能沿用的旧竞争印象

**完整名单还需要 JIT 与专用 CUDA 对手。** [Spyx](https://github.com/kmheckel/spyx) 已提供整网 SG-BPTT；其旧论文 notebook 使用 Haiku，新版本使用 Flax NNX，不能混用版本与成绩。其 experimental 时间并行神经元去掉 reset，不能冒充同一个普通 LIF。[并行神经元边界](https://spyx.readthedocs.io/en/latest/explanation/parallel-spiking-neurons/)

[Sinabs/EXODUS](https://github.com/synsense/sinabs-exodus) 应进入匹配的前馈 CUDA 子任务；梯度语义先资格检查，不能仅因为部署背景而排除。[Lava-DL SLAYER](https://lava-nc.org/lava-lib-dl/slayer/slayer.html) 已覆盖可学习动力学和延迟；[Rockpool 参数约束](https://rockpool.ai/in-depth/howto-constrained-opt.html) 与 [BindsNET 局部学习](https://github.com/BindsNET/bindsnet/blob/master/README.md?plain=1) 也表明这些功能不是市场空白。v1 不要求每个框架跑每个模型，但排除项必须有模型、梯度或硬件层面的理由。

**SpikingJelly 不能只取旧版 eager 实现。** 2.0.0rc1 页面已列 Triton、FlexSN、memopt 和实验性多 GPU；latest 的详细分布式能力要另核对是否进入 RC wheel 或指定源码提交。公平对照需要默认正确配置与官方优化配置各一组。[RC1](https://pypi.org/project/spikingjelly/2.0.0rc1/)、[FlexSN](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/flexsn.html)、[内存优化](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/memopt.html)

**自定义多状态不是空白。** snnTorch 的 `neuron_from_equations` 已能指定状态更新与 reset；但默认神经元的 reset 时刻和替代梯度可能不同。`LeakyParallel` 缺少普通 Leaky 的显式 reset，不能作为同方程优化开关直接加入速度比。[自定义神经元](https://snntorch.readthedocs.io/en/latest/snn.neurons_customneuron.html)、[Leaky](https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html)、[LeakyParallel](https://snntorch.readthedocs.io/en/latest/snn.neurons_leakyparallel.html)

**BrainPy 新生态是强直接对手。** 官方将新能力分布到 `brainpy.state`、brainstate、braintrace、braincell 等包；仅比较 classic BrainPy 会遗漏实际竞争者。BrainTrace 的在线方法不普遍等价于 BPTT，要比较质量与成本，而非声称同梯度加速。[迁移说明](https://brainx.chaobrain.com/summ/brainpy-to-brainx.html)、[D-RTRL](https://brainx.chaobrain.com/braintrace/tutorials/drtrl.html)

**复杂生物物理可训练也已有实现。** Jaxley 有 HH 与多室训练及多级 checkpointing；其 FAQ 同时指出 LIF/Izhikevich 尚无相应 surrogate-gradient 训练支持。能仿真某类神经元、能训练另一类神经元，不能合并成一个支持勾。[训练教程](https://jaxley.readthedocs.io/en/stable/tutorials/07_gradient_descent.html)、[FAQ](https://jaxley.readthedocs.io/en/stable/faq/question_03.html)

**NEST 不只是 STDP 基线。** 已有 e-prop 工作，但大规模论文的“百万神经元”负载测试包含固定活动的 ignore-and-fire 设定，主要考察状态传播；不能当成百万神经元真实任务训练达标的证据。我们同样要分开训练质量、固定活动吞吐和完整研究耗时。[论文及方法](https://arxiv.org/html/2511.21674v1)

## 版本候选与锁定策略

以下是本次查到的候选发行，不是已经安装并验证的环境。运行前必须记录完整 commit/wheel SHA256、所有依赖、编译器、驱动及适配器源码哈希。每个项目使用其兼容依赖环境，不强迫所有项目共享一个可能不兼容的 PyTorch/JAX 版本。

| 项目 | 候选版本 | 处理方式 |
| --- | --- | --- |
| Brian2 | 官方稳定发行待运行锁定 | 从官方包获取；不能把本机 Atlas 修改版当原版 |
| brian2modelfitting | 发行/提交及优化器组合待锁定 | 与 Brian2、Nevergrad/SBI 等依赖一起固定，独立参数拟合赛道 |
| Brian2CUDA、Brian2GeNN、GeNN | 各自发行与 Brian2 兼容组合待锁定 | 保留原版和扩展后端身份 |
| snnTorch | 1.0.0 | 固定 wheel 与 Torch 栈 |
| SpikingJelly | 0.0.0.0.14；2.0.0rc1/指定源码提交 | 稳定与前沿分栏，RC 和 latest 不自动等同 |
| Norse | 1.1.0 | 固定发行并核对旧文档中的 API |
| Spyx | 1.0.0 | 固定 NNX 版本及 JAX/Flax/Optax；latest 实验功能另外核验 |
| Sinabs/EXODUS、Lava-DL、Rockpool、BindsNET | 官方发行或源码提交待锁定 | 配对 CUDA/框架依赖与具体学习规则，不从主项目扩展能力 |
| BrainPy | 2.8.2 | classic 兼容参考；新 state 栈另锁 |
| brainstate | 0.5.4 | 同时锁 brainpy-state/brainunit/brainevent/braintools/JAX 等实际依赖 |
| braintrace | 0.2.6 | 在线算法赛道 |
| Jaxley | 0.14.0 | 复杂生物物理边界 |
| mlGeNN | 调研源码 `fca7e2c70df181014c0d0effddadb09ef3b9a91a` | 同时锁 GeNN、CUDA、NCCL；不要假定此 SHA 是最终采用版本 |
| NEST | 官方稳定发行待运行锁定 | 模型、e-prop 示例与 MPI 配置一并锁定 |
| NEST GPU | 发行/提交、CUDA 与 MPI 组合待锁定 | 独立于 CPU NEST 锁定；验证具体可塑性和远程连接组合 |

发行来源：[snnTorch](https://pypi.org/project/snntorch/)、[SpikingJelly](https://pypi.org/project/spikingjelly/)、[Norse](https://pypi.org/project/norse/)、[Spyx](https://pypi.org/project/spyx/)、[BrainPy](https://github.com/brainpy/BrainPy/releases)、[brainstate](https://github.com/chaobrain/brainstate/releases)、[braintrace](https://github.com/chaobrain/braintrace/releases)、[Jaxley](https://github.com/jaxleyverse/jaxley/releases)、[mlGeNN 源码](https://github.com/genn-team/ml_genn/tree/fca7e2c70df181014c0d0effddadb09ef3b9a91a)。

## Atlas 必须分开看的三层

**平台配置边界、当前部署 bundle、引擎研发源码不是同一版本。** 前面对话从平台配置得到的 2–4 状态、4,096 神经元、500,000 候选连接、2/4 ranks 不能作为原生引擎上限。

本次读到的平台当前原生训练 bundle 为 `f6609476fdb99f15afe7e868388db085d420ebf240c80068bcfea8cb95e65762`。18 个文件均通过 manifest SHA256 核对。它只证明本地文件身份与完整性，不证明运行性能，也不是 Linux/CUDA 二进制。

| 观察 | 当前证据 | 评估含义 |
| --- | --- | --- |
| 平台 BPTT 有独立 native plan | `backend/app/training_engine.py` 的 `run_gradient_worker` | 平台训练与原 Brian2 模型导入不是同一入口 |
| 平台准入较窄 | `backend/app/training_gradient.py`、`training_gradient_graph.py` | 引擎评估绕过网页/API 准入；平台可用性另外报告 |
| 当前 bundle 的原生 API 接受更宽配置 | `training.py` 检查每层最多 65,536 单元、2–256 MPI ranks | 这是语法准入上界，不是已成功训练规模 |
| MPI 划分数值工作，仍复制模型/输入/tape | bundle `training_mpi.c`；平台预检也明确写出复制 | 不能预先声称多卡降低单模型状态内存 |
| 每次 `trainer.step` 经过 JSON、文件与子进程；MPI 启动也在其中 | bundle `training.py:119`；`training_cluster_transport.py` | 主成绩计入真实调用成本；纯核另表，不混分母 |
| 已核对的 v3/v4 GPU 路径采用前后向 FP32、主机 optimizer FP64 | 引擎 `src/training/gpu.rs`；还包括 FP64 主参数转 FP32、每样本梯度转 FP64 累加 | 严格对照记录每阶段精度及归约顺序；最终选择的版本/后端重新核验，不能笼统标作全 FP32 |
| 当前 bundle 部分 GPU kernel 每个 sample 对应一个 lane | bundle `training_metal.metal` 明示 correctness-first | 支持 GPU 不能替代 GPU 利用率或速度优势证明 |
| 研发源码已有 Brian2 模型转换 | `lower_brian_training` 的 v3/v4 快照及 `dynamic=True` 的受限 v5 转换 | 不能再写成“尚无 Brian2 → 训练计划”；平台是否暴露该入口另外判定 |
| 原生 v4 支持每层 1–16 状态、每表达式最多 128 SSA 节点 | `NATIVE_TRAINING_V4.md` 与转换器检查 | 不把平台 2–4 状态配置当作整个引擎上限，也不推导任意 HH/多室都能接入 |
| 原生研发目录比部署 bundle 更新 | 相邻引擎仓库能力文档和各阶段冻结验收 | 新能力应作为另一候选版本；历史验收不能证明当前工作树所有哈希相同 |

本地平台来源：[训练入口](../backend/app/training_engine.py)、[准入与内存说明](../backend/app/training_gradient.py)、[集群 transport](../backend/app/training_cluster_transport.py)、[原生 bundle 生成](../scripts/sync_atlas_training.py)。研发引擎的入口与范围见 [Brian 转换源码](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/python/brian2_rust/training_brian.py:134)、[转换契约](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/NATIVE_TRAINING_BRIAN.md:17)、[v4 多状态](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/NATIVE_TRAINING_V4.md:10)、[v5 动态训练](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/NATIVE_TRAINING_DYNAMIC.md:5)。这些绝对路径用于本机审阅；执行时须把源码、文档、运行时及证据归档为带哈希的相对路径制品，运行协议不得依赖另一机器上的这些路径。

**原模型转换已存在，训练产物回到原 Brian 模型的完整验证链仍需补齐。** 当前 Bundle 保存 plan、weights、initial state 和 provenance，训练不自动把参数写回 Brian。已有测试分别验证原始快照前向一致性，以及 native 参数更新和 checkpoint 恢复；训练测试还明确断言原 Brian 权重保持原值。本次没有检出通用回写/export API 或“训练后回灌参数，再用原 Brian 独立重放”的完整验收。因此应把已完成的转换工作计入资产，把回写和端到端重放作为采用成本评估项。[Bundle 与不回写契约](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/python/brian2_rust/training_brian.py:126)、[前向与训练恢复测试](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/tests/test_training_brian.py:40)

已有硬件证据应按下表继承，不能把不同源码的通过数累加，或用早期 CUDA 验收给当前 v5 功能打勾。本次只读审查了记录和定向源码，**未重新核对全部历史冻结哈希与当前工作树相等，也未重新运行这些验收**。

| 历史版本与证据 | 实际执行范围 | 可以支持的结论 | 尚不能支持的结论 |
| --- | --- | --- | --- |
| 2026-10-02 v3 两主机 CPU | 两台物理 Linux 主机；1+1、2+2 MPI ranks；小模型两次 optimizer 更新、独立新进程 CLI 状态重放 | 跨物理主机 CPU BPTT 正确性已有证据，与同一 Linux binary 串行参考最大差约 1e-16 | 非当前 v5、非 GPU、非吞吐/扩展性；不是 Python checkpoint envelope 验收。该次 Rust 1.86 不能算作现行工具链性能成绩 |
| 2026-10-03 v4r3 单 L4 | 238 passed / 45 Metal skipped；其中 45 CUDA（19 单进程、26 MPI）及 193 CPU；2/8 ranks 共享一张 L4 | RK2/RK4、部分 refractory 状态保持、连续训练与新进程恢复的 NVIDIA 正确性 | 非多物理 GPU、非跨机 GPU；不能覆盖后续动态 v5 |
| 2026-10-03 v5r5 索引 GPU | 本地 CPU、真实 Metal、2/8 local MPI；407 passed / 230 NVIDIA-only skipped；637 identities | 受测动态索引动作的本地数值、状态和恢复证据 | CUDA 转换源码检查不是 NVCC 或 NVIDIA 硬件验收 |
| 2026-10-04 Subgroup 前端 / v5r7 runtime | 332 passed / 108 NVIDIA-only skipped；440 identities；原 Brian Cython 另有 10 项通过 | 受测 Subgroup、链接、随机及可塑性组合的本地转换和梯度正确性 | 不覆盖所有组合，也没有新增云/NVIDIA/跨机验收 |
| 2026-10-04 Uniform / v5r7 | 207 passed / 95 NVIDIA-only skipped；302 identities；使用单独保留的新 runner | 受测随机数及受影响模块的本地 CPU/Metal/MPI 正确性 | 不能用旧 runner 复现新算子；没有新版 CUDA 硬件证据 |

记录：[两物理主机 CPU](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/mpi-evidence/training-crosshost-20261002/README.md:1)、[v4r3 单 L4](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/mpi-evidence/training-rk-refractory-20261003/README.md)、[v5r5 索引 GPU](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/mpi-evidence/training-runtime-index-gpu-final-20261003/README.md)、[Subgroup](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/mpi-evidence/training-subgroup-final-20261004/README.md)、[Uniform](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/mpi-evidence/training-uniform-final-20261004/README.md)。当前动态训练文档明确 v5 CUDA 尚缺硬件验证；[跨主机 GPU 文档](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/NATIVE_TRAINING_CROSSHOST_GPU.md:1)仍是待执行方案。

原生训练的容量和并行约束也需要进入协议：

- **MPI 当前是计算 owner 分区，各 rank 仍保留完整模型、输入和 tape。** v5 的 GPU MPI 还采用逐动作同步和 replicated apply，主要目标是正确性；没有据此证明内存下降或强扩展。[MPI 训练约束](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/NATIVE_TRAINING_V3.md:108)、[动态 GPU MPI](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/NATIVE_TRAINING_DYNAMIC_GPU.md:173)
- **未检出 native training 的单进程多线程入口。** 本次定向检查 `src/training.rs`、`src/training/`、`training*.py` 未发现线程池/threads 配置；训练 CPU 单进程与 MPI 多进程应分栏。[README 中的 AOT 线程](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/README.md:440)属于仿真执行器，不能直接用于训练能力声明。
- **native tape 默认 64 MiB，许可上限 1 GiB；JSON 请求上限 64 MiB。** 预算含 native 逻辑数组和部分 GPU/MPI 暂存，不等于峰值 RAM/VRAM。TBPTT 仍保存完整获准序列 tape，现有恢复 checkpoint 也不能当作梯度重计算的内存优化。[默认值](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/python/brian2_rust/training.py:23)、[native 预算上限](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/src/training.rs:379)、[JSON 上限](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/src/train_main.rs:16)、[完整 tape 分配](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/src/training.rs:700)

定向检查训练测试、工具及训练验收记录后，找到独立 Brian/Cython 前向、独立 NumPy/局部 surrogate VJP，以及 CPU/Metal/CUDA/MPI 数值与恢复对照；没有检出 snnTorch、BrainPy、Jaxley 等外部框架的正式训练性能比较。首轮评估应复用这些语义门禁，再建立外部强基线和公开时序任务成绩。

已有 MNIST 3,000 样本、5 epoch、3 seeds 的记录证明平台能学习，但 STDP、BPTT、ANN 的配方与计时范围不同，不进入本轮速度榜。[历史比较脚本及口径](../scripts/mnist_method_comparison.py)

## 推荐的首轮范围

完整竞争格局不意味着第一轮把所有框架、模型、设备做笛卡尔积。

| 赛道 | 必测/优先对手 | 主问题 |
| --- | --- | --- |
| 固定模型 SG-BPTT | Atlas、snnTorch、SpikingJelly、Spyx、BrainPy state/brainstate；Norse 交叉检查，EXODUS 前馈专项、SLAYER 动力学专项 | LIF → 循环/稀疏 → 多状态 → 长序列的执行成本 |
| Brian2 原模型与局部学习 | Brian2 原版、Atlas；CUDA 加 Brian2CUDA/GeNN，分布式加 NEST/NEST GPU | 方程、事件与学习规则保留的正确性及成本 |
| 同任务最佳合法学习方法 | 上述梯度栈 + mlGeNN；BrainTrace/NEST e-prop 按任务纳入 | 同质量下的时间、内存与学习稳定性 |
| 能力边界 | Jaxley/braincell 的 HH/多室；SpikingJelly 深度 SNN | 明确 Atlas 的不支持项，不将替代模型伪装为同模型 |
| 参数拟合边界 | brian2modelfitting、Atlas 对应拟合流程；可按问题加入 Jaxley | traces/spikes 的误差目标、模拟次数、未见刺激预测；同外层优化器比较引擎，换优化方法则另表比较任务成本 |

参数拟合纳入本次广义 Training 竞争边界，首轮 SG-BPTT 核心榜不要求全部拟合方法参赛。拟合低训练误差、参数可辨识性和未见刺激预测应分别报告；SBI 后验估计网络的训练不能作为原神经模型已接入 BPTT 的证据。

首轮真实任务以 SHD 事件序列为主，MNIST 为训练流程检查；合成事件数据用来独立控制宽度、时间长度、连接密度和活动率。SHD 原始数据有 700 通道、20 类、8,156 train / 2,264 test，无官方 validation；划分、时间窗和 binning 必须在任何性能比较前冻结。[数据来源](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)

## 什么结果才足以支持产品决策

以下是本次提出的决策规则，不是领域统一标准，也不是已经实现的优势。

| 结论类型 | 所需证据 | 不足以支持的材料 |
| --- | --- | --- |
| 执行更快 | 同语义、同数值 profile、同设备及同工作量，完整训练 step/任务显著更快 | Atlas kernel 时间除以对手完整 epoch；不同 GPU 的倍数 |
| 能训练更大或更长 | 同模型、同梯度协议和同单设备预算，完成前向、反向和更新；OOM/超时边界完整披露 | 仅构建成功；增加总 GPU 内存；缩短 TBPTT 冒充全 BPTT |
| 分布式有收益 | 物理资源身份、固定全局 batch 强扩展、固定卡数跨机放置、多 rank 更新一致性 | 同一张卡启动多个 ranks；增加 batch 后的吞吐单值 |
| 研究任务成本下降 | 固定质量目标下的总时间、成本、成功率，多训练种子 | 只比较默认参数；失败种子丢弃；看测试集调参 |
| Brian2 迁移成本低 | 预选公开模型、改写清单、前向语义及训练后参数回灌验证 | 少量专门设计样例；兼容百分比分母不公开 |

建议内部“值得产品化”的门槛：在预先声明的一个重要模型族中，至少两个相邻规模取得 ≥2× 的完整 step 加速，或 ≥2× 的可训练时间长度/有效网络规模，且通过同语义正确性与质量门槛。小于此幅度仍可能有商业价值，但需用户工作量/成本证据另行判断。所有未领先、未支持、失败点同等保留。

如果只胜过 Brian2 前向、单核或未优化 Python 循环，不应宣布 Training 引擎领先。如果同模型没有优势，但原模型接入和验证显著省事，可以定位为研究工作流能力。如果稀疏循环、多状态或长序列中出现稳定容量/成本收益，再集中研发与产品叙事。

## 本轮可以冻结和仍需锁定的内容

现在已具体化：对手范围、赛道划分、复杂度梯度、CPU/GPU/分布式拓扑、正确性方法、计时与内存口径、失败分类、统计与宣传边界。详见配套协议和 JSON。

执行冻结还需要：选择 Atlas 的确切源码快照，锁定竞品发行与适配器，生成公共输入/权重/数据划分哈希，完成设备清单，跑正确性资格检查。当前没有这些制品，因此本报告不会把冻结候选写成已完成的基准测试或已证实的引擎优势。
