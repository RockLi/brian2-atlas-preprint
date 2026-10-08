# Brian Network → 原生训练计划

后续状态：已加入 [CUDA MPI 与多状态 GPU 实现](NATIVE_TRAINING_V4_GPU.md)，以及
[RK2/RK4 与一般单状态复位](NATIVE_TRAINING_INTEGRATORS.md)。各次硬件验收以对应证据版本为准。
现已新增 [固定时长 refractory](NATIVE_TRAINING_REFRACTORY.md) 和
[RK2/RK4 与 refractory 组合](NATIVE_TRAINING_RK_REFRACTORY.md)，使用现代计时语义。
已支持 [异质系数、逐神经元参数与阈值](NATIVE_TRAINING_HETEROGENEOUS.md)。
已支持 [时间依赖方程与训练时钟](NATIVE_TRAINING_TIME.md)。
[随机方程](NATIVE_TRAINING_STOCHASTIC.md)阶段的 GPU ABI 为 v4r6；动态突触整体目标仍在进行中。
已新增 `dynamic=True` 的 [v5 动态突触转换入口](NATIVE_TRAINING_DYNAMIC.md)，支持本地 CPU/MPI
的连续/事件驱动状态、pre/post、STDP 和随机边状态。另已接通选择范围内的
[运行时可变链接索引](NATIVE_TRAINING_RUNTIME_INDEX.md)，包括神经元及突触事件中的索引变更。
后续还支持[参数表索引](NATIVE_TRAINING_PARAMETER_GATHER.md)、
[状态依赖不应期](NATIVE_TRAINING_STATE_REFRACTORY.md)和
[Subgroup 突触端点](NATIVE_TRAINING_SUBGROUP.md)，当前原生动态 ABI 为 v5r7。
下文的静态突触限制适用于默认 v3/v4 路径，各阶段验收不能跨源码版本混用。
现已接入[有界纯函数与原生自动微分](NATIVE_TRAINING_PURE_FUNCTIONS.md)：可读源码的
无状态单返回表达式可用于神经元、突触和 regular 动作。新模块 CPU／真实 Metal／
本地 MPI 检查55通过、13 NVIDIA专项跳过；完整新版回归仍待验收。

`lower_brian_training` 将一个已建立连接、可读取数值的 Brian `Network` 转换为单状态 v3 或多状态 v4
训练快照。它读取实际 `NeuronGroup` 方程、namespace、时钟、reset、初态，以及
`Synapses` 的端点、权重和 pathway 调度；不运行网络、不改写 Brian 对象、不切换 Device。
单位检查使用 Brian 自身的 equation/statement 检查器，数值统一转为 SI。

```python
from brian2_rust import lower_brian_training, NativeLIFTrainer

bundle = lower_brian_training(
    network, input_group=inputs, layers=[hidden, output],
    trainable_neuron_parameters={hidden.name: ['tau', 'theta']},
    backend='metal', learning_rate=1e-6,
)
trainer = NativeLIFTrainer(bundle.plan, weights=bundle.weights)
result = trainer.step(
    spike_batch, labels,
    initial=[bundle.initial_membrane.copy() for _ in labels],
)
```

`input_group` 是外部脉冲端口。训练时仍由调用者传入 `batch × time × input_size`
数组；转换器不生成 Poisson 输入，也不执行输入组的内部动力学。
若比较 Brian 原始模型，应传入其实际的 0/1 脉冲序列。
`layers` 必须列出所有非输入神经元组，最后一组是分类输出。
监视器不影响训练语义；额外计算对象、未选中的神经元组、输入组接收入边均拒绝。

## 已支持的转换子集

| 部分 | 约束 |
| --- | --- |
| 神经元 | 1..16 个微分状态，包含 `v`；确定性 Euler/RK2/RK4，随机 Euler/Heun/Milstein；可选 CPU/Metal/CUDA 和 MPI |
| 方程 | 展开 Brian subexpression 后，使用 v3 的标量表达式算子集 |
| 系数 | 有限、时间无关的共享或逐神经元常量；支持 `i/N`；方程和 reset 可读取 `t`，支持 `xi`/命名噪声，遵循各随机积分器约束 |
| 可训练神经元参数 | 必须显式指定且为浮点 constant；shared 用一个槽位，非共享用逐神经元槽位 |
| 阈值 | `v > theta` 或正的无量纲数值；theta 可绑定共享或逐神经元参数 |
| reset | 单状态和多状态均支持有界顺序赋值；一般或混合单状态复位自动使用 v4 |
| 突触 | 一个零延迟 pre-spike pathway，且代码仅为 `v_post += weight` |
| 权重 | 浮点突触参数，单位与目标 v 相同；支持逐边与 `(shared)` 权重 |
| 时钟/调度 | 同 dt、默认 Network schedule、神经元默认 slot/order；含 `t` 时还要求同快照时刻 |
| refractory | 固定非负时间 Quantity，Euler/RK2/RK4；支持状态保持和入边条件写入，计时门停止梯度 |

连接按 Brian pathway 调度顺序排列；每个 pathway 内按来源编号稳定排序，保留同一
来源下的原始连接顺序和参数编号。共享权重只产生一个 optimizer slot。
Euler 更新编译为 `v + dt * rhs`；RK2/RK4 使用 Brian 内置方法生成的积分步骤，
编译为有共享节点的 SSA。可混合不同层的积分方法，仍要求相同 dt。每个最终状态
程序最多 128 节点，超出预算明确拒绝；不静默改用其他积分方法。
subtract reset 沿用 native v2/v3 的算术次序；与 Brian reset-after-synapses 的浮点
结果不承诺逐位一致，独立前向测试使用明确 f64 容差并逐脉冲比较。

`bundle.plan`、`weights`、`initial_membrane`、`initial_state` 是独立快照。多状态训练必须传入完整的 `initial_state`，其布局和复位规则见 [多状态 v4](NATIVE_TRAINING_V4.md)。`provenance` 保存对象名、
参数组到变量的绑定、dt、量纲和 SHA-256。修改 Brian 模型不会改变旧快照，训练也
不会把权重写回原模型。所有原生 store/restore、冻结和 MPI rank 契约继续适用。
参数按 SI 保存；学习率需匹配参数尺度。需要 log-tau 等正值参数化时，可用显式 v3 表达式。

不支持的模型抛出带 `code`、`owner` 的 `TrainingConversionError`，不会被省略后继续训练。
后续静态 v4 支持一维／二维 TimedArray、Poisson、已声明的随机积分器及 refractory，见
[时间输入契约](NATIVE_TRAINING_TIME.md)和对应 v4／SDE 文档。静态突触仍限于
默认调度下的零延迟加法投影；STDP、延迟、额外 runners、缓存和复杂阈值等使用
[动态训练入口](NATIVE_TRAINING_DYNAMIC.md)，并遵守各项支持范围。有界纯函数已支持
原生 AD；任意 Python／目标代码回调、未支持的积分器及自定义调度仍不能任意转换。
因此这仍是有检查的 Brian 子集转换。

静态 v4 Poisson 的 CPU／Metal 使用原静态布局和投影／复位顺序；原始 Brian
Cython 对照覆盖 cold/warm、carry/restore 及时间表率。验证状态和 CUDA 范围见
[Poisson 契约](NATIVE_TRAINING_POISSON.md)。

## 验证

`tests/test_training_brian.py` 使用 Brian 自身的运行时作为独立前向参照：
dimensionless/volt、zero/subtract reset、逐边/共享权重、非排序连接和输出反馈。
另测试 CPU/真实 Metal/真实 MPI 的训练与恢复、单位检查和拒绝边界。
本轮证据见 [Brian 转换验收](mpi-evidence/training-brian-20261002/README.md)。
既有 CUDA v1/v2/v3 内核的 L4 实机验收见
[CUDA 验收](mpi-evidence/training-accelerators-20261002/cuda-final/report.json)。
后续多状态、RK2/RK4 和固定时长 refractory 的 CUDA 转换端到端实测已通过，见
[2026-10-03 单 L4 验收](mpi-evidence/training-refractory-20261003/README.md)；各版本证据分别保留。

多状态与 Metal MPI 后续验收见 [v4 和 GPU MPI](NATIVE_TRAINING_V4.md)。

标准同步 `run_regularly` 的神经元、子群和突触动作现已接入 dynamic=True，
执行和验收范围见 [定时代码训练转换](NATIVE_TRAINING_REGULAR.md)。
