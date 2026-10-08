# 多状态原生自动微分与 Metal MPI BPTT

后续状态：已加入 [CUDA MPI 与多状态 GPU 实现](NATIVE_TRAINING_V4_GPU.md)；下述早期验收边界以其对应证据版本为准。

2026-10-02。本轮实现两个独立入口：CPU/CPU MPI 的多状态 v4 计划，
以及 Metal 上 v2/v3 投影计划的目标分区 MPI BPTT。前向、反向和参数更新均使用
原生执行器；Python 负责转换和进程协调。

## 多状态 v4

`b2-state-training-plan-v4` 为每层声明 1..16 个状态，编号 0 是电压 `v`。
不同层可以具有不同状态数。`state_equations[layer][state]` 保存离散更新 SSA，
所有更新同时读取该神经元上一时间步的完整状态向量。
`state_resets[layer][state]` 保存脉冲发生后的最终状态表达式；全部复位表达式
读取突触输入累加后的同一份状态。v4 的复位语义由这些程序决定，旧 `reset`
字段不参与 v4 计算。

```python
from brian2_rust import lower_brian_training, NativeLIFTrainer

# hidden/output 可以有 v、a、c 等相互耦合的确定性微分方程。
bundle = lower_brian_training(
    network, input_group=inputs, layers=[hidden, output],
    backend='cpu', mpi_ranks=2,
    trainable_neuron_parameters={hidden.name: ['tau', 'kick', 'theta']},
    learning_rate=1e-7,
)
trainer = NativeLIFTrainer(bundle.plan, weights=bundle.weights)
result = trainer.step(spike_batch, labels,
    initial=[bundle.initial_state.copy() for _ in labels])
trainer.store('/tmp/multistate.checkpoint')
# initial='carry' 延续全部状态，而非只延续 v。
next_result = trainer.step(spike_batch, labels, initial='carry')
```

Brian 转换仍要求显式 Euler、相同 dt、默认调度、静态零延迟加性突触。
Euler 子表达式会展开，SI 单位检查保留。状态按 **layer → state → neuron** 展平；
每层状态名称顺序为 `v`，再按字典序排列其余微分状态，记录在
`bundle.provenance['state_names']`。`bundle.initial_state` 是完整状态快照，
`initial_membrane` 仍只包含电压。直接构建计划时，可使用
`compile_training_equation(expr, states=['v', 'a'], parameters=...)`。

多状态 Brian reset 支持对微分状态的 `=`、`+=`、`-=`、`*=`、`/=`，表达式使用
现有有界算子集。转换器按语句顺序进行代换，再生成同时执行的最终表达式，例如
`a += kick + .1*v; v -= theta; c = .9*c + .05*a` 中 `c` 读取已更新的 `a`。
更新与复位参数可绑定同一优化器槽；参数声明仍须 `(constant, shared)`。
每个程序最多 128 个 SSA 节点，复位代换也有大小上限。

反向顺序是：复位 VJP → 突触 VJP → 阈值 surrogate → 耦合状态更新 VJP。
`detach_reset=True` 仅切断脉冲门的导数，保留复位表达式自身对状态和参数的导数。
TBPTT 在边界切断所有状态的 adjoint。输出新增 `final_state` 和
`initial_state_gradients`；原有 `final_membrane` / `initial_gradients` 只返回电压分量。
快照保存完整状态并锁定状态程序、运行时、MPI 布局；失败不提交部分结果。

CPU MPI 将一个神经元的所有状态放在同一计算 owner，归约跨 rank 的脉冲 adjoint
和共享参数梯度；参数仍由唯一 owner 更新。内存预算在分配前覆盖两份向量 tape、
实时状态、反向暂存、SSA 和 MPI gather 工作区。

## Metal MPI

对 v2/v3 显式投影计划设置 `backend='metal', mpi_ranks=2` 或 `4`。
Brian 单状态转换也可直接传入这两个选项。MPI rank 数合法范围是 2..256，
已测试 2/4/8；不要求每个 rank 都拥有神经元。

GPU kernel 只计算属于当前 rank 的目标神经元、入边及相应 VJP。
Metal command 完成后，原生 host 通过 MPI 同步脉冲、膜电位、源脉冲 adjoint 和
跨时间步 carry，再进入下一 GPU 阶段。通信缓冲区从 f32 转为 f64 做固定 rank
顺序归约，然后转回 f32；参数梯度和 host optimizer 使用 f64。
每个训练请求每 rank 调度 `3+4*time` 次，evaluate 为 `3+2*time` 次。

独立动态库符号 `b2_train_metal_mpi_v1` 防止旧库忽略 MPI 元数据。
各 rank 在计算前额外核对 GPU 动态库 SHA-256。GPU shader、MPI shim、native
binary、计划和运行时身份参与检查点约束。

这次实测是 **一台 Mac 上多个 MPI 进程共享一个 Metal GPU**，没有跨主机 GPU、
多物理 GPU、通信性能或加速比的验收。数组和 tape 仍在每个 rank 复制，
该实现不承诺内存随 rank 数下降。不同 rank 数会改变浮点归约分组，不能跨布局恢复。

## 当前边界

- v4 尚不支持 Metal/CUDA，多状态 GPU 与 GPU MPI 的组合入口会明确拒绝。
- CUDA MPI 尚未实现；本轮没有重新启动收费 L4 作业。既有 CUDA shader 和构建源
  保持上一轮版本，但本轮新 native binary 没有做 NVIDIA 实机回归。
- 多状态转换并不等于任意 Brian 模型：随机方程、refractory、自定义事件/调度、
  布尔分支、自定义函数、异质参数、可学习 delay、动态突触均未涵盖。
- 阈值仍为 `v >` 正标量或共享可训练参数；突触仍只写 `v_post`。
- 不带 detach 的复位 surrogate 会在反向计算未发放神经元的复位表达式值；
  表达式在该轨迹上也必须有定义，否则明确报域错误。

## 验收

本轮合并回归 **169 passed, 22 skipped**，其中新增 42 项全部通过；22 项为本机不可用的 CUDA 用例。

新增独立数值差分覆盖两层不同状态数、所有初始状态和参数梯度、耦合更新、
顺序复位、两种脉冲门导数及 full/TBPTT；另用 Brian 运行时对照无量纲/volt 模型。
MPI 检查 2/4/8 ranks、两次优化器更新、空 owner，以及包含全部状态的新进程精确恢复。
Metal MPI 与 CPU、单进程 Metal 对照循环/反馈/共享参数和 v3 方程，检查冻结、
mask、失败原子性与新进程恢复。

完整日志及源代码指纹见 [本轮验收记录](mpi-evidence/training-gpu-mpi-20261002/README.md)。
