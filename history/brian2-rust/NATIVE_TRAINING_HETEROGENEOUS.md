# 异质神经元系数、逐神经元参数与阈值

后续 [时间依赖方程](NATIVE_TRAINING_TIME.md) 已加入 `t`，当前 GPU ABI 为 v4r5；本页记录 v4r4 的验收范围。

2026-10-03，接续 [RK/refractory](NATIVE_TRAINING_RK_REFRACTORY.md)。
`lower_brian_training` 现在支持同一 NeuronGroup 内不同神经元使用不同常量系数，
以及对这些系数逐神经元训练。可训练参数须声明为浮点常量；整数和可变参数拒绝训练。
方程中的 `i` 和 `N` 也可转换。
本轮专注本地实现；跨主机验证按用户要求暂缓，没有新云端作业。

## 使用

例如原先要求整层一致的时间常数和阈值，现在可声明为非共享常量：

```python
equations = '''
dv/dt = (-v + bias + drive*i/N)/tau : 1
tau : second (constant)
theta : 1 (constant)
bias : 1 (constant)
drive : 1 (constant, shared)
'''
# NeuronGroup 使用 threshold='v > theta'、reset='v -= theta'，
# 并显式选择 method='euler'、'rk2' 或 'rk4'。
# 建立网络和连接后，为 tau/theta/bias 赋逐神经元数值。

bundle = lower_brian_training(
    network, input_group=inputs, layers=[hidden, output],
    trainable_neuron_parameters={
        hidden.name: ['tau', 'theta'],
        output.name: ['tau', 'theta'],
    },
)
trainer = NativeLIFTrainer(bundle.plan, weights=bundle.weights)
result = trainer.step(batch_spikes, labels, initial=[bundle.initial_state])
```

这里 `batch_spikes` 的 batch 为 1；多个样本应显式复制完整初态。
`bias` 与 `drive` 未选择训练，保留快照常数；每个神经元的 `tau`、`theta` 各有独立
optimizer slot。即使所有神经元最初取值相同，非共享可训练参数也不会合并为共享值。
学习结果不会自动写回原 Brian 对象。

## 参数与梯度契约

| 声明与选择 | 表示与训练行为 |
| --- | --- |
| `(constant, shared)` 且选择训练 | 一个整层共享参数；各神经元对其梯度累加 |
| `(constant)`、非共享且选择训练 | 连续的逐神经元参数组；神经元 j 使用槽位 j |
| 未选择训练、数值异质的 `(constant)` | 独立参数组，默认 `trainable=False`，保留原数值与 optimizer 状态 |
| 未选择训练、整层同值常量 | 编译为 SSA 常量，沿用既有表示 |
| `i` | 当前完整 NeuronGroup 的层内编号，默认冻结，不可作为可训练 Brian 参数选择 |
| `N` | 当前 NeuronGroup 的大小，编译为常量 |

冻结遵循原有参数组契约：禁止 optimizer 更新，不代表诊断输出中的该参数导数必为零。
所有可训练参数在前向更新、RK 阶段和顺序复位中引用同一槽位；阈值参数同时接收
发放 surrogate 路径和复位路径的梯度。`detach_reset`、full/TBPTT、refractory 离散
开关停止梯度与现有语义一致。

`bundle.provenance['bindings']` 对逐神经元组记录 `kind='neuron_array'`、
`layout='neuron'`、变量名、bank 和是否训练；共享组继续使用原有 `kind='neuron'`。
参数不会占用物理状态槽，因此不减少原有每层最多 16 个状态的额度；refractory
仍额外占一个计时槽。参数组数与内存预算继续受原生计划限制。

## 原生执行与 GPU ABI

新增 `neuron_parameter` SSA 节点，读取 `weights[bank][index + j]`。CPU 和 GPU
均使用层内神经元编号 j，MPI 采用相同编号，不使用 rank 内编号或全网络编号。
原生计划在运行前验证整个连续区间处于参数组边界内，越界不会访问内存或提交状态。

v4 的 `threshold_per_neuron` 按层指定阈值引用是否随 j 递增。所有阈值必须有限且为正，
包括组内最后一个神经元；标志、引用或长度缺失时拒绝执行。Python checkpoint 的拓扑
指纹也包含此字段。完整参数组、optimizer 与神经元状态参与保存和恢复。

GPU 新增 opcode 17，并将 v4 阈值元数据展开为逐神经元引用；v1/v2/v3 元数据保持原有
格式。库 ABI 更新为 `b2_train_metal_v4r4` / `b2_train_cuda_v4r4`，避免旧库解释新布局。
旧二进制/检查点不能直接混用；已有源码与运行时指纹检查继续生效。

## 验证与边界

测试入口 `tests/test_training_heterogeneous.py` 使用 3 个隐藏神经元和 2 个输出神经元，
验证不同层宽和参数寻址。覆盖：

- 独立 Brian 前向、SI 单位、从已运行网络提取快照，以及训练/冻结参数两种模式。
- Euler/RK2/RK4，固定时长 refractory，耦合状态与一般顺序复位。
- 独立 NumPy 局部 surrogate 数值差分：全部可训练参数、全部物理初态、detach/full/TBPTT。
- CPU、Metal、CUDA 及 2/8 MPI ranks 的两段连续训练和全新 Python 进程恢复。
- 参数/阈值越界、缺失引用、末尾阈值为零的原子失败，以及同值非共享参数不合并。

本轮实际执行结果见 [验收记录](mpi-evidence/training-heterogeneous-20261003/README.md)。
CUDA 共用 shader 的源生成已检查，新增 CUDA 测试已准备；本轮没有运行 nvcc 或 NVIDIA GPU。
未来的有界单 L4 入口为 `examples/modal_native_training_v4.py --suite heterogeneous`，
只选择受影响的 v4 GPU 回归与异质参数套件，不自动分配资源。

仍要求常量参数声明、相同 dt、默认调度、静态零延迟的 `v_post += weight`。
外部 namespace 向量、随时间变化的参数 `t`、随机方程、任意复杂阈值、动态突触、
可学习延迟、自定义事件及控制流尚未支持。阈值仍为 `v >` 正常量或单个参数名称；
本扩展放宽的是参数在神经元之间的取值，不改变硬阈值与 surrogate 的导数定义。
