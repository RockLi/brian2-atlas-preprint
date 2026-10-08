# 原生方程、CUDA 与 MPI 训练

2026-10-05 后续开发：标量 v3 已直接接入 `SimulationTime`、`NormalNoise`、
`UniformNoise`、`PoissonNoise`、逐神经元浮点参数和一维／二维 `TimedInput`。
CPU 与 Metal 开发验证已覆盖独立物理递推、正率 score VJP、零率整样本
one-count weak VJP、参数与初值有限差分，以及 serial／MPI 2／8。
本轮最终冻结回归仍待完成，详见
[本轮记录](mpi-evidence/training-scalar-stochastic-20261005/README.md)。
CUDA 共用内核已通过主机 C++ 语法检查，尚未在 NVIDIA 上验证。

这条路径保留 v3 的模拟输入、阈值参数、公开标量状态布局及浮点执行顺序：
subtract reset 在所有投影之前，zero reset 在所有投影之后。
`clock` 与 `noise_streams` 现在可与 `equations` 一起声明；随机身份包含原
seed／sequence／batch／layer／neuron／tick／stream，carry 和 checkpoint
保留已观测的 Poisson count／rate。旧导入记录的 rate DAG 与 score 均 detached。
`update_timed_input` 可在序列边界替换标量计划的冻结表值，不推进任何运行时计数。
标量 GPU 新上下文必须提供 `b2_train_scalar_context_v1()==1`，旧库明确拒绝。
下面各节保留早期验收版本的历史边界。

后续状态：已加入 [CUDA MPI 与多状态 GPU 实现](NATIVE_TRAINING_V4_GPU.md)；下述早期验收边界以其对应证据版本为准。

此阶段在 v1 dense、v2 循环/共享权重计划上增加三个入口：原生 CUDA BPTT、
CPU MPI 目标神经元分区 BPTT，以及 v3 标量更新方程自动微分。
前向、反向与 SGD/Adam 均由 Atlas 原生执行器计算。Python 只编译计划、启动进程、
传输数组和保存快照；不调用其他 ML/autodiff runtime。

本轮后续已增加 [多状态 v4 与 Metal MPI BPTT](NATIVE_TRAINING_V4.md)。

## 能力与边界

| 能力 | CPU | Metal | CUDA | MPI |
| --- | --- | --- | --- | --- |
| v1 dense、v2 循环/反馈/共享卷积 | f64 | GPU f32 | GPU f32 | CPU；v2 另支持 Metal 目标分区 |
| v3 标量更新表达式 VJP | 原生 | GPU | GPU | CPU / Metal 目标分区 |
| 可训练方程系数、时间常数、阈值 | 支持 | 支持 | 支持 | 唯一参数 owner 更新 |
| SGD/Adam、冻结、参数掩码、边界快照 | 支持 | 支持 | 支持 | 固定 rank 布局 |
| 任意 Brian 方程/事件自动转换 | 未实现 | 未实现 | 未实现 | 未实现 |
| 多状态神经元 | v4 有界子集 | 未实现 | 未实现 | v4 CPU MPI |
| refractory、可学习 delay、pooling | 未实现 | 未实现 | 未实现 | 未实现 |
| GPU MPI BPTT | — | v2/v3，单机实测 | 未实现 | Metal 目标分区 |
| 跨 rank 数恢复 | — | 未实现 | 未实现 | 未实现 |

这里的方程是**每层一个标量离散电压更新表达式**，不是任意 Brian 模型的自动可微编译。
阈值仍为硬阶跃，只有反向使用显式 fast-sigmoid surrogate；输出分类损失、同步事件次序、
reset/detach 与 TBPTT 语义沿用 v2。多状态 AdEx/Izhikevich、随机运算、布尔控制流、
任意外部函数、自定义事件与任意 reset 语句不在这个初始编译子集中。后续直接
标量路径已实现 clock/time、normal/uniform/Poisson、TimedArray 和有界 int32
位运算；也已接入[纯函数表达式](NATIVE_TRAINING_PURE_FUNCTIONS.md)的原生 AD。
各项实现、冻结回归状态与旧版本差异见对应最新证据，不能沿用此处初始验收结论。
输入值已经换算为调用方选定的数值单位；显式表达式入口不做 Brian 单位检查或自动选择积分器。
新增 [Brian Network 转换入口](NATIVE_TRAINING_BRIAN.md) 会执行 Brian 单位检查，
并将支持的单状态 Euler 网络转换到这个计划；仍不支持任意 Brian 模型。

## 方程与可训练参数

```python
from brian2_rust import (
    NativeLIFTrainer, lif_training_plan, dense_training_projection,
    compile_training_equation, neuron_parameter_bank,
)

projections = [
    dense_training_projection(0, 1, 2, 3),
    dense_training_projection(1, 2, 3, 2),
    neuron_parameter_bank(4),
]
# 参数组 2：两个 log_tau 和两个 threshold。tau = .1 + exp(log_tau)。
equations = [compile_training_equation(
    'v*(1-dt/(.1+exp(log_tau)))',
    parameters={'dt': .1, 'log_tau': (2, layer)},
) for layer in range(2)]
plan = lif_training_plan(
    [2, 3, 2], backend='metal', projections=projections,
    equations=equations, threshold_parameters=[[2, 2], [2, 3]],
)
# 如给 weights，须包含三组，长度为 6、6、4；例如最后一组是
# [log(.4), log(.4), 1.0, 1.0]。每个系数只有一个 optimizer slot。
trainer = NativeLIFTrainer(plan)
```

传入 `equations` 选择 `b2-equation-training-plan-v3`。每个非输入层必须提供一个程序。
保留名 `v` 是该 tick 更新前的膜电位；`parameters` 中数值是常数，`(bank, index)`
是原生参数引用。支持 `+ - * /`、字面量指数的 `**`，以及
`exp/log/tanh/sqrt/sin/cos`，最多 128 个 SSA 节点、源表达式最多 8192 字符。
Python 使用受限 AST 转换，不执行 `eval`；native 再校验操作、拓扑引用和参数索引。
native reverse mode 对每一个参数出现位置累加 VJP，可同时共享神经元系数与突触参数。

`neuron_parameter_bank(n)` 创建无突触边的独立参数组，仅 v3 允许。
权重、梯度、mask、trainable 和 Adam moments 仍按参数组保存；禁止为每个使用位置
创建独立 optimizer 更新。时间常数约束可以像示例一样通过表达式参数化。
`threshold_parameters` 的每层条目可以为 `[bank,index]` 或 `None`；`None` 使用固定阈值。
可训练阈值同时包括 spike surrogate 和 reset 对阈值的导数，即使 detach reset，
subtract reset 对显式阈值乘子的导数仍然保留。实际使用的阈值必须为正。
冻结组不阻止梯度传播；非法定义、非有限值或数学定义域错误在状态提交前失败。

前向另存更新前电压，反向按原始值重放表达式以计算 VJP，TBPTT 边界仍截断跨窗口
电压 adjoint。当前保留完整获准序列 tape，不宣称 TBPTT 将内存降为窗口大小。
预算计入表达式元数据、额外电压 tape、GPU staging/设备副本与 MPI 汇总缓冲区；
仍然是逻辑数组预算，不是包括驱动、编译器、JSON 和进程开销的 RSS 上限。

## CUDA 与数值契约

设置 `backend='cuda'`；Linux/Unix 上需要 `nvcc`、兼容驱动和真实 NVIDIA GPU。
构建在 trainer 的临时目录中完成，不安装或替换系统运行时。
CUDA 与 Metal 共用训练内核算术源码，只转换地址空间注解、线程索引和标量函数拼写。
CUDA 禁用 FMA contraction，不启用 fast math；每个 sample 一个 GPU lane，
反传在 lane 内按固定次序累加共享参数梯度，Rust 按 batch 顺序以 f64 汇总并更新 optimizer。
这是正确性实现，不是吞吐或扩展性优化，也没有 CPU 回退。

v3 使用独立的 `b2_train_metal_v3` / `b2_train_cuda_v3` 符号，防止旧动态库忽略方程元数据。
v1/v2 使用原有 v2 符号。所有 GPU 输出须有限，库文件每次执行前检查完整性。

CPU 为 f64，GPU 为 f32，host optimizer 为 f64。硬阈值附近不能保证 CPU/GPU 多步
训练轨迹逐脉冲相同：本轮原始测试中第二次更新出现约 `7.1e-8` 的阈值余量，f32 舍入
改变了隐藏脉冲；相同现象在 Metal 复现。跨精度 VJP 比较使用远离阈值临界点的轨迹，
GPU 内部 checkpoint replay 则要求同后端、同计划、同运行时代码逐值一致。

## MPI 目标分区 BPTT

设置 `mpi_ranks=2` 或 `4`（允许 2..256），`backend='cpu'`。
v2/v3 另可选择 `backend='metal'`，实际验收范围见 v4 文档。
该实现按展平后的目标神经元连续分区：`owner(k)=floor(k*ranks/neuron_count)`。
每条突触前向与边 VJP 由目标 owner 计算；方程、reset、阈值 VJP 由该神经元 owner 计算。
跨 rank 的 source-spike adjoint 在神经元 VJP 前同步。共享参数贡献先按 rank 顺序合并，
再按唯一参数编号分配 optimizer owner；每个参数只更新一次，最后同步权重和 moments。
这是模型计算分区，不是将 batch 分片的数据并行。

当前模型、输入和 tape 数组仍复制到各 rank；该版本没有 rank-local 内存节省、
通信量优化或扩展性承诺。归约通过 native MPI Gather + 有序相加 + Broadcast；
同 rank 布局可重放，改变 rank 数会改变浮点归约分组，因此 checkpoint 明确拒绝迁移。
MPI shim 的源码、编译器、MPI 版本、provider、架构，以及 native binary 与计划均进入快照契约。
所有 rank 在计算前核对请求、native binary 和 shim binary 的 SHA-256。
失败时 abort 全部 ranks；超时杀掉 launcher 进程组；Python 只有在成功 finalize 后提交状态。

macOS MPICH 5 的默认 libfabric `sockets` provider 在独立的最小 Init/Finalize 程序中也
出现退出挂起。trainer 仅为该平台的 MPICH 子进程默认选择已验证的 `FI_PROVIDER=tcp`，
保留调用方显式设置，不修改系统环境。其他平台沿用调用方 MPI 环境。
后续已在 23/24 两台 Linux 验证 1+1 / 2+2 ranks 的小模型方程 BPTT、两步 optimizer
及新进程序列化状态精确重放，见 [跨主机证据](mpi-evidence/training-crosshost-20261002/README.md)。
这不等同于 GPU MPI、规模化验收或跨主机 Python trainer checkpoint envelope 验收。
低层 CLI 可由外部 MPI launcher 启动，但请求、binary 和同一 shim 必须在所有主机相同路径可见。
原有 MPI STDP/CPU+Metal 混合 rank 能力独立保留，不能据此声称本次 BPTT 已支持混合 GPU rank。

## 复现与证据

```sh
CARGO_TARGET_DIR="$PWD/brian2-rust/target-native-training" \
  cargo build --offline --release --manifest-path brian2-rust/Cargo.toml --bin b2-train
export B2_TRAIN_RUNNER="$PWD/brian2-rust/target-native-training/release/b2-train"
export B2_RUNNER="$PWD/brian2-rust/target-mpi-training/release/b2-runner"
export PYTHONPATH="$PWD/brian2-rust/python"
export MPLCONFIGDIR=/tmp/b2-training-mpl
B2_TEST_GPU=1 B2_TEST_MPI=1 .venv/bin/python -m pytest -q \
  brian2-rust/tests/test_native_training.py \
  brian2-rust/tests/test_native_training_graph.py \
  brian2-rust/tests/test_native_training_equations.py \
  brian2-rust/tests/test_native_training_cuda.py \
  brian2-rust/tests/test_native_training_mpi.py
.venv/bin/python brian2-rust/examples/native_graph_training.py \
  --equations --backend metal --steps 80 --checkpoint /tmp/equation-metal.json
```

示例同时更新共享卷积核、循环连接、读出和神经元 log-tau 参数；`--mpi-ranks 2`
选择 CPU MPI，`--backend cuda` 选择 CUDA；`--resume --steps 0` 从新进程恢复验证。
这仍是功能示例，不是公开数据集的准确率结论。
完整结果、各轮失败记录、source/binary fingerprints 与硬件状态见
[验收记录](mpi-evidence/training-accelerators-20261002/README.md)。
