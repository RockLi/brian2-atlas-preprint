# CUDA MPI 与多状态 GPU BPTT

后续前端扩展：[RK2/RK4 与一般单状态复位](NATIVE_TRAINING_INTEGRATORS.md)。
该扩展的本机与后续 CUDA 验收另有记录，本页 L4 结果仍对应 2026-10-02 源码版本。
另见 [固定时长 refractory 与 v4r2 GPU ABI](NATIVE_TRAINING_REFRACTORY.md)；旧 L4
报告不覆盖此后的 GPU 元数据和计时器改动。新 v4r2 的单 L4 实测已通过，见
[2026-10-03 验收](mpi-evidence/training-refractory-20261003/README.md)。
最新 [RK/refractory 组合](NATIVE_TRAINING_RK_REFRACTORY.md) 使用 v4r3，已通过本地和单 L4 验证；
[跨主机／多物理 GPU 方案](NATIVE_TRAINING_CROSSHOST_GPU.md) 按用户要求暂缓。
[异质神经元参数](NATIVE_TRAINING_HETEROGENEOUS.md) 使用 v4r4；
[时间依赖方程](NATIVE_TRAINING_TIME.md) 使用 v4r5；
当前 [随机方程](NATIVE_TRAINING_STOCHASTIC.md) 使用 v4r6，旧 L4 结果不覆盖此后修改。

2026-10-02，接续 [v4 / Metal MPI](NATIVE_TRAINING_V4.md)。本轮把 v4 耦合状态更新、
顺序复位与完整状态 VJP 接到 Metal/CUDA GPU，并为 CUDA 接入逐阶段的目标分区 MPI。
实际 CUDA 验收状态以[本轮记录](mpi-evidence/training-v4-gpu-20261002/README.md)为准。

## 使用

Brian 转换入口和直接构建的 v4 计划都可选择 `backend='metal'` 或 `'cuda'`；
添加 `mpi_ranks=2` 选择对应 GPU MPI。单状态 v2/v3 的 CUDA MPI 使用相同入口。
GPU MPI 要求显式投影计划；v1 dense 可先用 v2 dense projection 表达。

```python
bundle = lower_brian_training(
    network, input_group=inputs, layers=[hidden, output],
    backend='cuda', mpi_ranks=2,
    trainable_neuron_parameters={hidden.name: ['tau', 'kick', 'theta']},
    learning_rate=1e-7,
)
trainer = NativeLIFTrainer(bundle.plan, weights=bundle.weights)
result = trainer.step(spike_batch, labels,
    initial=[bundle.initial_state.copy() for _ in labels])
trainer.store('/tmp/vector-gpu.checkpoint')
# 后续 trainer.step(..., initial='carry') 和恢复均携带全部状态。
```

状态、方程、复位、单位和 capability gate 沿用 v4；此次没有扩展到任意 Brian 语义。
不同层可有不同状态数，每层最多 16 个状态、每个表达式最多 128 个 SSA 节点。
支持有界非线性表达式、共享参数、反馈、冻结、mask、full BPTT 与全部状态的 TBPTT。
完整输出使用 `final_state` 和 `initial_state_gradients`；电压子集字段保持原有含义。

## GPU 和 MPI 执行

所有状态方程与 VJP 都在选定 GPU 运行。GPU 更新同时读取完整旧状态，复位读取
突触累加后的完整状态。反向先处理复位，再处理边和脉冲 adjoint，最后传播耦合更新。
独立状态 shader 同时用于 Metal 和 CUDA，后者只转换地址空间、线程入口和数学拼写。

v4 单 GPU 和 GPU MPI 都使用分阶段调度：每次训练请求 `3+4*time` 次、evaluate
为 `3+2*time` 次。单 GPU 跳过通信，GPU MPI 在阶段完成后归约。
目标神经元 owner 同时负责其全部状态和入边，共享参数由唯一 optimizer owner 更新。

CUDA 通过 `cudaDeviceSynchronize` 确认 kernel 完成，再把通信片段复制到 host，
调用现有 MPI 有序 f64 归约，转回 f32 并写回设备；不要求 CUDA-aware MPI。
Metal 使用 shared buffers 完成同样的阶段边界。两者都没有 CPU 计算回退。
GPU 前向/VJP 为 f32，host optimizer 为 f64；跨精度和跨 rank 数可能改变舍入分组。

新增 ABI 符号 `b2_train_cuda_mpi_v1`、`b2_train_cuda_v4`、`b2_train_metal_v4`。
v4 元数据使用独立状态宽度、每层布局和更新/复位程序索引；host 在分配前计算完整
向量 tape、设备缓冲区、暂存和 MPI 通信预算。中间域错误会保留错误标志，即使后续
时间步重新变为有限值，也不会提交成功结果。CUDA 资源使用 RAII，异常转换为原生错误。

CUDA 默认使用可见的逻辑设备 0。可以通过 launcher 为各 rank 设置
`CUDA_VISIBLE_DEVICES`，或设置 `B2_TRAIN_CUDA_DEVICE` 选择可见设备编号；非法编号
明确拒绝。默认多 rank 可以共享一张 GPU，当前没有自动多 GPU 绑定策略。

## 验证与限制

本机合并回归 **192 passed, 64 skipped**；64 项均要求本机不具备的 NVIDIA CUDA。
新增 Metal 测试覆盖 2/3 状态混合层、16 状态上限、非线性、两次更新、所有状态初态梯度、
2/8 MPI ranks、空 owner、freeze/mask、故障原子性、Brian 端到端以及新 Python 进程恢复。
前一轮 CPU 独立数值差分和 Brian 运行时对照继续包含在回归中。

CUDA 实机验证入口为 `examples/modal_native_training_v4.py`：一个 L4、最多 15 分钟、
一个容器、无自动重试。测试进程组超时会整体终止；只上传源码与选定测试，不上传
凭据、Git 数据、训练数据集或本地动态库。

最新 L4 实测 **100 passed, 64 skipped, 0 failed**，跳过项均为 Metal 用例。
其中实际 CUDA **64 项通过**（33 项单进程、31 项 MPI），另有 36 项 CPU 通过。
CUDA MPI 覆盖 2/4 rank 梯度与 optimizer 对照、8 rank 空 owner；v4 多状态 GPU
23 项通过，包含新 Python 进程 checkpoint 精确恢复、故障原子性和 Brian 端到端。
测试耗时 622.72 秒，总耗时 682.78 秒；作业已结束，无自动重试。

首次作业因原 Ubuntu MPICH 镜像的启动问题出现 42 项 rank mismatch 失败，记录保留。
修复后的入口固定 `mpich==5.0.1`，并在分配 GPU 前的 CPU 镜像构建阶段，校验实际
shim 的 2/4 rank 身份、归约及原生训练与串行一致性。本机和 Linux 预检均通过；
用户另行明确授权后完成上述 L4 验收，两次作业之间运行时代码和训练测试未改动。
详细日志、源文件指纹和验收边界见本轮记录。

本轮不包含跨主机 GPU、多物理 GPU、性能加速比或大模型扩展性验收。
所有 rank 仍持有完整模型/输入/tape，不能据此声称分布式内存节省。
随机方程、refractory、可学习延迟、任意事件/控制流和异质系数等 Brian 语义仍未涵盖。
