# 随机 refractory 表达式

动态训练转换已接入 refractory 中的标准 `rand()`／`randn()`。以下两种 Brian
表达式均可使用：

```python
refractory='randn() > 0'                  # 布尔释放条件
refractory='(1 + int(3*rand())) * dt'      # 每步重新计算的时长
```

使用 `lower_brian_dynamic_training` 或 `lower_brian_training(dynamic=True)`。
前端先检查原表达式的单位和 Boolean 类型，再给直接随机调用分配独立流。原表达式
保留在转换来源记录中，`neuron_draws[layer]['refractory']` 给出调用类型和流号。
命名 `xi` 流先分配，随后为 refractory、threshold、积分 stage 和 reset 调用点分配
剩余流；每层总计最多 16 个流。原来没有随机 refractory 的模型保留原有抽样地址。

## 运行语义

布尔 refractory 遵循 Brian 的锁存语义：活动神经元继续活动；进入 refractory
后，在表达式变成 false 时释放。条件保留短路，不通过随机条件改变已活动的状态。
时长 refractory 按当前 tick 重新求值，使用 Brian 的时间步舍入规则，与最后一次
spike 时间比较。时长不是在发放时预先抽取后固定保存。

每个调用点使用 seed、sequence、batch、layer、neuron、tick、stream 定址。
RK 积分 stage 共用本次 refractory 判断，方程中直接调用 rand/randn 的各 stage
仍有自己的调用点。原生路径允许预计算不会用到的计数随机样本；有效状态转移与
Brian 短路一致，并不要求消耗相同的全局随机数发生器序列。

refractory 是硬控制条件，不对随机释放选择或离散时长求导。固定抽样及控制路径
上的连续状态、可训练参数与 surrogate spike 梯度继续参与 full BPTT／TBPTT。
activity、lastspike、随机 tick／sequence 随 carry、store/restore 保存。

## 验证与边界

`tests/test_training_random_refractory.py` 使用独立逐步公式、固定控制的有限差分、
真实 Brian Cython 样本注入验证两种表达式、Euler／RK2／RK4、冷暖快照、权重和
初态梯度、检查点及本机 2/8 MPI。CPU 和实际 Metal 在本地运行；CUDA 用例保留，
没有 NVIDIA 资源时明确跳过。

本阶段正式回归位于 `mpi-evidence/training-cached-refractory-final-20261004/`，
终态验收已完成：344 passed／132 NVIDIA-only skipped／0 failed，另有 16 项独立检查通过。
本页新模块为 40 passed／20 NVIDIA-only skipped。`check_sde_ref.py` 补充命名 xi 与直接随机 refractory
组合的 Euler／Heun／Milstein、独立梯度和真实 Cython 对照。

这项实现复用既有 v4r7／v5r7 原生执行器。异步连续积分、外部 Python 回调、
int64／unsigned／位运算及 Synapses-to-Synapses 端点尚未覆盖。新版 NVIDIA
实测也仍待完成；跨主机／多物理 GPU 验证按用户要求暂缓。本阶段没有云调用。
