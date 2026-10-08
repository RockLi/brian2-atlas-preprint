# 时间依赖方程与训练时钟

2026-10-05 标量 v3 后续开发也支持显式 `SimulationTime` 和 `TimedInput`，
保持原标量 reset 顺序、模拟投影与 `final_membrane` 状态布局。通过 `clock`
声明 origin／dt，反向与前向读取同一 tick 的时间；CPU／Metal 共用既有精度拒绝
规则与 carry／checkpoint 契约。最终冻结回归状态见
[标量随机与时间扩展](mpi-evidence/training-scalar-stochastic-20261005/README.md)。
下文中的 Brian 自动转换规则与历史版本说明仍按相应路径理解。

后续[随机方程扩展](NATIVE_TRAINING_STOCHASTIC.md)已接入神经元 SDE，
[随机缓存](NATIVE_TRAINING_CACHED_NOISE.md)支持每步复用含时间项的表达式。
本地最新原生 ABI 为 v4r7／v5r7；下文保留本次时间扩展的历史验收范围。

2026-10-03，接续 [异质参数](NATIVE_TRAINING_HETEROGENEOUS.md)。
Brian 训练转换现支持微分方程、展开后的 subexpression 和顺序 reset 中的 `t`。
例如 `dv/dt = (-v + drive*sin(t/tau))/tau`、`a += kick*t/tau`。
Euler、RK2、RK4 可与多状态、固定 refractory 和既有参数训练组合。

## 时间语义

转换保存输入组时钟的 SI 时间 `clock.origin` 与 `clock.dt`；所有选中层必须具有相同
时刻和 dt。已运行过的 Network 从快照时刻开始，不把时间重置为零。
含 `t` 的模型使用 v4，即使只有一个微分状态。时间不占用神经元状态槽或参数组。

原生请求的 `start_tick` 默认为 0，表示相对快照的整数 tick。第 k 步的时间为
`origin + (start_tick + k) * dt`。RK2/RK4 中间阶段使用 Brian 内置更新器生成的
`t + dt/2`、`t + dt` 表达式；同一步的 reset 使用该步起始时间。
反向重放使用相同时间。时间、dt、origin 和 tick 均不求导；`tau`、`drive` 等
显式可训练参数仍通过完整表达式求导。阈值仍限于既有的正数或单个参数引用。

```python
bundle = lower_brian_training(
    network, input_group=inputs, layers=[hidden, output],
    trainable_neuron_parameters={hidden.name: ['tau', 'drive']},
)
trainer = NativeLIFTrainer(bundle.plan, weights=bundle.weights)
first = trainer.step(spikes[:, :4], labels, initial=[bundle.initial_state])
second = trainer.step(spikes[:, 4:], labels, initial='carry')
trainer.store('training.json')
```

上例 batch 为 1；其他 batch 大小需复制完整初态。

- 显式 `initial` 或默认初态从快照 tick 0 开始，每次新序列可重复同一时间区间。
- `initial='carry'` 同时继承上次成功训练的完整状态和 `clock_tick`。
- 手工续跑可传 `initial=previous['final_state'], start_tick=previous['final_tick']`；
  不可同时传 `initial='carry'` 和 `start_tick`。
- `evaluate`、`gradients` 可以读取 carry，但不提交神经元状态或时钟；失败也不提交。
- `clock_tick` 与累计训练工作量 `elapsed_ticks` 分开。重复训练新序列不会意外推进其绝对时间。
- checkpoint 保存时钟位置并校验；新进程恢复后继续使用同一位置。
  修改计划中的时钟属于拓扑变更，必须新建 trainer。

分段与整段运行的前向连续性在冻结参数时验证。分段训练默认在调用边界截断反向传播，
且每次调用分别计算均值 logits、loss 和 optimizer 更新；不承诺其梯度或学习结果
等同于一次完整 BPTT。

## 原生执行与资源边界

新增 `time` SSA 节点（GPU opcode 18），读取执行上下文，不产生时钟伴随量。
CPU/MPI 使用 f64。Metal/CUDA 将每步时间以 f32 表传入，共享同一 shader；
表的主机和设备空间计入内存预算。v4 ABI 更新为
`b2_train_metal_v4r5` / `b2_train_cuda_v4r5`，旧库不可解释新元数据。

原点必须有限且非负，dt 必须有限且为正，tick 和最终 tick 不超过 `2**53`。
溢出、无法区分相邻 tick 的 f64 或 GPU f32 时间均拒绝，不静默停止时钟。
旧检查点继续受二进制和源码哈希约束，不跨运行时版本直接混用。

## 验证与剩余范围

`tests/test_training_time.py` 覆盖独立 Brian 对照（SI 单位、预运行快照、仅 reset
依赖时间）、单状态闭式解、独立 NumPy surrogate 有限差分（全部参数与物理初态，
full/TBPTT、detach/非 detach）、CPU/Metal/CUDA 与 2/8 MPI ranks、分段和新进程
恢复、非法时钟与失败原子性。实际执行结果见 [本轮验收](mpi-evidence/training-time-20261003/README.md)。

CUDA 源码生成与测试入口已准备；本轮没有启动云资源，未运行 nvcc 或 NVIDIA GPU。
跨机器仍按用户要求暂缓。

解析形式的确定性时间依赖已经接入。后续阶段也已实现随机方程、动态突触状态、
STDP、延迟和动态 TimedArray；各自范围和验收见对应训练文档。
[有界纯函数 AD](NATIVE_TRAINING_PURE_FUNCTIONS.md)也已接入，包含标量 SI 闭包常量。
任意 Python／目标代码回调、随时间变化的任意阈值表达式和完整 Brian 语义仍未全部覆盖。


## 静态 v4 TimedArray（2026-10-05）

静态多状态前端可直接使用一维、二维 TimedArray；即使调用只有字面时间，
也选择带时钟的 v4 计划。共享 TimedArray 对象的不同名字仍映射到一个物理参数 bank，
保留别名来源。更新器、RK 中间阶段和 reset 均使用原采样网格，表外时间钳制到首末行；
列索引必须为有效整数。时间和列索引不求导，选中表值、普通参数及全部初态求导。
实际 MPI 的局部列索引错误先跨 rank 同步，再退出，避免后续集合操作等待。

CPU、Metal 与 CUDA 共享节点和元数据布局。静态 GPU 库必须声明
`b2_train_static_timed_input_v1()==1`，否则明确拒绝；每个节点的四字描述符及
主机／设备副本计入预算。GPU epsilon 必须在 f32 下为正，维度不超过 `2**24`。
CUDA 共享源码与能力拒绝控制已提供；本阶段没有 NVIDIA/NVCC 实测。

`update_timed_input` 也支持静态 v4。只可在调用边界替换冻结 bank 的同形有限值，
保留完整神经元状态、optimizer moments、时钟和随机序列；不执行训练轨迹。
已观察 Poisson count/rate 原样保留，重读同一身份不因新表值重新抽样或评分。
预算合计包含保留的抽样缓存；任何检查失败都不提交。

静态时间表驱动 SDE 的 Euler（合法分段常数采样网格）、Heun、Milstein 和
CPU Poisson 正率／零率 weak VJP 组合也纳入本阶段验证。
确切结果及重放程序见 [静态时间输入证据](mpi-evidence/training-static-timed-input-20261005/README.md)。
后续静态 v4 原生 GPU Poisson 路径已接入时间表值率、正率 score、零率完整
样本重放和保留旧观察值的表修改，见
[本轮 GPU 验证记录](mpi-evidence/training-poisson-static-gpu-20261005/README.md)。
静态 v3 时间表／Poisson 和其余 Brian 语义仍未完成，NVIDIA 实机未验收。
