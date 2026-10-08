# 原生随机方程训练

2026-10-03，接续 [时间依赖](NATIVE_TRAINING_TIME.md)。本页描述已经接入的神经元
随机微分方程；[随机方程与动态突触的整体目标](NATIVE_TRAINING_STOCHASTIC_SYNAPSE_GOAL.md)
仍在进行中。本页是随机积分器的历史阶段记录；后续已接入
[动态突触](NATIVE_TRAINING_DYNAMIC.md)、[显式神经元随机调用](NATIVE_TRAINING_NEURON_NOISE.md)、
[随机缓存](NATIVE_TRAINING_CACHED_NOISE.md)和[随机 refractory](NATIVE_TRAINING_RANDOM_REFRACTORY.md)。
这些实现仍不代表覆盖任意 Brian 语义。

## Brian 积分语义

转换器直接编译仓库内 Brian 的内置显式随机更新器，不自行替换随机积分公式：

| 方法 | Brian 支持的噪声 |
| --- | --- |
| `euler` | 加性噪声（Euler-Maruyama）；真实乘性噪声按 Brian 规则拒绝 |
| `heun` | 乘性和非对角噪声；遵循 Brian 的 Stratonovich Heun 公式 |
| `milstein` | Brian 的无导数 Milstein 公式，限对角噪声 |

公式依据本仓库 `brian2/stateupdaters/explicit.py`，限制依据该更新器自己的检查。
支持 `xi`/命名 `xi_*`，同一层内同名噪声在多个状态间共享；不同神经元、层、样本、
序列、tick 和不同噪声名独立。每层最多 16 个噪声流，每个更新程序仍受 128 个 SSA
节点预算约束。支持已有的时间依赖、常量/逐神经元系数、物理单位、顺序 reset、
固定 refractory 与 full/TBPTT。RK2/RK4 不被伪装成随机积分器。

```python
eqs = '''
dv/dt = (-v + drive)/tau + sigma*(v+1)*xi/sqrt(tau) : 1
tau : second (constant, shared)
drive : 1 (constant, shared)
sigma : 1 (constant, shared)
'''
# NeuronGroup(..., method='heun')
bundle = lower_brian_training(
    network, input_group=inputs, layers=[hidden, output], seed=7123,
    trainable_neuron_parameters={hidden.name: ['tau', 'sigma']},
)
trainer = NativeLIFTrainer(bundle.plan, weights=bundle.weights)
first = trainer.step(spikes[:, :4], labels, initial=[bundle.initial_state])
second = trainer.step(spikes[:, 4:], labels, initial='carry')
trainer.store('stochastic-training.json')
```

上例初态写法对应 batch 1。转换保存当前物理状态与时钟；它不复制 Brian 自身的
NumPy RNG 状态。因而同一个 seed 不承诺产生与 NumPy 相同的随机序列。前向对照
采用相同的明确正态样本注入 Brian，只替换抽样器，保留 Brian 的积分和事件执行。

## 随机流与反向传播

`noise` SSA 节点读取标准正态样本，更新器自身计算 `sqrt(dt) * normal`。样本通过
原生 Rust counter RNG（v1，嵌套 SplitMix64 finalizer + Box-Muller）由以下地址生成：
`(plan.seed, noise_sequence, batch_index, layer_index, neuron_index, tick, stream_index)`。
流名按字典序绑定 index，实际 Brian 更新器中赋值的先后不会交换流身份。
MPI rank 不参与地址，反向传播不会抽取另一条路径；refractory 不改变后续 draw 地址。

CPU 需要时按地址重算样本；GPU 路径由原生 Rust 生成只读 f32 样本表，再交给共享
Metal/CUDA 状态内核。不存在 Python 随机采样或 Python 自动微分运行时。
样本表的主机/设备内存计入预算。GPU ABI 为 `b2_train_metal_v4r6` /
`b2_train_cuda_v4r6`，避免旧动态库解释新元数据。

梯度为固定抽样路径下的 pathwise 导数：噪声本身停止梯度，噪声幅度、状态依赖的
扩散系数、时间常数以及积分器中间阶段保持求导。硬脉冲继续使用声明的 surrogate；
refractory 的离散门停止梯度；`detach_reset` 和 TBPTT 契约保持原有含义。

## 序列、续跑与恢复

- 新 trainer 的 `next_noise_sequence=0`。一次成功的新训练使用下一序列并推进序列计数。
- `initial='carry'` 保留已提交状态、clock tick 和原来的 `noise_sequence`。
- 显式 `noise_sequence=N` 可重复指定路径；它与时钟的 `start_tick` 相互独立。
- 手动续跑须同时传入旧 `final_state`、`final_tick` 和 `noise_sequence`；仅复制状态
  不足以重建随机路径。batch 下标也是地址的一部分，改变样本排序会改变其随机路径。
- `evaluate`/`gradients` 默认使用下一条新路径但不提交计数；也可以读取 carry。
  失败、只读执行和无效恢复都不推进序列位置。
- checkpoint 保存当前及下一序列，校验计划/seed/运行时；新 Python 进程可继续当前
  路径，也可接着开始下一路径。改变 seed 属于拓扑变更。

## 验证和剩余工作

`tests/test_training_stochastic.py` 包含 Brian 共噪声前向（单位、快照、refractory、
共享和多流耦合）、独立 NumPy 路径差分、原生抽样的地址和分布/相关性检查、双样本
CPU/Metal/CUDA 与 2/8 MPI ranks、分段/恢复、错误边界。实际运行见
[本轮证据](mpi-evidence/training-stochastic-20261003/README.md)。

CUDA 仅完成共享源码生成和测试收集；尚无本版 NVIDIA 实机证据。后续阶段已实现
动态突触的随机状态、pre/post 事件和 VJP，另见
[动态突触实现及验收](NATIVE_TRAINING_DYNAMIC.md)与
[持续进度记录](NATIVE_TRAINING_STOCHASTIC_SYNAPSE_GOAL.md)。本页的神经元随机方程
证据仍只覆盖本页范围，不能替代动态突触的独立验收或整体目标的完成审计。

原生显式 uniform SSA 及同步 `run_regularly` 的 `rand()` 已在后续阶段接入；
旧正态序列不变。详见 [均匀随机数契约与验收](NATIVE_TRAINING_UNIFORM.md)。


### 随机突触调用更新（2026-10-04）

标准 pre/post 路径现已接入显式 rand/randn，并通过稳定的发射身份区分同一 tick
同一条边的多个到达事件。连续突触 Euler 与 summed 的显式随机调用也已接入。
执行契约、独立参照和最终验收入口见 [随机突触事件](NATIVE_TRAINING_EVENT_NOISE.md)。
一般神经元 equation/threshold/reset 的任意显式随机调用仍有前端缺口。
