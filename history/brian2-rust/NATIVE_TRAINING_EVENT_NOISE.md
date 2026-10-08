# 随机突触事件、连续方程与 summed 汇总

动态训练前端现在为标准 `on_pre`／`on_post` 中的 `rand()`、`randn()` 分配独立
调用点流，支持临时量重用、可训练突触初值、event-driven 迹变量以及已有的
延迟／结构掩码／检查点接口。连续突触更新和 summed 表达式也使用相同显式
调用点转换；本轮独立模型验收覆盖 Euler 更新与 summed 混合均匀／正态调用。

## 随机事件身份

不能只用“到达 tick + 边编号”抽样：延迟缩短后，来自不同发射时刻的事件可以
同时抵达同一条边。新事件的抽样地址为 seed、sequence、batch、pathway domain、
edge、**发射 tick**、stream。每个被选神经元每 tick 最多发射一次 spike；不同
pathway 使用不同 domain。新事件的发射 tick 由到达 tick 减去该代事件原始延迟
恢复，缩短／压缩 FIFO 时仍保留原始延迟。下一代发射路由采用新的延迟。

从已运行的 Brian 网络导入的 pending 队列不一定保留原始发射时刻。这些条目使用
独立的、快照内稳定的正整数 ID；该 ID 和新发射地址属于不同命名空间，并且
不依赖到达 tick。ID 随队列重建和 store/restore 保存。单个导入 ID 最多对应一个
非零队列槽，原生验证会拒绝同一路径同一条边的重复 ID、错误路由延迟或一个
导入身份的重复投递。

均匀与正态事件分别使用 domain magic `0x4232455655303031` 和
`0x423245564e303031`，再以现有嵌套 SplitMix64 算法混合地址。均匀值使用高 53 位
生成 `[0,1)`；正态值使用 Box–Muller。负发射 tick 通过 u64 环绕表示，和受支持
的非负 tick 范围不相交。旧连续 SDE／定时随机序列保持不变。

GPU 的 Rust 主机层生成按动作／tick／样本索引的噪声表。Metal／CUDA 内核继续
使用现有噪声读取算子和 v5r7 ABI，均匀数转 float32 时仍保持上端开区间。
CPU 与 GPU 不保证逐位相同的浮点运算结果；随机地址不依赖 MPI rank 或动作编号。

## 执行和求导契约

- Brian 语句分析器先物化 pathway 子表达式，再替换随机调用。`u=rand()` 后重复
  使用 `u` 会复用样本；不同调用点则分配不同流。每个动作最多 16 个随机流。
- 新的同步调度 domain 与 neuron、连续 Synapses 和 run_regularly domain 分开。
  连续方程和 summed 表达式以当前 tick 抽样，事件路径以发射身份抽样。
- 随机值自身不求导。固定噪声路径上的状态、乘性幅度、参数和 spike surrogate
  参与 full BPTT／TBPTT。反向传播读取与前向相同的样本。
- 事件写入的 `delay` 保持既有调用边界契约：下一次 `step`、`evaluate` 或
  `gradients` 时重新锁定路由；TBPTT 边界不重设路由。要逐 tick 生效，应逐 tick
  调用，与逐次 `Network.run(dt)` 对照。`update_delays` 保留已发事件的到达时刻。
- 不保证与 Brian 全局随机数生成器在相同 seed 下抽出相同值。实际 Cython 对照
  使用同一组注入样本，检验 Brian 的调度、运算、调用次数与最终状态。

```python
syn = Synapses(source, target, 'w : 1',
    on_pre='u = rand(); w *= .99; v_post += w*u + .01*randn()')
# 连接并配置网络后，用 lower_brian_dynamic_training 转换。
# runner= 或 B2_TRAIN_RUNNER 应明确选择本轮新版二进制。
```

## 验证与使用

新版本机 runner 位于 `mpi-evidence/training-event-noise-final-20261004/b2-train`。
旧 runner 未覆盖，跨平台需要从同一源码重新编译。冻结源码、二进制摘要、完整
测试身份、进程状态和最终汇总位于同目录；以 `verification.json` 为最终验收依据。

`test_training_event_noise.py` 使用独立事件桶与模型公式，覆盖冷／暖快照、pre/post、
同 tick 同边多次抵达、可变延迟、队列重建、独立权重／初态差分、双样本及极值
seed/sequence、结构剪枝和生长、检查点、本机 CPU／Metal 2/8 MPI。非法身份和
live 队列多重投递有拒绝测试。`test_training_synaptic_draws.py` 覆盖连续突触和
summed 的实际 Cython 对照及 full/TBPTT 差分。Rust 单元测试验证事件身份不变量。

## 剩余范围

神经元方程、threshold、reset 的标准显式随机调用已在后续阶段接入，见
[NATIVE_TRAINING_NEURON_NOISE.md](NATIVE_TRAINING_NEURON_NOISE.md)。
constant-over-dt 随机缓存和直接随机 refractory 表达式仍待接入。异步连续积分、任意 Python 回调、int64／
unsigned／位运算以及 Synapses-to-Synapses 端点仍有缺口。新版 NVIDIA 代码未在
本轮实机验收，跨主机／多物理 GPU 按用户要求暂缓。本轮只使用本地资源。

本阶段正式结果：**311 passed / 134 NVIDIA-only skipped / 0 failed**，445 身份、
11 完整模块、289 冻结文件；父进程 session 39928 exit 0 已观察。另有 3 项 Rust
单元测试通过。完整记录见 [验收报告](mpi-evidence/training-event-noise-final-20261004/README.md)。
