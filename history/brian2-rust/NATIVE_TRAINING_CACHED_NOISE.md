# constant-over-dt 随机子表达式

动态训练前端接入标准 Brian `SubexpressionUpdater`。神经元和突触中标记
`constant over dt` 的子表达式保存为可携带状态，由原有更新器的 slot、order、
name 决定执行顺序。转换不会把缓存表达式在每次引用处展开。

```python
group = NeuronGroup(
    2,
    '''dv/dt=(-v+drive)/ms : 1
       u=rand() : 1 (constant over dt)
       drive=gain*u : 1 (constant over dt)
       gain : 1 (constant)''',
    threshold='v > .6 + .02*u', reset='v = .1*u',
    method='rk2', dt=.2*ms,
)
```

通过 `lower_brian_dynamic_training` 或 `lower_brian_training(dynamic=True)` 转换。
静态 v3/v4 前端仍拒绝这些额外更新器，避免忽略缓存行为。

## 调度与求导

- 缓存更新器沿用 Brian 的标量／向量语句分离。共享神经元缓存先计算一次，
  再写回物理共享值，向量段保留其局部引用。普通缓存按神经元或突触逐项计算。
- 标准默认时机为 `before_start`。改变更新器的 `when` 时，动作按实际网络调度
  执行，例如 `after_groups` 读取积分后的状态，`end` 读取 reset 后的状态。
- 缓存依赖链可以包含浮点、Boolean、时间依赖和已支持的函数。缓存供方程、
  阈值、reset、pre/post 以及 refractory 表达式读取。RK 各 stage 复用本步缓存。
- 每个随机调用点按 seed、sequence、batch、updater domain、entity、tick、stream
  定址；共享段与向量段使用不同的 stream 区间。每个更新器最多 16 个调用点。
  `regular_runner_layout[updater.name]['draws']` 记录缓存更新器的抽样布局。
- 缓存更新是可微动作：固定抽样路径上的状态及可训练幅度参与 full BPTT／TBPTT。
  Boolean／整数控制继续停止梯度。缓存值随 carry、store/restore 保存。
- 延迟事件读取投递当步的缓存值。缓存中的 rand/randn 与事件路径中的直接随机
  调用具有不同语义，后者仍使用已有的稳定发射事件身份。
- 零连接突触不执行逐边缓存抽样。其余结构 mask 与缓存重建使用已有的动态状态
  机制；更新器的注册不依赖存在 spike 或非零连接。

本轮保留原生二进制及 GPU ABI：v4r7／v5r7，不需要新增原生指令。

## Brian 约束与剩余范围

仅接入实际所属对象的标准 `SubexpressionUpdater`，保留模板、活动状态、时钟、
单位、数据类型、预算和函数身份检查。同步更新器须使用网络 timestep。
Brian 本身禁止突触子表达式同时使用 `shared` 和 `constant over dt`；前端不放宽
这个约束。神经元共享缓存可用。

外部输入群体由调用方提供 spike；它自身的标准缓存更新器与积分器一样不进入训练图。
此时没有被训练网络使用的输入缓存不会占用噪声流或改变转换结果。突触若读取输入
群体的可变缓存，仍显式拒绝，避免把一个运行时变量冻结成快照常量。

缓存驱动的布尔／时长 refractory 与直接在 refractory 表达式中调用 rand/randn
不同；后者现已接入，见 `NATIVE_TRAINING_RANDOM_REFRACTORY.md`。
异步连续积分、外部 Python 回调、int64／unsigned／位运算、
Synapses-to-Synapses 端点及新版 NVIDIA 实测仍有缺口。跨机器／多物理 GPU 验证
继续按用户要求暂缓。本轮没有云调用。

## 验证

`test_training_cached_noise.py` 用独立逐步公式、实际 Cython 注入样本与梯度有限
差分检查共享缓存、依赖链、Boolean、时间项、RK、冷暖快照、refractory、可训练
突触初值和 pre/post 复用；覆盖 carry、检查点及 CPU／Metal 本机 2/8 MPI。
`test_training_cached_noise_edges.py` 检查延迟／暖队列读取当步缓存、零连接、
更新位置、缓存驱动 refractory 与非法更新器拒绝。

完整正式回归已通过：344 passed／132 NVIDIA-only skipped／0 failed；另有 16 项独立检查通过。
源码、终态与逐项记录入口：`mpi-evidence/training-cached-refractory-final-20261004/README.md`。
其中本页两个新模块为 49 passed／22 NVIDIA-only skipped；不累计 pilot 或失败运行计数。


## Distinct cached updater clocks (2026-10-05)

The v5 itinerary admits distinct standard SubexpressionUpdater clocks. Actual
Cython and independent physical equations now verify Poisson caches at .4ms /
.2ms or .1ms with .2ms neuron clocks, including threshold/reset consumption,
idle visits after the last main transition, full/split checkpoints and local
CPU/Metal MPI. The original mixed uniform/normal RK2 fixture also agrees at
its exact modified .4ms updater clock. The earlier negative clock test was
obsolete and is replaced by positive execution evidence; inactive, overridden
and static-runner rejection controls remain intact. See
`mpi-evidence/training-poisson-event-replay-20261005/README.md` for exact frozen
sources, the preserved initial failure, complete reruns and acceptance limits.
