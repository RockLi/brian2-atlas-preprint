# 状态依赖 refractory 与精确时间比较

2026-10-04。动态训练入口 `lower_brian_dynamic_training`（或
`lower_brian_training(..., dynamic=True)`）增加字符串 refractory 表达式。
固定 Quantity refractory 继续使用原路径。本轮12模块本地验收为871通过、262 NVIDIA跳过、0失败；记录见
[冻结回归](mpi-evidence/training-state-refractory-final-r2-20261004/README.md)。

## 语义

- 布尔条件，如 `refractory='u > limit'`：每步积分前更新
  `not_refractory = not_refractory or not(condition)`，符合 Brian 的重新激活规则。
- 时间表达式，如 `refractory='r + shift'`：每步读取当前时长，与距上次脉冲的
  离散步数比较。因此处于不应期时修改 r 可以改变结束时间。
- 神经元积分、阈值、突触 pre/post 和 reset 保留 Brian 调度顺序；
  `(unless refractory)` 的神经元写入受活动标志约束，突触自身的可塑性更新继续。
- 支持冷启动和已运行网络的快照、显式 `not_refractory` / `lastspike` 引用，
  以及条件中通过运行时索引读取链接变量。表达式必须返回 Boolean 或时间量纲。

控制状态包含活动标志、距脉冲的 int32 步数、普通浮点 lastspike，以及保存 lastspike
完整 binary64 位模式的两个 int32 槽。步数在 INT32_MAX 饱和；脉冲发生后写成 1，
供下一步使用。控制状态随 carry、store/restore 和 MPI 状态同步一起保存。
时长换算遵循现代 Brian timestep 的 `int((duration + 0.001*dt)/dt)`；超出 int32
范围会拒绝，不能静默溢出。旧式 refractory timing 仍不支持。

## 时间精度

CPU 使用 binary64；Metal/CUDA 普通模型状态和梯度仍使用 float32。
简单的 `t-lastspike` 比较（`< <= == != >= >`，两侧顺序均可）由前端生成专用
`elapsed_compare` 节点。字面量及基本常量算术形成的比较阈值保留 binary64；动态
右侧表达式继续使用后端模型精度。设备以整数运算执行 binary64 相减和比较，
避免 `.0014-.0008 < .0006` 在 float32 中改变硬门决策。

`time_word` 保存时钟的两个整数位片段；`elapsed_compare` 检查其类型、索引和
有限性，并遵循惰性 Boolean/select 分支。GPU 原始时间取自与 CPU 相同的时钟样本，
无需主机计算每个神经元的门。动态 GPU ABI 升至 `v5r7`，旧库会明确拒绝。
检查点的运行时指纹随实现变化，不能假定旧版本检查点可跨版本恢复。

这项精确路径不代表所有任意时间算式都自动具有 binary64 GPU 精度。例如时间
经非线性函数或其他组合运算后比较，仍遵循普通表达式的数值精度。

## 梯度与验证

不应期门、持续时长的离散决策和时间戳停止梯度；活动时的模型状态、连续参数、
动态突触与 surrogate 阈值继续参与 BPTT。`detach_reset` 和 TBPTT 契约不变。
不会把结束时间的离散变化解释成连续导数。

新增测试覆盖实际 Brian Cython 前向对照、Euler/RK4、固定噪声下全部浮点参数和
初态的独立有限差分、optimizer/carry/checkpoint、条件链接及非 root 错误回滚。
精确时钟测试覆盖六种比较与相邻浮点边界、负间隔、signed zero、subnormal、
惰性错误隔离、时间戳持久化、旧 ABI 拒绝及 CUDA 源码转换；同一整数算法另与
硬件 binary64 比较一百万组随机输入。后者是 CPU 算法检查，设备证据来自实际
Metal 测试。MPI 为本机 2/8 进程，非多物理 GPU 或跨主机证明。

更新的 NVIDIA/NVCC 实机验收仍未运行。跨主机按用户要求暂缓；整体随机方程与
动态突触目标仍需后续组合及广泛 Brian 语义验收，不能据此宣称任意模型全部完成。

后续[联合随机模型验收](NATIVE_TRAINING_STOCHASTIC_COMBINATIONS.md)已覆盖上述
不应期与神经元/突触乘性 SDE、summed、event-driven 和 mutable-delay 的同时使用；
详细范围以该记录为准，不表示任意组合均完成。
