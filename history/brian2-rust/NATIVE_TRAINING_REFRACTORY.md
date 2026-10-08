# 固定时长 refractory 的原生 BPTT

后续扩展：2026-10-04 已加入[状态依赖 refractory 与精确时间比较](NATIVE_TRAINING_STATE_REFRACTORY.md)。以下为对应历史阶段记录；最新本地验收以扩展文档链接为准。

2026-10-03。Brian 转换和 v4 原生执行器增加固定时长 refractory。
本文描述 Euler 基础实现（v4r2），已通过 CPU、Metal、MPI 和后续单 L4 验收。
最新扩展为 [RK2/RK4 与 refractory 组合](NATIVE_TRAINING_RK_REFRACTORY.md)，使用 v4r3 ABI；
最新源码的验证状态以该扩展记录为准，旧 L4 结果不能覆盖新增 opcode。

## Brian 转换

使用 `NeuronGroup(..., method='euler', refractory=3*ms)`，并按需为微分方程添加
`(unless refractory)`。转换器仍要求相同 dt、默认调度、静态零延迟加性突触。
时长须为有限非负、带时间单位的标量 Quantity；按 Brian 的 `timestep(duration, dt)`
转换为整数步数，沿用现代 refractory timing。测试覆盖零时长、整步和非整步时长。

有 refractory 的层会在完整状态末尾增加 `__refractory_ticks`，占一个状态槽，
因此最多 15 个物理微分状态。无 refractory 的层仍最多 16 个状态。
转换从实际时钟和 `lastspike` 快照计算计时状态，支持从已经运行的 Brian 网络接续。
计时器与全部物理状态一同参与 `initial='carry'`、store/restore 和 MPI owner 分区；
调用者应传 `bundle.initial_state`，不能只传 `initial_membrane`。

## 调度与导数

每步开始，计时器为零的神经元可以发放。带 `(unless refractory)` 的状态只在
此时更新，其余状态继续 Euler 更新。计时器每步递减至零；发放后写入
`max(refractory_steps-1, 0)`，以便下一步使用同一离散时间约定。

如果 `v` 带 `(unless refractory)`，处于 refractory 或刚在当前阈值阶段发放的
神经元都屏蔽本步入边写入。后者即使 refractory 时长为零也成立：Brian 在阈值阶段
就把 `not_refractory` 置为 false，突触阶段位于其后。复位随后照常执行。

refractory 计时器及事件条件写入门采用 **固定离散轨迹、停止梯度** 的语义。
处于 refractory 的阈值 surrogate 为零；被屏蔽入边的权重/源脉冲梯度为零；
保持不变的物理状态沿恒等映射传播梯度。正常状态更新、复位和参数继续原生 VJP。
`detach_reset` 仍只决定复位脉冲门的 surrogate，计时器初态梯度始终为零。
不对时长、事件时间或 refractory 结束时刻求导。

## 原生计划与 ABI

v4 可添加每层一个可空条目的 `refractory` 列表。例如 `{'steps': 3, 'clamp': [0]}`
表示三步时长、保持电压状态。该层最后一个状态必须是专用计时器；更新/复位程序
为它声明 identity placeholder，执行器负责离散递减和复位。其他可微程序不能读取
该计时器；clamp 不能包含计时器，初态必须是范围内的非负整数。步数上限为 2^24，
保证 f32 计时值精确表示。检查在训练前执行，失败不提交部分权重或状态。

GPU 每层元数据从四项扩展到六项，包含时长标志和保持状态位掩码；native 预留完整
容量并按新偏移提取电压输出。动态库使用 `b2_train_metal_v4r2` /
`b2_train_cuda_v4r2`，避免旧 v4 库错误解释新布局。既有检查点会按原生运行时和
动态库指纹拒绝跨版本恢复，不能把旧二进制/检查点视为可无缝替换。

## 验证与边界

测试入口为 `tests/test_training_refractory.py`，证据见
[本轮记录](mpi-evidence/training-refractory-20261003/README.md)。
独立 Brian 运行时比较脉冲、最终物理状态和计时状态；独立 NumPy 局部 surrogate
数值差分检查全部权重、共享神经元参数及物理初态梯度，覆盖保持组合、detach 和 TBPTT。
另有 2/8 ranks、空 owner、CPU/Metal 对照及包含非零计时器的新 Python 进程恢复。

条件式或表达式时长 refractory、可训练时长、legacy timing 仍明确拒绝。
Euler 基础版本曾拒绝 RK2/RK4 组合；现已通过上述后续扩展支持。
随机微分方程、动态突触、异质系数、自定义事件/调度以及任意 Brian 语义仍未覆盖。
本轮单 L4 CUDA 实机验收已完成：113 passed、33 skipped（Metal）、0 failed，
其中 33 项实际 CUDA（18 项 MPI），包括 RK 前端和 refractory 恢复。
跨主机 GPU、多物理 GPU 验证仍未完成。

准备后的云入口可使用 `--suite refractory` 选择 v4 GPU 回归、RK 前端及 refractory
套件，保留 CPU MPI 构建预检、单 L4、900 秒函数上限、720 秒测试超时和 `retries=0`。
由于 Modal 抢占可独立触发输入重放，入口另加跨容器原子单次执行锁，在替换容器中
拒绝再次运行测试，并实时输出逐项日志。正常执行路径已通过本次云端验收；
重复拒绝与超时终止通过 4 项本地测试，本次未注入平台抢占。
它无法阻止平台短暂分配替换容器；再次执行仍需新的单次云资源授权。
