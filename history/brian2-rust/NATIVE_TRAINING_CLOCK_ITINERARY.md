# 原生多时钟访问核心与同刻顺序

`src/training/clock.rs` 新增共享时钟游标。它逐次描述 Brian 实际执行的时钟访问，
不仅提供主时钟时刻看到的其他时钟值。现有动态训练的时间采样和结构重建边界已
改为使用这个核心；不同 timestep 的连续积分动作尚未全部接入访问调度，不能将
本阶段称为完整异步训练。

## 访问和恢复

每次访问提供最早网络时间、全部时钟的 pending tick／时间，以及本次 active
时钟集合。仅 active 时钟前进。相同时间的时钟按 `order` 选择最早时钟，随后使用
Brian 的严格 epsilon 比较合并；初始时间和区间终点使用包含等号的舍入检查。
两处边界比较不能互换。

游标可流式提供访问，不必预先分配包含全部动作的训练 tape。每个游标快照保存
本段起始时间、初始执行次数、当前执行次数、pending ticks 和本段访问次数。
`restore` 使用有工作量上限的重放验证快照；`restart` 按新的区间起点对齐 ticks，
保留执行次数。精确续接一个既有区间与开始新的 Brian `Network.run` 区间是两种
操作。执行次数为后续异步随机抽样提供独立于绝对 clock tick 的身份；当前训练
动作尚未使用这组新计数。

限制为最多 256 个时钟，计入每次访问的全部时钟后最多 1000 万工作单位。时间、dt
和 tick 必须有限且可精确前进，tick 小于 2**53；执行次数不能溢出。一次时钟推进
先检查所有 active 时钟，再提交更新。训练动作 tape 的内存预算仍需由调用者检查。

## 修复同刻与区间边界

两个不同 dt 的时钟恰好同刻时，选择哪个最早时钟会改变第三个近邻时钟是否被合并。
这是因为 Brian 的合并容差取两个时钟较小的 dt，合并关系不具传递性。前端现在
按实际 `Network.run` 使用的 clock set 迭代顺序，保存每种 dt 首次出现的次序到
`dynamic.clocks.order`。该顺序随计划跨进程保存。旧计划没有 order 时沿用原数组
次序；新前端须配套本轮原生执行器，旧执行器不能读取新增字段。

Brian 首先检查最早时钟的结束条件，但 `Clock.advance` 仍会拒绝任一 active 时钟
超过自己的区间结束步。新核心保留这一拒绝规则，不静默多执行时钟。现有时间采样
现在遍历到区间结束，并检查主时钟访问数与请求步数一致。结构重建边界单独使用
舍入后的主时钟终点，允许其在容差内略早于暖快照的 Network.t。

本阶段只修改 Rust 主机调度和前端元数据，GPU kernel ABI 仍为 v4r7／v5r7。

## 验证

`test_training_clock_itinerary.py` 把原生访问流逐条对照实际 Brian Network 的回调
记录，覆盖快／慢／非整数比／相同 dt、第三时钟、同刻优先级、冷暖起点、跨进程
恢复和分段 Network.run。区间越界同时检查 Brian 的 StopIteration 与原生事务性
失败。CPU 和实际 Metal 还验证了同刻顺序对已有训练时间采样的影响。

本轮正式回归入口：`mpi-evidence/training-clock-itinerary-final-20261004/`。
正式结果为 230 passed／43 NVIDIA-only skipped／0 failed；另有 10 项 Rust 测试和 8 项独立 MPI 检查通过。
源码、退出码和逐项记录已由 verify_completion.py 核对，早期 pilot 不累计。

## 后续连接与剩余范围

[异步动作阶段](NATIVE_TRAINING_ASYNC_ACTIONS.md)已为连续突触、定时更新和缓存动作
绑定实际 clock，接入 CPU/GPU/MPI 的访问级前向和 VJP、随机 calls 以及主 frame
边界。神经元阈值和 spike pathway 仍限定在主时钟。

- 多时钟神经元的多次发放与跨访问 spike buffer 生命周期。
- 按 pathway 时钟执行的事件投递、延迟队列、触发身份及其伴随。
- 神经元自己的积分、refractory、读出归一化和多次事件的完整衔接。

Python 回调自动求导、扩展整数类型／位运算、Synapses-to-Synapses 端点和新版
NVIDIA 实测也仍有缺口；跨主机／多物理 GPU 按用户要求暂缓。本阶段没有云调用。

## 训练检查点连接（2026-10-04，后续阶段）

新增 `src/training/clock_state.rs`，为训练请求和结果提供 `clock_state`：包含
`next_tick/start/initial_calls/calls/ticks/visits`。本机 CPU、Metal 和 MPI 使用同一
主机时钟表；设备仍执行原有动作序列，GPU kernel ABI 无改动。

`initial='carry'` 同时传递 committed neuron state、主时钟 tick、随机 sequence
和完整 cursor。在新 Brian run 的主时钟边界重新对齐 pending ticks，保留执行
计数。默认无 cursor 的手动请求仍按旧连续前缀重放；精确手动分段请传入上次
结果的 `clock_state`、`final_tick` 和 `noise_sequence`。只成功的 train 更新
trainer 游标，evaluate/gradients 只返回候选结果。返回值的字典修改不影响
trainer 已提交的 cursor。

store 把 cursor 纳入 checksum；restore 在任何 Python 状态提交前，通过原生
调度器有界重放，检查 ticks、执行次数、本段起点、访问次数及主时钟边界的一致性。
校验是状态内部一致性验证，不是随机流历史的来源认证。已提交且有动态时钟计划
的 checkpoint 不允许丢失 cursor。没有动态时钟的旧式计划仍可正常恢复。

update_delays、update_timed_input 和 update_mask 原样携带 cursor，不消耗
时钟或 RNG。结构重生的时间戳使用已提交 cursor 的实际 pending clock 值。
连续训练的 cursor 临时存储已纳入内存预算。

新增 `test_training_clock_checkpoint.py` 对照真实 Brian 分段运行中的全部
时钟 ticks/calls 和最终状态，覆盖冷暖起点、非整数比、多进程 checkpoint、
随机流 carry、只读调用、被篡改但重新签名的 checkpoint、区间失败事务性、
结构/延迟/输入更新及实际 Metal、本机 2/8 MPI。正式证据见
`mpi-evidence/training-clock-checkpoint-final-20261004/`；验收以该目录
`completion-verification.json` 为准。

这一步完成训练 cursor 的保存和恢复，不放宽连续突触/runner 的异步执行限制。
完整访问动作、反向 tape、事件/延迟队列及动作随机计数仍在后续工作范围。
