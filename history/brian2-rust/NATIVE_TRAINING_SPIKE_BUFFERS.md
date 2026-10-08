# 持久脉冲缓冲与逐访问发放

本阶段扩展原生 v5 动作 IR，为多时钟神经元前端提供事件执行基础。后续已接入
[多时钟群体与 pathway 队列自动转换](NATIVE_TRAINING_MULTICLOCK_NEURONS.md)。
本文保留原生缓冲阶段的契约与历史验收，不能将此阶段视作全部语义完成。

`dynamic.spike_buffers` 是与 `voltage` 等长的状态索引数组，省略时保留旧计划行为。
每个单元必须是独立、可微的 binary state，位于神经元物理状态区之后。原生阈值
动作是这些单元的唯一写入者，普通动作、间接写入、参数初始化和 detached 存储
均不得覆盖它们。初值保存暖快照中的最后一次阈值结果。

缓冲模式允许阈值绑定非主时钟。阈值实际执行时，同时生成本次访问的发放记录并
覆盖缓冲；未执行时缓冲保持原值。消费者使用 state trigger，按自己的时钟读取
最近结果。调度层须区分源时钟上的一次消费与用户明确安排的跨时钟重复读取，
不能把某个缓冲为 1 当成每个全局访问都产生了一次新发放。

反向传播把对持久缓冲的伴随与本次发放的 loss seed 合并，再通过对应阈值返回
电压或 surrogate margin。每次阈值覆盖都会清除旧缓冲伴随，即使 refractory
活动条件阻止了发放也必须清除。普通 state trigger 已支持跨访问回传；TBPTT
在主输入帧边界同时切断模型状态和持久缓冲，carry/checkpoint 则保存其前向值。

结果 `event_visits` 提供：

- `clock_times`：每次访问中各时钟的 pending 时间。
- `neuron_clocks`：每个神经元阈值所用时钟索引。
- `frame_indices`：访问归属的主输入帧；首次主访问之前的暖访问归入 frame 0。
- `spikes`：batch × visit × neuron 的逐访问发放，保留多次事件。

原有 `spikes` 为 batch × 主输入帧 × neuron。在缓冲模式中其值是发放次数，
可能大于 1；旧同步计划仍为 0/1。每次实际发放都参与 loss，按原主输入帧数
归一化，不按访问数或缓冲读取次数归一化。逐事件实际时间应从 `event_visits`
读取，不能从主帧计数推测。

CPU、Metal、CUDA 的前向和 VJP 使用相同规则。动态设备 ABI 为 v5r9，新增
缓冲布局元数据；MPI 阈值 owner 计算后同步更新副本，反向覆盖在 collective
应用阶段清除缓冲伴随。v4 路径不变，v5r8 主机/设备不能混用。

计划验证还检查 surrogate margin 与阈值动作的时钟一致，禁止使用另一时钟上
可能未更新的 margin。缓冲模式中的内部直接 spike trigger 被拒绝，必须显式
引用状态缓冲；外部输入直接 trigger 仍仅限主输入时钟。

自动转换、按路径时钟量化/迁移的延迟历史，以及暖 Brian spikespace 的采集已在
后续多时钟前端阶段接入；其 after_run 清理选项与本文默认持久缓冲契约分开。
新版 NVIDIA 设备实测仍未完成，跨机器继续暂缓。

本轮正式验收 294 passed／129 NVIDIA-only skipped／0 failed，6 个完整模块、
423 个身份、318 个冻结文件；另有 17 项 Rust 测试通过。新增模块为 39 passed／
17 NVIDIA-only skipped，实际覆盖 CPU、Metal、本机 CPU/Metal 2/8 MPI。父进程
exit 0 已观察，证据核对通过，入口为
`mpi-evidence/training-spike-buffers-final-20261004/README.md`。
