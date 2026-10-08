# 多时钟神经元与路径延迟自动转换

动态 Brian 前端按每个群体自己的 dt 编译积分、随机扩散与 refractory tick。
输入时钟继续定义主帧、loss 归一化和 TBPTT 边界。静态前端仍要求共同步长。
暖快照检查每个群体的 pending 时间与下一次 Network.run 对齐，不再要求不同
时钟的 pending 时间相等。

当群体或阈值使用非主时钟时，前端建立 `spike_buffers`，从 `_spikespace` 采集
暖值，并将内部触发改为显式状态缓冲引用。阈值在其时钟上生成一次发放，reset
和路径按各自调度消费。真实发放记录见 `event_visits`；主帧 `spikes` 是计数。
已有同主时钟模型保留原有图形布局。

Brian Cython 的阈值模板在 `after_run` 清空 spikespace 的计数。因此此类前端
计划声明 `clear_spike_buffers=true`：运行内部保留缓冲，完成运行后仅将返回的
可续接状态中的缓冲清零，保留本次事件输出和梯度。通用原生缓冲 IR 默认仍可
跨请求保留缓冲；该选项不会改变任意手写计划的既有语义。

延迟布局的每个 Path 可声明 `clock`。初次 delay 量化、warm pending 读取、
运行边界 update_delays、动态 delay route 准备均使用路径 dt。所有移位、清空、
record、selector、pending 和事件动作继承路径 clock；重建与 checkpoint 保留
该身份。旧请求省略 clock 时使用主时钟。

标准路径的时钟必须匹配源群体/阈值时钟，避免把慢时钟旧缓冲当成新事件反复
入队。内部路径源通过脉冲缓冲索引与阈值时钟配对验证，外部输入路径仍使用主
输入时钟。队列验证同时审计动作时钟、触发尾槽、规范移位图和 source 身份；
不能通过修改路径 clock 而保留原动作图来绕过检查。

CPU/Metal/CUDA 共用原生队列重建，设备 ABI 保持 v5r9。主机请求新增路径时钟
和边界清理字段，必须使用本轮配套执行器；设备 ABI 不变不代表旧主机可读取。

正式本地验收：530 passed／199 NVIDIA-only skipped／0 failed，九个完整模块、
729 个身份、319 个冻结文件；另有 17 项 Rust 测试通过。新增前端模块为
130 passed／63 NVIDIA-only skipped，覆盖 CPU、实际 Metal、本机 2/8 MPI。
父进程 session 78072 exit 0 已观察，源码、执行器、完整身份与终态核对通过。
证据入口：[完整验收记录](mpi-evidence/training-multiclock-frontend-final-20261004/README.md)。

新版 NVIDIA 设备实测尚未完成，CUDA 证据仅为翻译后内核的主机 C++ 语法检查。
跨机器继续暂缓；时钟边界组合、自定义回调等剩余范围见该目录 NEXT.md。

后续已实现并验证[暖时钟变更及多时钟动态突触组合](NATIVE_TRAINING_MULTICLOCK_TRANSITIONS.md)，
包括只读 dt 重对齐、队列重采样、连续 SDE/summed/随机缓存与共享索引组合。
