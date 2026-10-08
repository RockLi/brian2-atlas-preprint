# 异步连续突触与定时动作

本阶段把真实时钟访问接入训练前向与反向：连续突触、标准 `run_regularly` 和
`SubexpressionUpdater` 可使用快于、慢于或非整数倍于主时钟的 dt。神经元、输入
及 spike pathway 仍使用主时钟；多时钟神经元和异步事件缓冲属于后续阶段。

## 动作与访问

每个 action 可声明 `clock`，省略等同于 0。前端按产生动作的实际 CodeRunner
绑定时钟，保留 Brian 的 slot/order/name 和边顺序。共享赋值、向量赋值、间接
索引和 cached expression 的动作段都继承该 runner 时钟。

`itinerary.rs` 先有界重放访问以计数，再根据实际访问数检查上下文、地址、时钟和
随机地址表的总内存预算，然后分配。只有某个训练动作可能执行的访问才保存；
仅监视器时钟执行的访问仍推进核心游标，不能直接从时钟集合中删除。

每次保存各时钟的 pending times、calls、active 标志及可选主输入 frame。
CPU 和 GPU 只执行 active clock 对应的动作；未执行动作是恒等映射。前向记录
执行前上下文与间接地址，反向按 visit/action 的实际逆序回放。

## 时间、随机流与梯度

`Time/TimeWord` 使用动作时钟自己的时间。显式 `ClockTime` 仍读取指定对象的
pending time。突触积分的 dt 为突触时钟步长；`t_pre/t_post` 读取端点时钟。
定时更新器代码中的 `t/dt` 保持 Brian 的 owner 变量绑定，不因 runner 选择了
不同频率而强制换成 runner 时钟。

随机采样地址使用所属时钟的实际 calls，前向和 VJP 复用相同地址。同步计划的
样本身份保持不变，快时钟中间访问不再复用主步样本。命名 xi 的扩散因子使用
实际积分 dt。验证同时覆盖命名 xi、独立 rand/randn 及固定样本 Cython 对照。

输出 spikes 的形状仍为 batch × 主输入步 × neuron。只有主访问产生 loss seed，
logit/梯度按主输入步数归一化。TBPTT 在指定主 frame 的所有动作之前截断：其前
面的中间访问属于上一段；该主访问以及同次 coalescing 的其他动作属于下一段。
暖快照可能存在首次主访问之前的动作，它们正常影响初态梯度。

## 设备与 MPI

动态 GPU ABI 升为 v5r8，v4 ABI 不变。Metal 和 CUDA 使用同一份访问/动作解释器，
新增主 frame、active clocks、action clock 和每访问精确时钟数据。设备执行真正
的前向和 VJP，不使用 CPU 计算替代。

MPI 各 rank 共享相同访问顺序和 owner 划分。主机直接跳过不活动或被 mask 禁用的
动作，避免对空动作提交设备命令与 collective。TBPTT 截断通过独立 phase 8
提交，不依赖 action 0 是否活动。`gpu_dispatches` 根据实际启用动作与截断计数。

## 续接与限制

沿用完整 `clock_state` checkpoint。carry 在新 Brian run 边界重新对齐 ticks，
保留 calls；全段与分段使用相同的随机身份。异步手动请求若只提供 start_tick，
先重建一段规范的历史，再开始新 run；需要保持真实分段历史时应同时传入之前
返回的 clock_state。同步无 cursor 请求保留已有的连续前缀采样约定。

尚未放宽非主时钟的阈值、直接 spike trigger 或 event-noise pathway。它们需要
各自 spike buffer 生命周期、跨访问事件身份和按 pathway 时钟定义的延迟队列，
否则会漏掉快速神经元多次发放或重复消费慢时钟事件。该限制在原生验证器中明确
拒绝，不会静默执行错误的主步近似。

正式证据目录为 `mpi-evidence/training-async-actions-final-20261004/`。验收状态
以 `completion-verification.json` 和已观察的父进程终态为准，pilot 不累计。
本轮已验收 283 passed／136 NVIDIA-only skipped／0 failed（7 个完整模块、
419 个身份、317 个冻结文件），另有 16 项 Rust 测试通过；父进程 exit 0 已观察。
新增异步模块为 58 passed／27 NVIDIA-only skipped。CPU、实际 Metal、本机
CPU/Metal 2/8 MPI 均有覆盖，CUDA 翻译后源码通过主机 C++ 语法检查。
NVIDIA 实机、跨主机、多物理 GPU 不在本轮本地实测范围；整体目标尚未完成。
