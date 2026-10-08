# 随机方程、动态突触和状态依赖不应期的联合验证

2026-10-04。此前分别实现的组件已接入同一模型作独立验证；本轮没有修改原生执行器
或前端实现。基线为 v5r7，原有245份冻结文件及原生二进制均保持一致。
最终验收84通过、42 NVIDIA跳过、0失败（126用例），见[本轮记录](mpi-evidence/training-stochastic-refractory-combined-final-20261004/README.md)。

## 同一模型中的能力

- 神经元 Heun：状态间共享噪声，以及两个噪声流下的交叉状态依赖；Milstein：
  Brian 支持的独立、对角乘性噪声。电压受不应期约束，另一状态继续演化。
- 不应期为 `u > limit` 或当前 `r + shift`；前者是布尔条件，后者会在运行中改变。
- 同一 Synapses 对象同时包含事件驱动的指数迹 `ap`、clock-driven 乘性 SDE 状态
  `z`、可塑性权重 `w`、可训练噪声强度 `rho`、summed 神经元输入和 pre/post 路径。
- pre/post 路径既改变可塑性状态，也改变自身延迟。队列在后续调用中按当前延迟
  建立新路由，同时保留已经在途的事件。
- optimizer、full/TBPTT、双样本、carry、store/restore、本机2/8进程 MPI，
  包括非 root 动作发生表达式域错误时的原子回滚。

Brian 本身不允许 summed 表达式直接依赖 event-driven `ap`。本模型采用合法的
`q_post = w + .2*w*w + .1*z`，事件迹通过可塑性更新影响 w。没有绕过 Brian 的模型
约束。Heun/Milstein 也继续遵守 Brian 各自的噪声结构限制。

## 独立证据

`test_training_stochastic_refractory_combined.py` 使用实际 Cython 生成代码，
只在 `randn` 缓冲区补充时注入共同的外部高斯样本。样本顺序从生成代码的标量调用
顺序及 Brian updater 调度取得，并核对抽样次数和神经元／突触的独立噪声域。
积分、summed、阈值、事件路径、队列和 reset 都由原始 Cython 代码执行。

测试从冷网络和先运行两步的网络转换，连续执行多个长度不同的块，每块恢复检查点
后与 Brian 对照物理状态、脉冲、活动标志及可变延迟。这验证了暖启动快照和在途
事件在接续后的效果。共同噪声用于比较积分／调度语义；不声称两种后端默认随机
数发生器会逐位生成同样的序列。

`test_training_stochastic_refractory_vjp.py` 独立编写向量 SDE 公式、事件时间桶、
summed、条件写入和顺序可塑性变换，未解释原生 SSA 或 action 程序。有限差分遍历
所有浮点参数和初态，包括待到达队列输入；用已声明的 surrogate 及固定离散轨迹
约定检查 full/TBPTT，并要求神经元 sigma、突触 rho 的梯度确实非零。

可变延迟会扩展内部队列布局。物理状态按公开 provenance 比较，初态伴随须保留
完整原始输入长度，队列延续通过跨块 Brian 对照检查。检查点恢复使用 `trainer.plan`
（当前运行时计划），而不是已经过期的初始 `bundle.plan`。

## 边界

实际设备验证为 Metal；CUDA 例项需 NVIDIA 资源，目前跳过。本地 MPI 共用本机
资源，不代表多物理 GPU 或跨主机验收。本轮覆盖上述联合模型，不能推广为任意
噪声耦合、任意异步连续积分或任意 Brian 回调均已支持。整体目标仍保持 active；
剩余范围见本轮 `NEXT.md`，跨机器继续暂缓。
