# 暖时钟变更与动态突触组合

动态转换现在支持已运行 Brian 网络的合法 deferred dt 变更。Brian 给 clock.dt
赋值后不会立刻重写 t/timestep 数组，而是在下次 Network.run 对齐。因此转换器
读取下一次运行的 pending 时间，用 Brian 的 check_dt 检查变更，再计算新网格
上的时间；不会调用 Network.run、clock.prepare 或 queue.prepare。

例如网络停在 0.61 ms，原群体步长为 0.1/0.3 ms，物理 pending 时间为
0.7/0.9 ms。改为 0.2/0.6 ms 后，下一次运行时间为 0.8/1.2 ms。前端不能把旧
数组时间当作新网格上的时间，也不能为绕过校验而先运行一次零时长网络，后者
会准备或重采样事件队列并改变模型状态。

输入 origin、固定 refractory 剩余 tick 和表达式 refractory 的 elapsed tick
均使用新 pending 时间。旧步长标记、时间/timestep 数组与事件队列保持原样。
普通未改步长的输入/神经元仍要求与 network.t 对应，陈旧快照不会被自动接受。
附加 runner 的 dt 变更同样验证；暖网络新加入的监视器在下一次运行正常对齐。

路径 pending 快照仍通过安装的 CSpikeQueue 头文件读取，并复现 Brian 原生
prepare 对旧 dt 到新 dt 的队列重采样规则。路径自身 dt 用于新发放的延迟量化；
已有历史按其原到达时间重采样，不能简单把所有历史改成新 delay 长度。

新的联合验证覆盖多时钟神经元、连续 SDE 突触、summed、随机缓存、延迟路径，
并覆盖共享/linked 状态、运行时索引和定时动作。独立整数时间调度与 FIFO 参考
计算不调用原生动作解释器；对全部训练参数和可微初态检查 full/TBPTT 差分。
真实 Brian Cython 比较冷暖状态、事件时间及固定随机样本消费顺序。本机 MPI
同时检查 batch、carry、新训练器 checkpoint 与 CPU/Metal 一致性。

正式验收已通过：615 passed／198 NVIDIA-only skipped／0 failed，九个完整模块、
813 个身份、321 个冻结文件；父进程 exit 0 和完整证据已核对。
[本轮证据](mpi-evidence/training-multiclock-transitions-final-20261004/README.md)。
本轮没有修改原生源码，复用前一轮冻结执行器 v5r9；构建、Rust 17 项和 CUDA
主机语法证据明确继承，不作为本轮新运行计数。CUDA 实机仍未验证，跨机器暂缓。
剩余标准函数、随机分布及特殊对象等缺口见该证据目录 NEXT.md。
