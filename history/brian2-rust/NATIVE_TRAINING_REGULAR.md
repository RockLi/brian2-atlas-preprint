# 标准 run_regularly 训练转换

`lower_brian_training(..., dynamic=True)` 与 `lower_brian_dynamic_training(...)`
现可转换选中 NeuronGroup、其连续 Subgroup 及 Synapses 上的标准 `run_regularly`。
下文为 native v5r7 原阶段记录。后续 v5r8 已支持[异步定时动作](NATIVE_TRAINING_ASYNC_ACTIONS.md)，
包括不同频率的 rand/randn 调用。转换器生成原生动作，不用 Python 执行模型或反向传播。

## 执行与梯度语义

- 使用 Brian 的语句分析与单位检查，按 `Network.sorted_objects` 的 slot/order/name
  执行。原阶段限定相同 dt；v5r8 已按 runner 实际时钟执行。
- 标量数组在标量代码之前读取，每个逻辑别名使用独立局部值；标量代码每步执行一次。
  物理数组按 Cython 的名称排序写回，向量循环继续使用原标量局部值。不能把共享代码
  对每个神经元或每条边重复执行，也不能在别名写回后重新读取全局值代替局部值。
- 局部中间值和子表达式缓存保留 Brian 的重新计算位置；浮点、Boolean 和 int32
  物理状态按类型处理。不支持的 int64 临时量显式拒绝。
- 子群仅遍历视图范围，物理地址保持父群体地址；linked 运行时索引使用语句前读取、
  写回时的最终局部索引。条件写入按 Brian 的 refractory 规则嵌入变换。
- 定时修改的突触参数进入 carried runtime state，可训练初值只在初始化时注入。
  Adam 更新不能覆盖续跑中的突触权重。只在 runner 中读取的 shared/array 神经元
  训练参数仍读取优化器 bank，不能折叠成普通常量。
- 向量突触动作遵守每条边的结构 mask；标量共享更新仍按时钟执行，包括零连接对象。
- `randn()` 逐调用点使用独立原生 normal stream。标量与向量流不复用调用点，
  临时变量的多次引用复用同一次抽样；支持冷／暖快照和分段 carry。`t`、`dt`、
  TimedArray 仍走原生时间及输入 bank。
- 上述动作都进入现有 full/TBPTT 图。标量快照的伴随、别名读取梯度和排序写回
  使用普通原生状态 VJP；整数／Boolean 控制量停止梯度。

## 验证入口

新增完整模块 `tests/test_training_regular.py`、`tests/test_training_regular_edges.py`：
真实编译 Cython 前向、共享／子群／动态索引／不应期、多个调度槽、固定共同噪声、
暖快照、store/restore、空代码／纯标量／空突触、子表达式、typed state、TimedArray、
常量 bank、结构 mask、Adam carry；独立模型公式验证所有浮点权重和显式初态的
full/TBPTT 导数。本地 CPU、真实 Metal 和 2/8 MPI 使用同一动作图。

正式冻结验收目录：[完整记录](mpi-evidence/training-regular-final-20261004/README.md)。
结果为 232 passed / 56 NVIDIA-only skipped / 0 failed，288 个测试身份、278 个冻结文件；
父进程 session 74065 已 exit 0，最终验证脚本通过。两个新增模块占 57 passed / 24 skipped。
中间试验和失败诊断保留于 `mpi-evidence/training-regular-20261004/`，不计入正式结果。

## 尚未完成

多时钟神经元与异步事件缓冲、任意 Python 回调、int64/unsigned/
位运算、Synapses-to-Synapses 端点仍不能视为完成。NVIDIA 新实现仍待独立实机验收；
跨主机／多物理 GPU 继续按用户要求暂缓。本阶段没有调用云资源。

后续已接入同步定时代码的 `rand()`、混合正态／均匀调用点，并修复 masked
向量临时量的重建归属；见 [原生均匀随机数](NATIVE_TRAINING_UNIFORM.md)。
