# MPI 分段训练、STDP 与边界检查点

2026-10-02。本次扩展沿用目标神经元拥有入边的 MPI 执行计划，增加可变突触状态、
pre/post 学习路径、分段运行和磁盘恢复。CPU `reference-f64` 保持独立 Rust reference
的运算和事件顺序；真实 CPU/Metal 混合执行需要显式 `mixed-f32`。
本机验收及命令见 [training-20261002](mpi-evidence/training-20261002/README.md)。

## 运行入口

Python 只启动一次；Device 在每段内部启动、收集并结束 MPI ranks。
无需让 Python 脚本本身运行在 `mpiexec` 下。

```python
import brian2 as b
import brian2_rust

b.set_device('rust_standalone', engine='mpi', ranks=2,
             directory='/tmp/atlas-mpi-training', runner='/path/to/b2-runner')
# 创建固定候选连接的 NeuronGroup、Synapses 和 Monitor。
net = b.Network(group, synapses, spikes)
net.run(0*b.ms)                 # 初始化；允许零时长
net.run(20*b.ms)
net.store('training', filename='/tmp/training.checkpoint')
group.rates = next_sample_rates # 示例变量：须在方程中定义
net.run(20*b.ms)
net.restore('training', filename='/tmp/training.checkpoint',
            restore_random_state=True)
net.run(20*b.ms)                # 同一配置精确重放
```

完整可运行版本是 [examples/mpi_training.py](examples/mpi_training.py)。从仓库根目录：

```bash
CARGO_TARGET_DIR="$PWD/brian2-rust/target-mpi-training" \
  cargo build --offline --release --manifest-path brian2-rust/Cargo.toml --bin b2-runner
export B2_RUNNER="$PWD/brian2-rust/target-mpi-training/release/b2-runner"
export PYTHONPATH="$PWD/brian2-rust/python"
.venv/bin/python brian2-rust/examples/mpi_training.py \
  --directory /tmp/atlas-mpi-first --checkpoint /tmp/atlas-mpi.checkpoint
.venv/bin/python brian2-rust/examples/mpi_training.py \
  --directory /tmp/atlas-mpi-resume --checkpoint /tmp/atlas-mpi.checkpoint --resume
```

混合设备增加 `--rank-backends cpu,metal`。两个进程必须使用相同 rank 数、设备顺序、
数值模式及运行时版本。显式传入隔离 runner，不需安装或替换共享运行时。

## 支持范围

| 能力 | 当前契约 |
| --- | --- |
| Pair/Triplet STDP | canonical pre/post spike 路径；event-driven/clock-driven trace 与 `lastupdate` |
| 突触变量 | 显式/binary CSR 的 bool/f32/f64/i32/i64/u32/u64 状态与参数；binary 的逐边参数仍限 f64 |
| 事件顺序 | pre 保留原 source-major/重复边顺序；post 的目标邻接行按原创建 edge id 排序 |
| 写入 | owner 本地突触与本地目标神经元；post 路径不会写远端突触前神经元 |
| 延迟 | 显式连接支持统一和逐边整数 tick 延迟；binary CSR 仍限统一延迟 |
| 分段与 pending | 显式/binary 的队列、绝对 clock、RNG 身份及 monitor 历史可以继续运行 |
| 主机边界操作 | 输入/权重写入、归一化、固化、外部读出更新；下一段重新导出当前状态 |
| 自定义神经元 | 已有可降低的更新/阈值/重置方程；允许 NeuronGroup `after_groups` regular guard |
| 固定候选结构可塑性 | 本节规定的 live/born/active_after 掩码与代际保护；候选边数及 endpoint 不变 |
| fixed-total procedural | 保持原先一次 tick-0 activation；可变状态、pending 和分段训练仍拒绝 |
| GPU ranks | 神经元更新由真实 GPU 执行；STDP、队列及 MPI 通信仍在各 rank CPU |

多 clock、任意自定义槽、动态突触前状态读取、远端 pre 写入、linked variables、
一般自定义事件和实时 CSR 迁移仍不支持。只读 pre 状态保留现有静态证明。
网络规模预算及 [MPI.md](MPI.md)、[MPI_GPU.md](MPI_GPU.md) 的其他限制继续生效。

## 检查点与一致性

`Network.store` 保存一次成功完成的运行边界。所有 owner 的状态和 pending 事件收集到
协调端、ranks 正常结束后，才提交 Device 状态；没有暂停正在执行的 rank 并逐个落盘。
快照包括 Brian 神经元/突触数组、trace/lastupdate、monitor、refractory、绝对时间、
原始 RNG 状态、pending 布局与事件。统一 pre 延迟保存源端点批次；统一 post 按本地
目标邻接展开；逐边 pending 将原创建 edge id 映射为 owner 的紧凑入边 id。

磁盘快照沿用原子临时文件、fsync、替换与校验封装，拒绝尾随数据。加载前验证
`b2-mpi-boundary-checkpoint-v1` 的网络定义/拓扑 hash、rank 数、连续目标分区政策、
设备顺序、数值模式、包源码 hash、validator 二进制 hash、Rust 编译器、OS/CPU ABI、
Brian2/Numpy 版本。网络定义 hash 排除每段时长和记录窗口；变量值是快照状态。
配置不匹配在修改现有数组前报错。连续运行也检查相同契约，禁止中途修改 rank 或设备。

`restore_random_state=True` 是精确重放所需参数；省略它遵循 Brian 的新随机流语义。
新进程要用相同显式对象名称和候选连接顺序重建网络，再 restore。源码变化会使旧快照
失效；当前没有版本升级、跨 ABI、跨设备精度或重新分区迁移工具。
文件包含 Python pickle，限可信本地快照，不是可安全导入任意来源的交换格式。

任一 rank 错误触发 MPI_Abort，launcher 超时终止本地进程组；失败的运行没有可提交的
新边界。可从上一个已提交快照重试。当前没有在线 rank 故障接管或跨主机容错协议。

## 有界结构可塑性

采用已分配的候选连接，host 只在运行边界修改 `live`、`born`、`active_after`、权重和
trace。`active_after` 必须为该代出生时间加第一条 pre pathway 的该边延迟。
新生边清零 apre/apost/其他学习 trace；裁剪和生长策略仍由应用实现。

存在上述三个保留字段时，MPI lowering 在自动 event-driven trace 更新之前检查：
事件发射 tick 必须不早于 `active_after/pre_dt - pre_delay_ticks` 得到的出生 tick。
这同时保护 pre 与 post 旧延迟事件，避免旧代事件修改新生权重/trace/lastupdate。
具体传播、学习和 dead-edge 掩码仍须由方程显式乘 `live` 或作条件判断。
此契约需要 pre pathway；不支持更换 endpoint、扩展候选数组、重新分片或一般 CSR 生长。

低层的原始 pending API 不提供结构更新事务；使用 Device 的成功运行边界。
checkpoint 并非 O(local edges) 的分散文件：host 仍持有训练数组与历史，协调端有完整
快照开销。大型网络要限制 monitor、队列及保存频率，不能据此宣称内存与 ranks 线性缩放。

## 平台兼容性与验收

[verify_next_brain_mpi_training.py](tools/verify_next_brain_mpi_training.py) 只读调用平台的实际
build/present/consolidate/readout/freeze/structural/store/restore 方法。测试先用平台已有
reference 初始化，再在独立 Device 中选择 MPI；平台当前并未因此自动开放 engine='mpi'。
未修改平台文件、安装包或运行服务。

diehl-cook、reservoir、local-receptive、feedforward、convolutional、recurrent 六种
小规模 LIF 模型以 2 CPU ranks 通过训练、候选重连、冻结验证和恢复后第二样本精确重放。
卷积模型这里使用平台已有 STDP 连接，不能等同于下述原生 BPTT 的卷积共享权重支持。
CPU Pair/Triplet 使用 1/2/4 ranks 与独立 Rust reference 位模式比较；分段测试还覆盖
统一/非统一 pre 延迟、post 延迟、重复边、RNG、trace、monitor 和 fresh-process 恢复。
真实 CPU/Metal 两 ranks 完成同政策的连续/分段及 fresh-process 重放。
补充小规模探针通过 feedforward/custom/weight 结构策略、reservoir/AdEx/fixed、
recurrent/Izhikevich/random-rewire 的相同生命周期；这不是所有组合的笛卡尔积验收。
本轮没有 CUDA 训练硬件或跨主机训练验收，也没有训练吞吐/扩展性结论。

运行新增验收：

```bash
B2_TEST_MPI=1 B2_TEST_GPU=1 B2_TEST_MPI_GPU=metal \
  .venv/bin/python -m pytest -q brian2-rust/tests/test_mpi_training.py
```

## 与监督训练的关系

上述功能提供本地事件学习及可重复的训练生命周期。原生 surrogate BPTT 是独立的
[NATIVE_TRAINING.md](NATIVE_TRAINING.md) v1/v2 计划和执行器，当前单机 CPU/Metal。
MPI STDP 不会自动变成监督梯度训练。

未来 MPI backward-plan 应显式声明 parameter/state/tape ownership、每个前向节点的
VJP、跨 rank adjoint 通信节点、全局 batch/loss 分母、反向时间/槽次序及截断边界。
目标 owner 保存入边权重及 optimizer moments；源 adjoint 需要从目标 owner 返回源
owner，并按全局 edge/batch 次序确定性合并。精确 f64 不应直接用无序浮点 MPI_SUM。
每步须全 rank prepare 后 commit，快照同时绑定 forward/backward 计划、optimizer、
通信序列及设备精度。以上是后续接口约束，本次尚未实现 MPI VJP 或分布式 optimizer。

v2 共享参数增加一个独立约束：同一参数可能被不同目标 owner 的多条边引用，不能让
各 rank 对同一逻辑参数各自执行 optimizer。未来计划需为共享参数指定唯一 owner，
确定性归并来自所有边 owner 的梯度，并同步更新后的参数；这是后续 MPI backward-plan
工作，不是当前单机 v2 已具备的分布式能力。

## 原生监督式 MPI BPTT

后续显式训练计划现有 CPU MPI 目标分区 BPTT，含循环/共享参数与 v3 标量方程、
唯一参数 owner optimizer、固定 ranks checkpoint。它目前复制模型和 tape 数组；
不等同于本文 STDP 的分片内存实现，亦未支持 GPU MPI BPTT。见
[NATIVE_TRAINING_V3.md](NATIVE_TRAINING_V3.md) 的语义和验收范围。
