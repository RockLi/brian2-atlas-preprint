# CPU MPI：确定性执行与分区存储

分支 `codex/mpi-cpu`，基于 ExecutionPlan 提交 `5c34cb2d`，2026-09-07。
B2IR v1 和独立 Rust reference 不变。MPI 是可选运行依赖，普通 CPU/GPU
用户不需要安装 MPI，也没有新增 Rust crate 依赖。

按 rank 混用 CPU/Metal/CUDA 的可选神经元更新卸载路径见
[MPI_GPU.md](MPI_GPU.md)；它要求显式 `mixed-f32` 数值模式。本文以下仍描述
默认的 CPU `reference-f64` 契约。

2026-10-02 新增显式/binary 连接的 pre/post STDP、分段续跑和边界磁盘恢复，详见
[MPI_TRAINING.md](MPI_TRAINING.md)。下表已更新；既有性能记录仍属于其原验收版本。

## 使用

安装 MPICH 或 Open MPI，使同一套实现的 `mpicc` 和 `mpiexec` 在 PATH 中；
还需要现有 Python/Brian2 环境、Rust/Cargo、C 编译器。

```python
import brian2 as b
import brian2_rust

b.set_device("rust_standalone", engine="mpi", ranks=4,
             directory="mpi-example")
clock = b.Clock(dt=0.1*b.ms)
g = b.NeuronGroup(100, "dv/dt=200*Hz:1", threshold="v>1",
                  reset="v=0", clock=clock, method="euler")
s = b.Synapses(g, g, "w:1 (constant)", on_pre="v_post += w", clock=clock)
s.connect(i=list(range(100)), j=[(i+1) % 100 for i in range(100)])
s.w = 0.1
s.delay = 0.2*b.ms
spikes = b.SpikeMonitor(g)
b.Network(g, s, spikes).run(10*b.ms)
print(b.get_device().explain_plan())
```

**Python 入口只启动一次**。Device 会编译 Rust AOT + MPI C shim 并启动
`mpiexec -n 4`；不要再用 mpiexec 启动这个 Python 脚本。此版本每 rank 一个
计算线程，`threads>1` 仍只属于原 `engine="aot"`。

直接操作已导出的模型：

```python
plan = brian2_rust.build_execution_plan(model, backend="mpi", ranks=4)
print(brian2_rust.explain_plan(plan))
brian2_rust.write_mpi_project(model, "mpi-project", ranks=4, plan=plan)
brian2_rust.compile_mpi_project("mpi-project")
observed = brian2_rust.run_mpi_project("mpi-project", "mpi-result")
```

大量同构 fixed-total 投影可在 `write_mpi_project` 中显式设置
`compact_projections=True`，用配方表和保序循环减少生成代码重复；manifest
记录实际压缩或布局回退。该选项不改变分片、调度或浮点顺序，默认关闭。
[十亿级数值验证与完整模型编译限制](mpi-evidence/full32-compiler-experiments/README.md)。


项目包含 `main.rs`、`mpi_bridge.c`、小型索引 `instance.bin`、
`instance.rank-N.bin` 分片、`instance-identities.rs`、`execution-plan.json`、
`manifest.json`；编译后增加 `b2-mpi` 和 `build.json`。生成源码仍复用现有
canonical slot lowering，没有复制一份神经元模型解释器。

## 计划、执行与正确性

`DistributedPlan` 保存经过独立 Rust 验证的 LogicalPlan、三个 B2IR layer
hash、rank 数、每 population 的连续区间、每 rank 的目标入边数量、路由统计、
最小跨 rank 延迟及通信节点。发射时重新派生完整计划并比较，不能仅改一个 hash
或导入任意 JSON 来跳过验证。

每个神经元属于一个 rank；每条边属于其目标神经元所在 rank。计算和写入只在
owner 上进行。每个 threshold/event-source 节点执行后使用 MPI_Allgatherv
交换 population 内的全局 neuron index，排序并检查唯一性；population 身份由
通信节点隐含绑定。所有 rank 以同样的 canonical 节点顺序进入 collectives。
第一版向所有 rank 发送 spike，计划里的路由统计尚未用于选择性通信。

目标 rank 保留原 source-major 和边创建顺序。统一延迟 pathway 按源编号排队，
到达时遍历本地入边；逐边延迟仍按投递顺序排队。跨 tick 维持入队顺序，不能用
MPI 消息到达顺序替代事件顺序。标准 pre pathway 的
延迟允许零或正整数 tick；threshold 交换先于同 tick 的 on_pre。当前逐 tick/threshold 通信，最小跨 rank 延迟只用于
解释；尚未做 lookahead 批处理。

种子、stream、绝对 tick、原 population 内 neuron index 和原 edge index
保持不变；不增加 `seed + rank`，也不重新编号本地边。f64 算术沿用 CPU
canonical lowering；不使用浮点 MPI_SUM 合并状态。模拟结束后，每个数组元素
只有 owner 贡献位模式，以整数通信收集状态、监控和 refractory，保留负零等
原始位模式。目标分片的神经元状态按连续区间收集；可变逐突触状态携带原始全局
edge id，以 65,536 项有界分块只在 rank 0 恢复原连接顺序，不在每个 rank 复制全局
edge/value 数组。只有 rank 0 写出标准 results/events dump 并回填 Device。

所有 rank 在计算前核对编译计划身份及 communicator 大小。可执行文件只绑定
小型实例索引和每个分片的长度/SHA-256；各 rank 流式读取自己的分片，对真正进入
状态数组的字节计算摘要，加载结束时验证，再开始仿真。不嵌入大型实例，也没有
校验后重新打开输入的竞态。修改模型、种子或 rank 数仍需要重建。
任一 rank 的输入/执行错误会调用 MPI_Abort，避免其他 rank 等待后续 collective。
Python launcher 有超时并终止本地进程组；这不提供集群故障恢复。

## 范围

| 支持 | 暂不支持，会在规划阶段拒绝 |
| --- | --- |
| 一个共享 clock，标准 update/threshold/reset 和 pre/post pathway 槽；NeuronGroup after_groups regular guard | 多 clock、其他自定义执行槽 |
| bool/f32/f64/i32/i64/u32/u64 神经元状态和参数、固定 refractory | 表达式 refractory |
| 静态显式、binary CSR 或 fixed-total recipe；重复边、子组、循环连接 | fixed-indegree procedural |
| 非负整数 tick 延迟、多 pre/post pathway；fixed-total 支持 clipped-normal/uniform 逐边延迟 | binary CSR 逐边延迟 |
| 显式/binary 的可变 bool/f32/f64/i32/i64/u32/u64 突触状态、clock/event-driven update、post `summed`；pre/post 写本地突触或目标状态 | 可变 fixed-total 状态、pre `summed`、远端 pre 写入；binary 逐边非 f64 参数 |
| on_pre 读取常量/目标状态、全程只读的突触前状态 | 运行中可变突触前状态读取、linked variables |
| SpikeGenerator、普通 spike 和 StateMonitor；原 counter RNG | 自定义事件、EventMonitor、用户 Function |
| 显式/binary 分段续跑、pending 恢复和同配置边界 checkpoint；fixed-total 一次 tick-0 activation | fixed-total 续跑、运行中 rank checkpoint、重新分区恢复 |

突触前状态只有在所有神经元和突触 CodeObject（含 refractory 隐式写入）都不写入
该状态时才可读取；计划保存 `readonly_pre_states` 证明。FlyWire 的静态
`transmission_pre` 掩码因此可用，动态突触前状态仍需要未来的状态通信。
CSR 路由用有界切片统计；实例流式加载时逐字节核对编译身份。

B2IR 自身的验证和预算限制仍适用。表中的支持面由规划器限制，测试记录说明
具体实测模型；不代表所有数学表达式和所有网络规模都已逐一测试。

Skaar 等 2025 的 explicit NMDA 固定输入诊断覆盖两个 RK4 逐边状态、延迟
`on_pre`、每 tick post `summed` 和 i32 monitor。640 神经元、737,280 总突触在
1/2/4/8 ranks 的预热及每组五次运行均与串行 Rust 结果逐字节一致。该组运行发生
在本机 MacBook Air，仅用于正确性和 codegen 诊断；其时间、rank 间比值和通信
占比不构成任何性能或扩展性结论。修复前的 2,560 神经元固定输入 1/2/4-rank
探针仍保留作缺陷证据，但不代表当前性能。正式 MPI 性能必须分别在 Mac Studio
27 和 Linux 23 上以原生 Rust 1.98.1、原生 MPI 和明确 CPU 绑定重新测量。

同一通用路径也支持 Brian2 原生 `PoissonInput`：它作为 `synapses` slot 的
population CodeObject 在 owner 的全局神经元索引范围上执行，counter RNG 的
population/index/stream 身份不随分片改变。一个三输入源最小模型已通过
1/2/4-rank 逐字节回归；Skaar 2025 未替换输入的 640 神经元模型也在 1/2/4 ranks
与同一冻结 B2IR 的串行结果逐字节一致。审计发现 canonical 突触循环即使不读取
突触前端点，也会为每条边、每个 tick 用 CSR `partition_point` 恢复 source。
通用 codegen 现仅在 CodeObject effects 或 reduction 地址确实需要时重建端点。
修复后 1/2/4-rank 本机诊断均保持结果逐字节一致，并确认不必要的 endpoint
恢复已从生成代码中移除。本机 MacBook Air 的单次时间和由此导出的比值不进入
性能结论。
论文首个 2,560 神经元规模也完成一次保留原始 `PoissonInput` 的 4-rank 控制；
完整结果与优化串行 Rust 逐字节一致。该结果是本机单次正确性控制；其时间及与
历史串行运行的比值全部排除，不作为性能或 rank 扩展曲线。

正式单机 MPI 性能已在 Mac Studio 27 和 Linux 23 分别完成。两台均使用原生
Rust 1.98.1 和原生 MPICH 5.0.1；27 全程为 arm64，不含 Rosetta。正式模型保留
作者原始 `PoissonInput`、float64 RK4、0.1 ms 步长、一秒生物时间、全部连接和
原始延迟。每个配置先预热一次，再测量五次；640 和 2,560 神经元共 132 次输出
全部与各自冻结 B2IR 的串行参考逐字节一致。

27 在 2,560 神经元下的 1/2/4/8/16-rank 模拟中位数为
241.461/125.297/66.430/38.552/29.240 秒，对 rank 1 的最大加速为 8.258×。
23 在同一规模下的 1/2/4/8/16/32-rank 中位数为
382.433/173.816/85.963/50.064/23.017/13.265 秒，32 ranks 为 28.829×。
640 神经元在 27 上于 8 ranks 达到最低中位数，16 ranks 因通信占比升至 75.2%
而回退；23 到 32 ranks 仍改善，但通信占比已达 61.2%。2,560 神经元的 23/32
预编译模拟仍扩展良好，不过首次计划和分片耗时 425.344 秒，远高于模拟本身，
因此一次性端到端运行并不受益。27 的 macOS 拒绝硬亲和性；23 每个 rank 固定
到 CPU 48–79 中的独立物理核。两台主机之间不计算速度比。

## 分布式 fixed-total 建图

`connect_fixed_total` 可直接用于 MPI：Python 只输出生成规则与神经元分片，
不在前端展开连接。每个 rank 处理全局抽样编号的独立连续区间，整数
Allreduce/Exscan 计算 source-major 全局边身份；有界 Alltoallv 把连接交给目标 owner。
本地按全局边身份排序，权重/延迟和运行时 RNG 使用与 reference 相同的原边编号。
支持 clipped-normal / uniform 逐边常量和延迟，以及共享常量/统一延迟。

规划器不会为统计图而重新展开全部连接：procedural 入边数显示为 `None`，
跨 rank 最短延迟不宣称已知。运行报告的 `procedural_topology` 保存每投影各 rank
实际入边数、所生成抽样数及建图/初始化时间；`rank_stats` 每两项为
`[owned_edges, generated_draws]`。静态图仍保留原来的准确路由统计。

默认每 rank 累计最多构建 10,000,000 条 procedural 入边；
`B2_MPI_MAX_LOCAL_EDGES` 可显式调整。预算在全局计数后、本地边数组分配前检查，
超出会中止整个 communicator。这个边数限制不是完整内存预算：参数数目、队列、
监控和已有静态图同样占内存，大规模运行仍必须使用已验收的进程树资源隔离。

目前仍需每投影 O(source_count) 全源索引，本地排序为 O(local_edges log local_edges)，
构建期有 16 bytes/edge 的临时记录；尚未支持区域映射或 fixed-indegree。
[实现与受限双节点验收](MPI_PROCEDURAL.md) 记录精确范围。

## 内存与性能限制

每个 rank 只加载本地可变神经元状态、参数、refractory、入边 CSR、突触参数与
自己的外部事件源。存储下标局部化，但方程里的 neuron/edge ID 和 RNG 身份保持全局值。
分片用有界 CSR 切片构建；不会为每条 binary CSR 边创建 Python 对象或全量选择数组。

全程只读的突触前状态目前仍复制到各 rank，作为静态读取数据；每个投影还保留
全局 source-offset 索引。StateMonitor 仍采用完整记录布局（非 owner 占位），结束时归并；
spike/EventStream 历史只在 rank 0 保存。rank 0 的最终输出缓冲区随记录量增长。
因此仍需控制监控规模。边界 checkpoint 收集到协调端，并非各 rank 独立快照；
尚未提供运行中的分散 checkpoint 或输出流式归并。

每 rank 一个线程；Linux 自动绑核按节点内 rank 选择，尊重进程已有的 CPU 允许集合，
避免所有本地 rank 都绑定第一个核心。`rank_cpu_ids` 记录实际选择，-1 表示未绑核。
输入校验不再把全量实例嵌入 Rust 编译单元，构建内存不会因这种嵌入随边数据膨胀。
仍使用逐 population、逐 tick 的全体 spike 交换；没有订阅路由、lookahead、MPI+CUDA、
MPI+多线程或动态重分区，不承诺增加 rank 一定加速。

结果里的 `mpi-runtime.json`（同样嵌入 summary 的 `mpi`）记录实际 rank 数、
处理器名称、每 rank 拥有的神经元数、发放数、投递边事件数、通信调用数和时间。
`simulation_and_recording_seconds` 目前包含最终结果收集，`spike_exchange_seconds`
仅覆盖 spike 交换及整理。比较时要分别报告编译、MPI 启动、初始化、计算/通信、
最终收集和写出；现有时间字段不足以作严格的强/弱扩展结论。

## 验证与复现

在工作树根目录运行（`python` 指向安装了此 Brian2 源码扩展的环境）：

```sh
cargo build --release --locked --manifest-path brian2-rust/Cargo.toml
PYTHONPATH=brian2-rust/python:. B2_TEST_MPI=1 python -m pytest brian2-rust/tests/test_mpi.py -q
PYTHONPATH=brian2-rust/python:. python brian2-rust/examples/mpi_cuba.py \
  --output /tmp/mpi-cuba-new --ranks 1 2 4
```

`B2_TEST_MPI=1` 必须允许 MPI 创建本地进程和监听端口；未设置时真实 MPI 用例
会 skip，纯规划验证仍执行。示例生成单份模型与独立 Rust reference，分别编译
1/2/4 ranks，并比较完整 results.bin/events.bin 的 SHA-256；结果和原始文件
保存在输出目录，失败不会被容差比较隐藏。

本机验收记录见 [mpi-evidence/README.md](mpi-evidence/README.md)。
后续已通过 23/24 两台 Linux 的 1+1、2+2 ranks 验证：CUBA 和随机突触/不同
延迟模型的完整结果、事件文件均与同源码 Linux reference 逐字节一致。
环境指纹、原始结果和失败排查记录见
[双节点验收](mpi-evidence/linux-two-node/README.md)。

Teleport 环境可使用 `tools/mpi_teleport_launch.py`：由本机分别登录两台并
启动 Hydra manual proxy，MPI 通信走私网；无需向节点复制登录凭据。
当前验证组合为配套 MPICH 4.2.3、PMI-1、ch3:sock。
[复现步骤](mpi-evidence/linux-two-node/REPRODUCE.md) 包含准确参数和目录约定。

分片优化前的完整 FlyWire EI 网络已通过四种条件 × 1/2/4 ranks 的 1 秒验收：
139,255 个生物神经元、15,091,983 条加权边，完整状态及事件与独立 reference
逐字节一致。MPI 进程峰值 RSS 395.95–400.50 MiB，构建峰值约 7.49 GiB。
数据、内存记录和复现脚本见 [全模型验收](mpi-evidence/flywire-full/README.md)
及 [复现步骤](mpi-evidence/flywire-full/REPRODUCE.md)。

最新分片优化的 33 次前后对照、逐字节验证及内存记录见
[rank-local 验收](mpi-evidence/rank-local/README.md)。旧的 396–401 MiB / 7.49 GiB
是历史实现数据；本轮四 rank 每进程峰值 76.5 MiB，构建峰值 663.7 MiB。

## 下一阶段

1. 在已完成的 rank-local 可变状态/入边/输入基础上，在 fixed-total 分片生成基础上补齐区域映射、其他 procedural 规则和静态前状态 ghost 压缩。
2. 按订阅路由投递 spike，推导安全的 lookahead 窗口并减少通信次数。
3. 建立本地 pending 队列、随机状态及监控恢复契约，再开放分段运行和 checkpoint。
4. 在已通过的 FlyWire 全模型上做通信与内存优化，再扩展到 Brunel/PD14 子集，报告强/弱扩展；
   然后推进可塑性和 MPI+CUDA。

底层 API 允许传入 `launcher_args=["--hostfile", "hosts.txt"]` 等当前 MPI 实现的
选项。集群上需要相容的 MPI 运行库、架构，以及每台机器相同绝对路径下的可执行
文件和实例。本次双节点使用上述配套 Hydra 工具；未验证所有 MPI 实现或启动参数。

多脑区皮层模型的规模、延迟约束和适配缺口见 [MULTI_AREA_MPI.md](MULTI_AREA_MPI.md)。

## 后续进展：官方参数与脑区归属

显式脑区/population 归属、Poisson 输入、空投影索引释放及共享内核已接入，
官方参数和 V1/V2 缩放适配的实际编译资源问题见
[MPI_MULTI_AREA_ADAPTER.md](MPI_MULTI_AREA_ADAPTER.md)。上述历史验收限制不代表最新功能上限。

### 大量 population 的编译优化（可选）

`write_mpi_project(..., compact_populations=True)` 同时启用 projection 压缩，
并将每个 population 的局部变量归入一个保留原始字段类型的聚合值。
初始化顺序、浮点计算和收集顺序保持不变，默认生成路径不受影响。
不支持的 projection 布局沿用普通生成器；预期源码片段变化时拒绝生成。
MAM 示例对应 `--compact-populations`。完整 32 脑区的编译和运行证据见
[population storage 验收](mpi-evidence/full32-population-storage/README.md)。


### MPI 分阶段计时

`mpi-runtime.json` 的 `rank_stage_seconds` 按 rank 排列，每个 rank 的五列名称
见 `rank_stage_columns`：本地初始化、初始化等待、本地仿真、仿真结束等待、
结果收集与报告。初始化后和仿真后各有一次同步，防止建图等待被算进脉冲交换。
旧 summary 的 `simulation_and_recording_seconds` 仍包含收集与报告；细分比较请
使用上述列。收集列不包含最后的计时收集、根 rank 的 JSON 序列化和磁盘写出。
[完整模型计时记录](mpi-evidence/full32-stage-timing/README.md) 保留对照和数值验证。


### 投影报告批量通信（可选压缩路径）

projection 压缩路径将投影统计按最多 128 个投影分批收集，保持每个投影的
整数溢出检查、rank 累加顺序及浮点计时的原始位模式。完整模型的 8,344 个投影
从 33,376 次报告通信降到 66 次；32 ranks 时每批接收载荷最多 160 KiB/rank。
`mpi-runtime.json` 中 `projection_report_collectives` 和
`projection_report_packet_values` 记录实际批次数和最大发送 u64 数。
完整 100 ms 对照的结果和事件文件逐字节一致，收集与报告从 10.01 秒降至
0.74 秒；这是一次顺序对照的测量。详见
[批量报告验收](mpi-evidence/full32-report-batching/README.md)。

完整 32 脑区的 500 ms 运行也已通过：约 413 万神经元、241 亿条递归突触，
四台机器共 32 ranks，总耗时 289.3 秒，各台峰值 132–135 GiB；
前 200 ms 的脉冲和记录轨迹与先前运行一致，全部资源守卫无 OOM。
这仍是工程运行和一致性验证，未作完整模型独立模拟器或上游 Brian2 性能对照。


### 事件队列槽位优化与错误诊断

混合延迟的共享加法内核复用当前 tick 的环形槽位，并用掩码完成单次回绕；
超出快速路径条件的延迟和整数回绕仍走原取模路径。FIFO 顺序、浮点加法和
队列元素类型不变。完整 500 ms 对照的结果文件逐字节一致，仿真阶段在一次
顺序对照中从 183.08 秒降至 176.70 秒，内存峰值基本不变。

MPI 错误退出路径增加一次固定 100 ms 的代理输出缓冲时间，以缓解 Hydra
在转发诊断前终止代理造成的错误文本丢失；不调用集体同步，正常仿真不走此路径。
100 项 Linux 回归及重复损坏输入检查通过。完整证据与测量限制见
[事件投递优化验收](mpi-evidence/event-delivery-kernel/README.md)。
