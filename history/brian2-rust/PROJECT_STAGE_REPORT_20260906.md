# Brian2 Rust Standalone 阶段项目报告

**报告日期：** 2026-09-06  
**当前分支：** `gate0/minimal-rust-backend`  
**报告生成时 HEAD：** `c396b5d8`  
**稳定 IR：** `b2ir-v1` / protocol `1.0`  
**项目阶段：** Gate 0 高性能单机后端，B2IR v1 已冻结，兼容面和执行优化持续扩展

## 1. 执行摘要

项目已经从“单组、同构方程、文本结果回填”的早期探针，发展为可由普通 Brian2 `Network` 驱动的独立 Rust standalone 后端。当前核心成果是：

- 每个 population 拥有独立状态/参数 schema、方程、clock、threshold/reset、refractory、monitor 与 `CodeObjectSpec`；IR 不再有两个 population 或八个 Synapses 的结构性上限。
- `b2ir-v1` 已冻结，包含全局 scheduler、Effect Algebra、类型与单位系统、命名事件、linked view、Function Contract、Definition/Instance/Run hash 域和稳定 canonical encoding。
- reference executor 与模型专用 AOT executor 共用同一份已验证 IR；不支持的语义在构建前 fail closed，不静默切回 NumPy/C++。
- 原生结果采用 versioned little-endian binary dump，Python 通过 NumPy memmap/view 读取；大型数组不再经 CSV 或 JSON 回填。
- 单进程内并行使用常驻 `std::thread` worker、target-owned event routing、degree-balanced partition、共享 dispatch 和安全的 plasticity/summed reduction；未依赖 Rayon。
- PD14 全规模 77,169 神经元、298,880,968 突触已在 16GB M3、M1 Ultra 和双路 EPYC 上由 Rust 完成；FlyWire 139,255 神经元、15,091,983 加权边也已完成三机运行和完整结果校验。

当前性能结论不是“所有场景均已最终胜出”。M1 Ultra 的完整核心门 24/24 通过；EPYC 为 23/24，唯一缺口是串行 PoissonInput 慢 9.1%；交互负载较高的本地 M3 为 23/24，异常 STDP 结果与同机安静复测不一致，应视为受污染样本。更重要的是，双路 EPYC 上 NEST 的 96 核 PD14 最优结果仍比当前 Rust 快 6.44 倍，说明 NUMA/多核执行架构仍是下一阶段的主要性能目标。

## 2. 项目目标与原则

项目保留 Brian2 Python 建模体验，同时把模型语义与执行实现隔离：

1. 语义兼容优先：schedule、单位、事件时序、索引域和公开结果必须可验证。
2. 单机性能优先：先打通 CPU 本地性、事件路由、内存布局和并行调度，再扩展 GPU/MPI。
3. 内存有界：拓扑、monitor、checkpoint 和结果协议必须能支撑长时程与亿级连接。
4. 优化必须通用：根据 IR 类型、effects、拓扑和工作量自动选择计划，不要求用户为 CUBA、COBAHH、PD14 等模型手工写特例。
5. 明确拒绝优于静默降级：capability report 在运行前解释不支持项。

## 3. 当前架构

```text
Brian2 Python API
        |
        v
Frontend Adapter / CodeObjectSpec lowering
        |
        v
Canonical B2IR v1
  - Definition / Instance / Run
  - typed expressions + SI dimensions
  - global schedule + effects
  - events, topology, functions, views
        |
        +--> Rust validator + capability report
        |
        +--> Reference executor
        |
        `--> Effect/topology planner --> model-specific Rust AOT
                                      --> persistent std::thread executor
        |
        v
results.bin / event sidecar / summary.json
        |
        v
NumPy memmap/view --> Brian2 state and monitor objects
```

### 3.1 B2IR v1

冻结契约包括：

- 任意数量 population/Synapses 的独立 schema、clock 和 endpoint domain；
- canonical `(slot, order, name, id)` 全局顺序及 RAW/WAR/WAW effect 验证；
- `f32/f64/i32/i64/u32/u64/bool/index/tick` 类型与七维 SI dimension 推导；
- 命名 EventStream、custom event、EventMonitor、pathway、fixed/expression refractory；
- portable `b2ir-function-v1` 与稳定 C11 native ABI adapter；
- Definition/Instance/Run 独立哈希、绝对 tick、RNG identity 和 pending-event continuation；
- v34-v37 旧 probe 只能通过显式验证迁移进入 v1。

权威格式为 `B2IR.md`，冻结证据为 `B2IR_FREEZE_REPORT.md`，架构决策为 `docs/adr/0001-freeze-b2ir-v1.md`。

### 3.2 执行与并行

AOT 不是按示例名称优化。生成器根据 effects 和 topology 自动判定：

- 独立 neuron/state/threshold 分片；
- target-owned `on_pre`，同一 target 始终由同一 owner 更新；
- heterogeneous delay 的 incoming CSR 与 owner-local queue；
- degree-balanced target partition；
- 多 projection/source-clock event dispatch 融合；
- event-driven pre/post plasticity 的独立 edge/endpoint 分片；
- post-summed target-owner reduction 与 final-only 可观测性优化；
- 无法证明安全或工作量过小时自动使用严格串行路径。

这使优化对普通用户透明，并让未来 CPU、Metal/CUDA 与 MPI planner 可以复用同一 effect/topology 事实。

### 3.3 结果与生命周期

`b2-result-dump-v3` 保存类型化 population 轨迹、spike、最终状态、不应期、突触状态和计数。热循环只写内存，运行结束后顺序写一个二进制文件；Python 校验 header、shape、有限性、事件顺序和 footer 后建立 memmap/view。

已支持连续多段 `run()`、绝对时间、跨段 delay queue、monitor append、counter RNG 延续，以及内存/文件 `store()/restore()`。`recording_window_steps=N` 可把原生缓冲、dump 和 Python 公开窗口限制在最近 N 个 tick。完整历史的 streaming disk sink 尚未实现。

## 4. 功能进度

### 4.1 已完成或进入回归契约

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| 异构 population | 完成 | 独立方程、dtype、dt、threshold/reset/refractory、monitor 与 CodeObjectSpec |
| 对象数量限制 | IR 层解除 | 当前预算为合计最多 1,000,000 neurons、每组最多 32 个可变状态；不再硬编码两组/八投影 |
| ODE | 核心完成 | `euler`、`rk4`、`exponential_euler`；默认 deterministic method selection 可落到 exact/euler/heun 抽象语句 |
| 数据类型 | 核心完成 | f32/f64 数值状态及整数/布尔 SoA、显式转换与原生宽度 dump |
| 不应期 | 完成 | fixed、expression/boolean/per-neuron、`unless refractory` |
| Synapses | 核心完成 | self/cross group、subgroup endpoint、mutable/constant/shared state、clock/event-driven ODE |
| pathway/delay | 完成 | 多 pre/post pathway、独立 scalar/per-edge delay、跨 run pending event |
| summed variables | 受限完成 | full population 与连续 subgroup，target-owner reduction |
| 随机与输入 | 核心完成 | counter RNG、rand/randn/poisson、PoissonGroup/Input、TimedArray、SpikeGeneratorGroup |
| 事件 | 核心完成 | 命名 custom event、run_on_event、EventMonitor、event sidecar |
| linked variable | 受限完成 | 同尺寸、连续 subgroup、固定数组和本地整数动态索引 |
| user Function | 受限完成 | portable pure expression 与 content-addressed C ABI AOT 实现 |
| monitor | 核心完成 | StateMonitor、SpikeMonitor、EventMonitor、多段 append、rolling window |
| procedural topology | 完成 | fixed-total、逐边 initializer、heterogeneous delay、binary CSR |
| report/profile | 完成 | capability report、粗粒度 profile 和可选 phase profile |
| checkpoint | 完成 | RNG、状态、monitor 和 pending delay queue 可恢复 |

### 4.2 仍未完成

| 缺口 | 影响 | 建议优先级 |
| --- | --- | --- |
| AOT 任意 `when/order` 与全部 `run_regularly` slot | reference 可表达，但 AOT 仍需 slot-driven codegen | P0/P1 |
| run 之间的完整 mutation 契约 | 自动路径已有部分能力，排队 build 仍受限 | P0 |
| string/generator/state-dependent `connect()` | 普通复杂 Brian2 构网不能直接迁移 | P1 |
| `PopulationRateMonitor`、SpikeMonitor 自定义变量 | 常用分析 API 缺口 | P1 |
| non-identity self-link、Synapses endpoint linked alias | linked-variable 高级语义不完整 | P1 |
| 显式 exact/rk2/heun/milstein/GSL 与 SDE | 数值方法覆盖仍小于 Brian2 | P1/P2 |
| Synapses-to-Synapses、structural plasticity | 高级突触拓扑/运行期改图未支持 | 后续 |
| `SpatialNeuron`/morphology | 多区室模型未支持 | 后续 |
| 全历史 streaming disk sink/lazy reader | 超长记录仍只能 rolling window 或一次性回填 | P2 |
| GPU backend | IR 已预留 CUDA/Metal/WGSL implementation descriptor，但没有正式 kernel/runtime | 下一主线 |
| MPI/cluster backend | 尚未实现 NUMA-aware 单机之后的跨节点 planner/runtime | 后续主线 |

## 5. 正确性与静态质量

### 5.1 B2IR 冻结门

冻结报告记录三台机器均使用原生 rustc 1.98.1 / LLVM 22.1.8：

| 主机 | Python 测试 | 参数化 subtest | Rust gate | 结果 |
| --- | ---: | ---: | --- | --- |
| M3 MacBook Air | 174/174 | 121/121 | clippy `-D warnings` | PASS |
| M1 Ultra Mac Studio | 174/174 | 121/121 | 同源验证 | PASS |
| 双路 EPYC Linux | 174/174 | 121/121 | rustfmt + clippy | PASS |

后续 `775583a7` 完成 canonical float、artifact identity、migration envelope、refractory effect 和 AOT ordering/fusion 缺口；提交记录的完整验证为 217 tests + 133 subtests、6 Rust tests、clippy、rustfmt。当前 `c396b5d8` 又增加 FlyWire replay 有界存储，定向测试 6/6 通过。

### 5.2 正确性策略

- deterministic 模型比较完整轨迹、最终状态、事件 tick/index/count、refractory 和 monitor 时间；
- reference/AOT 对同一 B2IR 要求严格一致，跨 C++/NumPy 使用明确数值容差；
- 随机模型要求固定 seed/worker 可复现和统计分布一致，不伪称复制 C++/NumPy RNG 位流；
- 并行路径验证 1/N worker 结果一致；
- malformed IR、单位错误、shape/alias/effect/delay 错误和 artifact 篡改均 fail closed。

## 6. 三机统一核心性能套件

统一套件每个场景 5 次测量，覆盖 scale、two-population、events、CUBA、COBAHH、STDP、Litwin-Kumar 派生 plasticity、PoissonGroup、PoissonInput，并以相同线程数比较 Rust worker 与 Brian2 C++ OpenMP。表中 `C++/Rust > 1` 表示 Rust 更快。

### 6.1 环境和门结果

| 主机 | CPU/内存 | 串行门 | 并发门 | 备注 |
| --- | --- | ---: | ---: | --- |
| M3 MacBook Air | 4P+4E，16GB | 23/24 | PASS | 测试期间 Chrome/renderer 高负载；STDP 异常样本不作为发布基线 |
| M1 Ultra Mac Studio | 20 cores，128GB | 24/24 | PASS | 安静主机，原生 ARM64 rustc |
| 双路 EPYC 9454 | 96 physical/192 logical，约 1.1TiB | 23/24 | PASS，1-48 threads | 串行 PoissonInput 唯一缺口 |

### 6.2 代表性串行中位数

| 主机 | 模型 | Rust ms | C++ ms | C++/Rust | 门 |
| --- | --- | ---: | ---: | ---: | --- |
| M3 | CUBA | 7.428 | 8.771 | 1.18x | PASS |
| M3 | COBAHH | 3,067.237 | 4,155.870 | 1.35x | PASS |
| M3 | STDP | 815.139 | 701.233 | 0.86x | **NOISY FAIL** |
| M3 | Litwin-Kumar short | 15.293 | 124.807 | 8.16x | PASS |
| M3 | PoissonGroup | 360.048 | 1,667.810 | 4.63x | PASS |
| M3 | PoissonInput | 3,605.778 | 4,166.790 | 1.16x | PASS |
| M1 Ultra | CUBA | 5.151 | 6.444 | 1.25x | PASS |
| M1 Ultra | COBAHH | 3,155.521 | 4,122.370 | 1.31x | PASS |
| M1 Ultra | STDP | 111.175 | 121.935 | 1.10x | PASS |
| M1 Ultra | Litwin-Kumar short | 3.213 | 56.052 | 17.44x | PASS |
| M1 Ultra | PoissonGroup | 116.992 | 774.630 | 6.62x | PASS |
| M1 Ultra | PoissonInput | 1,302.302 | 1,910.600 | 1.47x | PASS |
| EPYC | CUBA | 8.876 | 10.950 | 1.23x | PASS |
| EPYC | COBAHH | 4,650.407 | 10,843.200 | 2.33x | PASS |
| EPYC | STDP | 159.330 | 250.977 | 1.58x | PASS |
| EPYC | Litwin-Kumar short | 8.536 | 81.505 | 9.55x | PASS |
| EPYC | PoissonGroup | 164.657 | 384.664 | 2.34x | PASS |
| EPYC | PoissonInput | 1,430.652 | 1,310.920 | 0.92x | **FAIL** |

同机安静的 B2IR freeze paired gate 上，M3 STDP 为 93.470ms 对 99.359ms，Rust 快 1.06x。因此统一套件中的 815ms 异常不能归因于 STDP 代码回归，必须在关闭交互负载后重测。

### 6.3 最高线程档位

| 主机/线程 | 工作负载 | Rust ms | C++ ms | Rust 自身扩展 | C++/Rust |
| --- | --- | ---: | ---: | ---: | ---: |
| M3 / 8 | CUBA100k | 203.863 | 998.916 | 1.81x | 4.90x |
| M3 / 8 | COBAHH | 1,473.933 | 4,234.980 | 2.02x | 2.87x |
| M3 / 8 | PoissonInput | 693.787 | 1,335.390 | 2.29x | 1.92x |
| M1 Ultra / 16 | CUBA100k | 20.014 | 697.293 | 6.04x | 34.84x |
| M1 Ultra / 16 | COBAHH | 277.926 | 9,046.550 | 11.42x | 32.55x |
| M1 Ultra / 16 | PoissonInput | 91.858 | 571.523 | 14.25x | 6.22x |
| EPYC / 48 | CUBA100k | 103.967 | 252.885 | 2.04x | 2.43x |
| EPYC / 48 | COBAHH | 561.086 | 1,605.080 | 8.30x | 2.86x |
| EPYC / 48 | PoissonInput | 66.203 | 68.989 | 21.61x | 1.04x |

M1 Ultra 的结果证明通用 target-owned 和 work-aware 设计可以显著超过 C++ OpenMP。EPYC 上 CUBA 在约 16 线程后退化、COBAHH 在约 32 线程附近饱和，说明全局 barrier、NUMA ownership、queue locality 和跨 socket memory traffic 仍限制扩展。

## 7. 业内模型与大规模工作负载

### 7.1 Potjans-Diesmann 2014

统一 Device 版本使用 8 populations、55 projections、77,169 neurons、298,880,968 recurrent synapses、0.1ms dt。本轮运行 1,000ms，不记录全量 spikes；Rust 共投递 942,223,086 synaptic events。

| 主机 | Rust threads | Rust simulation | Rust end-to-end | C++ simulation | C++ end-to-end | Simulation speedup | Rust/C++ child RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 16GB | 4 | 19.138s | 32.970s | 未运行 | 未运行 | - | 3.86GiB / - |
| M1 Ultra | 8 | 4.409s | 19.029s | 23.580s | 74.250s | 5.35x | 6.96 / 18.89GiB |
| EPYC | 8 | 5.727s | 22.527s | 37.787s | 109.208s | 6.60x | 6.92 / 18.43GiB |

M1 Ultra 与 EPYC 上，Rust end-to-end 分别快 3.90x 和 4.85x，native child RSS 分别低 2.71x 和 2.66x。M3 上 Rust 完整运行成功；没有强行启动 C++，因为另外两台机器实测 C++ child peak 超过 18GiB，不能把“预计超出 16GB”写成一次实际 OOM 结果。

与 NEST 3.10 的既有 PD14 对比没有在本轮重跑：Rust 在 M1 Ultra 上快 1.99x，但 NEST 在 EPYC 96 核最优点仍快 6.44x。这是当前 CPU 扩展路线最重要的反例和优化目标。

### 7.2 Brunel asynchronous irregular 网络

规模为 12,500 neurons、15,625,000 recurrent synapses、1.2s biological time，最后 0.2s 用于统计。

| 主机 | Rust/C++ threads | Rust end-to-end | C++ end-to-end | 加速 |
| --- | --- | ---: | ---: | ---: |
| M3 | 4 / 1 | 5.190s | 18.902s | 3.64x |
| M1 Ultra | 8 / 1 | 4.121s | 21.398s | 5.19x |
| EPYC | 8 / 1 | 3.626s | 53.318s | 14.70x |

Rust/C++ 的 mean rate 为 37.550/37.481Hz，ISI-CV 为 0.354/0.357，0.1ms population-count Fano 为 23.884/23.605。不同 RNG 不要求 spike 位流相同，这里验证的是模型相态与统计一致性。

### 7.3 Litwin-Kumar 派生长时程系统门

该测试是 40E+10I、`p=0.1`、dt=1ms 的生命周期/可塑性系统门，不是论文规模复现。两段各 500 生物秒，总计 1,000 秒；只保留最后 1,000 ticks。

| 主机 | Threads | 三段 native 时间 | Checkpoint | Result dump | 恢复重放 |
| --- | ---: | --- | ---: | ---: | --- |
| M3 | 4 | 0.073/0.076/0.074s | 28,971B | 15,778B | bitwise exact |
| M1 Ultra | 8 | 0.117/0.111/0.115s | 28,968B | 15,778B | bitwise exact |
| EPYC | 8 | 0.084/0.084/0.085s | 28,962B | 15,778B | bitwise exact |

它证明长时间轴、rolling monitor、checkpoint 和多段恢复成立；不能据此宣称已复现 Litwin-Kumar 或 Zenke 的科学结果。

### 7.4 FlyWire full weighted topology

工作负载使用真实 FlyWire v783 加权拓扑：139,255 neurons、15,091,983 directed weighted edges、54,492,922 biological contacts；神经动力学是工程化 homogeneous excitatory LIF，并非拟合的果蝇全脑生物物理模型。每次 replay 比较全部 final state/refractory、16-neuron traces、所有 spike ID/time/count。

| 主机 | Threads | Rust | C++ | C++/Rust | Rust/C++ RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| M3 | 1 | 2.867s | 4.551s | 1.59x | 304 / 806MiB |
| M3 | 4 | 2.728s | 5.035s | 1.85x | 309 / 818MiB |
| M3 | 8 | 3.152s | 5.750s | 1.82x | 320 / 833MiB |
| M1 Ultra | 1 | 2.469s | 3.891s | 1.58x | 348 / 810MiB |
| M1 Ultra | 4 | 1.700s | 4.698s | 2.76x | - |
| M1 Ultra | 8 | 1.081s | 6.235s | 5.77x | - |
| M1 Ultra | 16 | 0.751s | 8.665s | 11.54x | 382 / 890MiB |
| EPYC | 1 | 4.014s | 7.164s | 1.78x | 300 / 752MiB |
| EPYC | 4 | 2.118s | 7.441s | 3.51x | - |
| EPYC | 8 | 1.423s | 6.142s | 4.32x | - |
| EPYC | 16 | 0.867s | 5.968s | 6.88x | 318 / 793MiB |

M3 本轮 spread 为 40%-48%，只能作为方向性结果；M1 Ultra 16-thread Rust、EPYC 4/8-thread C++ 也超过 15% 稳定性阈值。性能倍数必须与原始 spread 一起解释，不能选择单次最优值宣传。

## 8. 存储问题与本轮修复

旧 FlyWire benchmark 会在一个 `TemporaryDirectory` 中保留所有 warmup 和 measured replay 的大型 C++ result 目录，磁盘需求随 repeats 和 thread levels 线性增长。本机三次运行因此在完成前写满磁盘。

`c396b5d8` 的修复在每次 replay 完成计时、RSS 读取和全量正确性比较之后删除该轮 result directory。profile 等需要读取结果的调用默认仍保留目录；标准 benchmark/replay 显式启用 cleanup。修复后完整 M3 1/4/8 threads x 5 repeats 成功，定向测试 6/6 通过。

经用户确认，本轮还清理了：

- `/private/tmp` 中 82 个旧 Brian2/B2IR/Brunel/PD14 目录，约 6.68GiB；
- 仓库 `brian2-rust/output` 中 8,444 个 `device-*`、17 个 `benchmark-*` 和已识别的历史大型 build/benchmark 目录；
- `brian2-rust/output` 从约 19GiB 降到 1.6GiB，系统可用空间回升到约 26GiB。

被删内容不可恢复，但均可由源码和数据集重建。保留了 FlyWire 原始/验证数据集、PD14/FlyWire 源数据、本轮三机 JSON/Markdown 报告和当前源码。

## 9. 版本一致性与报告边界

三机统一 24-scenario 核心套件启动时以 `bfae9880` 为冻结基线：M1 Ultra 是 clean checkout；EPYC 的复制目录没有 Git metadata，但内容来自同一 snapshot；M3 运行期间工作树已包含后来提交为 `775583a7` 的 B2IR 完整性改动，并受交互负载污染。

因此本报告把以下证据分开：

1. `bfae9880` 三机完整性能基线；
2. `b2ir-v1` freeze report 的三机 correctness 和 paired CUBA/COBAHH/STDP 性能门；
3. `775583a7` 的完整测试记录；
4. `c396b5d8` 的本机 FlyWire 完整 replay 和有界存储测试。

在打 release tag 前，应在安静主机上将当前 HEAD 的统一 24-scenario 套件重新跑遍三机。当前数据足以评估架构方向和定位热点，但不应被描述为 `c396b5d8` 的完整三机发布认证。

## 10. 下一阶段建议

1. **冻结当前 HEAD 的 clean 三机基线。** 重跑统一套件，优先复核 M3 STDP 和 EPYC serial PoissonInput。
2. **NUMA-aware CPU executor。** per-core target shard、socket-local queue/first-touch、分层 barrier、压缩 spike exchange；目标是缩小 EPYC PD14 对 NEST 的 6.44x 缺口。
3. **消除剩余 AOT scheduler 缺口。** slot-driven codegen 覆盖任意合法 `when/order`、`run_regularly` 和 monitor/event placement，同时继续由 Effect Algebra 证明融合合法性。
4. **完成高频 Brian2 兼容项。** generator/state-dependent connect、PopulationRateMonitor、SpikeMonitor variables、linked endpoint alias、更多 deterministic/SDE integrator。
5. **正式 bounded streaming sink。** rolling window 保持现状语义；新增保留全历史但边运行边落盘的独立 sink/reader 契约。
6. **GPU 作为一等后端。** 先冻结 LogicalPlan/Kernel ABI、SoA/CSR buffer 和 event queue contract，再实现 Metal（Apple）与 CUDA；同一 B2IR 和 correctness corpus 对比 GeNN/Brian2GeNN。
7. **MPI 最后叠加。** 在单机 NUMA owner 模型稳定后，把同一 target ownership 扩展为 rank-local shard；先解决 capacity，再以通信/计算比决定 speed 模式。

## 11. 阶段结论

Rust standalone 已经证明三件关键事情：Brian2 用户模型可以在不修改前端主循环的情况下被结构化 lowering；异构 population、事件、可塑性和大规模 procedural topology 可以进入一个稳定 IR；精心设计的内存布局与 target-owned 并行在 Apple Silicon 和中等线程数上能显著超过 Brian2 C++ standalone。

下一阶段不应继续堆积模型名称特例。最高价值工作是让统一 scheduler/effect/topology planner 在 NUMA CPU、Metal/CUDA 和未来 MPI 上产生不同 physical plan，同时保留同一 B2IR 语义与测试 corpus。项目已经跨过“能否实现”的门槛，当前挑战是 clean release engineering、长尾兼容和多核/多设备扩展效率。

## 12. 原始证据位置

- 本地 M3：`/private/tmp/brian2-stageperf-results-20260906/`
- 收集的 M1 Ultra：`/private/tmp/brian2-stageperf-collected-20260906/m1ultra/`
- 收集的 EPYC：`/private/tmp/brian2-stageperf-collected-20260906/epyc/`
- B2IR 冻结报告：`brian2-rust/B2IR_FREEZE_REPORT.md`
- 用户兼容矩阵：`brian2-rust/COMPATIBILITY.md`
- PD14 详细报告：`brian2-rust/PD14.md`、`PD14_COMPARISON.md`、`PD14_LINUX_COMPARISON.md`、`PD14_NEST_COMPARISON.md`

