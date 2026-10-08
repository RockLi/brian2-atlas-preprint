# ExecutionPlan v0 与 Apple Metal 初版

2026-09-06；分支 `codex/execution-plan-research`。Plan 是 backend-private 实验接口，
不修改冻结的 B2IR v1，也不冻结跨语言或跨 GPU 的 kernel ABI。

## 执行管道

```text
Brian2 → B2IR v1 → 原 Rust 完整性/语义验证
                       ↓
              LogicalPlan（canonical nodes、完整 effects、依赖、clock activation）
                       ↓
       ┌───────────────┴────────────────────┐
       ↓                                    ↓
CpuPhysicalPlan                         MetalPlan / explicit float32
compact-v6 / general-v1 / slot-v1        时间融合 kernel / coupled target-owned DAG
       ↓                                    ↓
Rust AOT + instance.bin                 MSL runtime compile + Metal command buffer
       └───────────────┬────────────────────┘
                       ↓
           结果验证、Device 回填、RuntimeBinding
```

`plan.py` 定义不可变 plan、验证和 explain；`planner.py` 负责 CPU 选择及合法性判断；
`native.py` 消费 plan 输出 Rust；`slot_codegen.py` 发射 canonical 节点；`metal.py`
、`metal_dag.py` 和 `metal_runtime/bridge.m` 实现 GPU。reference executor 保持独立。

## API

```python
import brian2 as b
import brian2_rust

b.set_device("rust_standalone", engine="aot", threads=4)
# 创建并运行 Network 后：
print(b.get_device().explain_plan())
report = b.get_device().explain_plan(format="dict")

# 已导出的模型也可只验证/规划，不执行 simulation：
plan = brian2_rust.build_execution_plan(model)
print(brian2_rust.explain_plan(plan))
```

编译前 explain 标为 runtime unbound。成功运行后 Device 的 explain 附带真实线程数、
亲和性、并行路径和 timing；`runtime-binding.json` 保存这些观察值。没有采集的 owner map
和 queue capacity 明确为 `not_instrumented`，不推算 NUMA 实际驻留。

`ExecutionPlan` 的 layer hashes 绑定 definition、instance、run；policy hash 不包含
instance 值身份，但包含实例影响的选择、布局尺寸和完整逻辑图。发射前重新派生并比较
全部 plan，拒绝被修改的节点、选择或模型。JSON 用于检查，不支持把任意 JSON 当作可信
可执行计划。外部输入必须使用 `build_execution_plan` 的独立 Rust 验证边界；内部
`verify_execution_plan` 不替代 B2IR 语义验证。

`native/execution-plan.json` 与 manifest 的 plan/source hashes 对应。实例替换继续检查
Definition/Run、生成源码和 native Function 身份，并新增 Rust 语义验证及 plan policy
一致性；旧 artifact 缺少 policy 身份时要求重建。运行期线程设置每次重新绑定。

## CPU 行为

- compact/general 路径消费原有融合、route、target ownership、work threshold、summed
  和 emitter 选择。合法模型的既有优化源码通过旧生成器的逐字节对照。
- 当 fixed-phase 不能证明等价，或不支持某个 slot 时，选择 `slot-v1`：按验证后的
  canonical 节点和 active clock 生成串行 Rust 代码，支持前端已导出的合法 `when/order`。
  这条路径可能比优化路径慢，explain 会说明选择原因。
- 延迟队列保持 pathway、事件和 edge 顺序；分段运行依赖的 EventStream 独立于是否
  存在 SpikeMonitor。当前 procedural topology 的既有不可续跑限制仍适用。
- 修复了 final-only summed 漏查跨 population linked reader 的问题。最小反例原 AOT
  轨迹为 `[0,0,0,0]`，正确轨迹为 `[0,1,3,6]`；现在按 completed resource reads 判定。
- 内存报告为已声明数组 payload 和生命周期清单，动态队列、记录开销及 topology layout
  分别标注。没有实现 arena reuse，也不声称精确预测 Peak RSS。

## Apple Metal 范围与数值契约

```python
b.set_device("rust_standalone", engine="metal", numeric_mode="float32")
# 低发放率场景可显式测试稀疏投递；默认 event_delivery="scan"。
b.set_device("rust_standalone", engine="metal", numeric_mode="float32",
             event_delivery="sparse")
```

必须显式选择 `float32`。B2IR/CPU 原有 `reference-f64` 不变；即使公开 Brian 数组为
f64，GPU 算术仍标为 `b2-metal-f32-v0`。不能将扩大存储位宽误认为 f64 计算。

初版支持相互独立的 population、f32/f64 输入、常用浮点表达式、纯 portable Function、
state update/threshold/reset、固定及表达式 refractory、可使用独立 clock 的 run_regularly（通过 DAG 调度）、
StateMonitor、SpikeMonitor 及 Device 分段执行/store/restore。每 population 可有多个
StateMonitor，沿用 v1 在首个监控节点记录变量/索引并集的规则；不同 population 可以使用不同 clock。一个 GPU lane 完成一个 neuron 的
全部 ticks；需要独立代码 clock 或跨 lane 依赖时使用 DAG。

有连接模型现已支持 **多 clock、显式拓扑、整数 tick 延迟的 pre spike pathway、不可变突触状态**，
以及写入 post 状态的 `summed`。支持循环连接、子组、重复边、多 pathway 和空连接。
每个目标 neuron 独占写入，事件按 source/edge 顺序投递，summed 按连接创建顺序累加。
可写突触状态、pre/post pathway、event-driven 塑性、突触积分及 pre/post summed
采用按效应选择的并行策略：独立边更新/路径按 edge 分派，写目标神经元的 eligible pre
路径按 target 分派，独立 summed 按端点分派；存在跨目标依赖或共享写入时保留
canonical GPU 顺序。事件整数预留和稳定边顺序保持浮点累加顺序。
procedural topology、binary CSR、参数和延迟 initializer 先经共享 Rust host preparation
物化，再在 GPU 上模拟；沿用冻结 v1 的单 activation 限制。当前功能与验证索引见
[GPU_IMPLEMENTATION_STATUS.md](GPU_IMPLEMENTATION_STATUS.md)。

`MetalPlan.strategy` 自动选择 `independent-temporal-fusion` 或
`canonical-target-owned-dag`。后者记录 kernel→logical nodes、buffer bindings、dtype
和保守依赖链；只融合相邻且属于同一 population/clock 的步骤，每 tick 保留 canonical
顺序。缓冲区在一次 run 内复用，每 64 ticks 提交一批，并以显式 buffer barrier 同步
阶段。独立于 SpikeMonitor 的 EventStream 写入 `events.bin`；零延迟和延迟模型的 Device
分段运行和 store/restore 已验证。记录窗口内的事件布尔矩阵额外需要 N×window bytes。

Metal/CUDA 已支持静态及周期 `SpikeGeneratorGroup` 输入：每个神经元通过 int64
稀疏事件表查找绝对 tick，输入存储为 O(N + spikes)，不经过 float32 时间比较。
独立时钟、空输入、延迟/可变突触、排队 build 和 pending store/restore 的实测见
[GPU SpikeGenerator 验证](execution-plan-evidence/gpu-spike-generator/README.md)。耦合 DAG
现已支持耦合多 clock；各 stage 按自己的整数 tick 执行，主机端沿用 Rust 的
f64 时间合并规则。突触积分仍使用源 clock，每段运行边界须对齐所有 clock。

整数和布尔状态现已保留原始位模式，64 位整数使用两个 32 位槽；监控、参数、
延迟塑性和结果传输使用一致的类型布局。整数运算遵守回绕与 floor 除法规则，
除零会拒绝结果。类型状态沿用按效应选择的 edge/target/canonical 策略，浮点表达式
仍遵守显式 float32 模式。详见 [类型存储验证](execution-plan-evidence/gpu-typed-storage/README.md)。

linked variables 已支持 identity、constant、整数 state/parameter 索引。GPU gather
在每个消费代码块或 monitor 前绑定输入，保持原始整数索引、输入快照与越界错误。
非 identity 自链接使用单 GPU lane 保持 reference 的批次/reset 提交顺序。
详见 [链接状态验证](execution-plan-evidence/gpu-links/README.md)。

population 的 `run_regularly(dt=...)` 可以使用独立 clock，各 stage 的 t/dt/整数 tick
来自自己的时钟，监控和 population 存储仍绑定 population clock。参见
[独立代码时钟验证](execution-plan-evidence/gpu-regular-clock/README.md)。
原生 Metal/CUDA Function source adapter 已在 M3、L4、A100 验证，见
[GPU_FUNCTIONS.md](GPU_FUNCTIONS.md)。显式单 workgroup 实验见
[GPU_WORKGROUP.md](GPU_WORKGROUP.md)；通用跨 workgroup 常驻同步、跨节点和 NUMA
优化仍属后续工作。

`EventMonitor(group, 'spike', variables=[...])` 已由专用 DAG stage 按其 when/order
记录状态和参数，支持阈值之前的旧 fired 标记、突触阶段和 reset 之后的变量值。
每个 monitor 保留独立的事件和值，写入既有 typed `events.bin`；没有突触时也可记录。
原始 EventMonitor 数据保留完整 activation，Device 再应用公开的记录窗口。
其 dense 采样容量为 N×steps×变量数，分配前检查 buffer 上限。
独立 StateMonitor clock 与非 start 槽不属于冻结 v1，仍须前端/IR/reference 协同扩展。

命名事件现已使用独立 flags 和历史通道，支持各事件的 threshold、reset、EventMonitor
及 pre/post pathway。spike 保留第 0 通道，其 refractory 更新不作用于其他事件。
各 pathway 按自己的事件阈值计算采样偏移；在阈值之前执行时，消费上一 tick
的该事件标记。正延迟和零延迟均已支持，并按偏移重建跨 run 的待投递事件。
自定义事件即使没有 spike、synapse 或 EventMonitor，也通过 DAG 保留独立事件流。

表达式 refractory 已支持持续时间表达式与布尔保持条件。状态更新使用本 tick 新的
`not_refractory`，冻结目标变量的突触写入遵守同一 gate；绝对 elapsed tick 通过整数
状态续接。负值/越界 `timestep()` 会拒绝输出结果，保持 B2IR 的布尔操作数求值、语句
条件屏蔽和无 spike 时 reset 标量检查的语义。验证见
[GPU refractory 测试](execution-plan-evidence/gpu-refractory/README.md)。

`event_delivery="scan"` 每 tick 扫描入边；新增 `"sparse"` 只从 fired source 展开出边，
用整数 atomic reservation 填入目标队列，再按 canonical source/edge rank 排序后执行。
每个目标本轮事件数不足入度的 1/4 时排序紧凑队列；更密集时消费端直接扫描原 CSR，
避免密集排序，但已发生的源端展开仍有成本。默认保留 scan，因为密集发放及当前循环
网络实测中 sparse 可能更慢。两个策略共享同一浮点 lowering，并逐位比较结果。

稀疏队列容量恰为每目标入度：每源每 pathway/tick 最多出现一次；消费后清零计数，
跨 pathway/tick 重用。额外数组 payload 为每 projection
`4 * (3*edges + source_count + target_count + 1)` bytes。该上界依赖当前零延迟限制，
延迟 pathway 使用下面的独立路径。`MetalPlan.event_delivery`、dispatch roles、bindings
及 `elided_nodes` 参与计划 hash；空且无 refractory 的 state update 可删除，再融合
相邻的同 population 步骤。无事件的 population 不返回虚构的 spike EventStream。

延迟路径支持统一延迟、逐边延迟及已有 pending 事件。每 pathway 在 canonical 位置将
source fired 写入环形历史，长度 `R=min(max_delay+1, max(1,run_steps))`；目标线程按
delay 降序、source、边创建顺序扫描入边。固定到达 tick 下，这等价于按入队 tick 升序
投递。已有 pending 先按原顺序消费，统一 delay 的 source item 会稳定展开成边。
这保留了浮点累加顺序，不能用仅按 source rank 的零延迟队列替代。

`scan` 延迟路径选择 `delayed-target-scan`，并增加 `delay-history-enqueue` stage。
每 pathway 额外 GPU
payload 为 `Ns*R + 8*E + 12*P + 8*Nt + 4` bytes，P 是展开后的 pending 边事件数；
空数组的最小分配及 host 临时排序另计。分配 ring 和 pending 前检查大小，超过每 buffer
预算即拒绝。延迟大于单段长度可由 Device 保留未来事件，分段、store/restore、ring wrap
及滚动窗口已验证；截断记录窗口必须覆盖最大 delay，沿用前端的检查。

`sparse` 延迟路径现按每个 source 的实际 delay 分组，检查该组对应的历史事件，
只展开活跃组的出边。目标队列保存 delay 降序/source/边创建顺序的 rank；少量到达时
排序紧凑队列，密集到达时消费端回到有序入边扫描。每边在固定到达 tick 下最多对应
一个本段 emission，所以队列容量仍为入度；已有 pending 独立消费，不占此容量。
消费后清零计数，允许与零延迟 pathway 共享 projection 的 scratch 队列。

新增 delay 分组表每 pathway 需 `4*Ns + 8*G + 8*E + 8` bytes，G 为实际 source/delay
组数，最坏为 E；另使用前述 sparse projection 数组。分组很多或密集发放时，检查和
展开成本仍可能超过扫描，默认继续使用 scan。实际选择写入 dispatch roles：
`delay-source-group-enqueue` / `delayed-target-sparse`。

当 source population 与入队相邻、source_start=0 且覆盖整个 population 时，可合并为
`population-delay-source-enqueue`：每个 lane 在更新自己的 fired 后读取同一标志与
历史行，不需要跨 lane 屏障。目标消费仍是独立 dispatch，并保留 buffer barrier。
子组和不相邻的调度不做这项融合。LogicalPlan 的节点顺序保持不变。

同一 `MetalExecutor` 的 DAG replay 会缓存验证后模型的初始缓冲区，复用拓扑排序、
delay 分组和数值解码结果。每次执行按实际 dispatch 的可写 bindings 复制工作数组，
返回的突触状态和 refractory 数据也独立复制；CPU-f32/GPU 交替执行不会污染缓存。
改变模型需创建新 executor。每次调用仍检查 `max_buffer_bytes`，首次准备时间以
`storage_preparation_seconds` 单独返回，缓存命中时为 0；编译和 GPU 数据复制另计。
这降低 warm replay 的准备成本，**不代表新 Device run 或首次编译没有启动开销**。
Device 当前每段仍创建新的 executor。缓存、工作副本、GPU buffer 和结果一起占用
内存，因此每 buffer 限额仍不能解释为整个进程的内存预算。

为保证 pending 重建正确，**非零延迟或带 pending 的 pathway 必须排在源 threshold
之后**；提前消费上一 tick 事件的延迟调度暂时明确拒绝。原有零延迟、无 pending 的
pre-threshold 路径继续支持。精度契约仍为显式 f32，延迟支持不改变阈值判定。

`build_execution_plan(..., backend="metal", numeric_mode="float32", event_delivery="sparse")`
和 `MetalExecutor(..., numeric_mode="float32", event_delivery="sparse")` 可直接选择同一路径；
CPU 拒绝该选项。多阶段同步及 gather 仍可能超过 Rust 的事件驱动 CPU 路径；输入准备、
记录和读回开销也需要单独衡量。

使用 macOS 的 Metal framework 运行时编译，无需独立 `xcrun metal` 工具。没有可用设备
时明确失败。每 buffer 默认限制 512 MiB；这不是整个进程的总内存预算。

指数与 exponential-euler 的稳定形式经过独立 reference 数值测试；同一 float32 kernel
主体另编译为 C++ CPU 对照（非 Rust f32 后端），区分精度/时间融合带来的收益与 GPU 收益。
Metal 算术的 subnormal 下溢按设备行为归零；共享 exp 对照显式应用相同规则。
不承诺 float32 的 spike 时间与 f64 永远一致。

## 验证与复现

```sh
cargo build --release --manifest-path brian2-rust/Cargo.toml --offline
PYTHONPATH=brian2-rust/python:. python -m pytest brian2-rust/tests -q
B2_TEST_METAL=1 PYTHONPATH=brian2-rust/python:. python -m pytest brian2-rust/tests/test_metal.py brian2-rust/tests/test_metal_dag.py brian2-rust/tests/test_metal_delays.py -q
PYTHONPATH=brian2-rust/python:. python brian2-rust/examples/metal_benchmark.py \
  --output /tmp/b2-metal-benchmark --model hh --neurons 16384 --steps 200 \
  --repeats 7 --threads 1,4,8
```

Apple GPU 测试必须在允许访问 GPU 的进程中运行。编译、warm-up 不计入 simulation 时间；
报告另外记录构建与完整 replay 时间。正确性门要求 GPU/CPU-f32 状态与轨迹位模式一致，
spike/refractory 一致；原 Rust f64 的严格比较与逐变量误差、spike 数和时间偏移独立报告。
性能门还要求最慢一次 GPU 比所有被测 CPU 配置的最快样本快，不能仅凭一个有利中位数。

默认 HH 基准是 **独立 HH ionic dynamics**，不是 recurrent COBAHH。
新增 `--model coupled --degree 64` 测试循环 LIF 网络、零延迟投递及有实时读取者的 summed。结果不能推广为所有网络
均快于 Rust CPU。测试和实测证据见本目录 `execution-plan-evidence/`。
另见 [M1 Ultra / macOS 14.5 验证](execution-plan-evidence/m1-ultra/README.md)，包含旧 SDK
编译兼容及无系统默认 GPU 时的设备枚举回退。

新增连网实测见 [Metal DAG 验证](execution-plan-evidence/metal-dag/README.md)：M3 性能门
失败；M1 Ultra 共享负载下约 1.52×，尚不能作为空闲机器的稳定加速比。

## 后续范围

原生 GPU 已接入 `rand`、`randn`、PoissonGroup 和 PoissonInput/binomial，
计划及结果明确记录 `b2-counter-f32-u24-v0`。counter 沿用 B2IR 的
seed/stream/整数 tick/index 定义；均匀数取 24 位，不能视为原 f64 随机值。
突触使用原始 edge 编号与投递 tick，并暂走 canonical 单 lane 路径。
正态及二项采样按 f32 运算；二项反演初始概率下溢时已接入原生 BTRS，
保留用户明确选择的正态近似路径，并限制拒绝次数。
见 [大参数二项验证](execution-plan-evidence/gpu-binomial/README.md)。直接 `poisson()` 已接入原生 product/PTRS 采样，
大参数接受率使用稳定的 deviance/Stirling 分解；参数与结果仍为显式 f32，
不保证与 f64 相同的逐样本结果。见 [Poisson 验证](execution-plan-evidence/gpu-poisson/README.md)。
验证范围和数值限制见 [随机输入验证](execution-plan-evidence/gpu-random/README.md)。

一维/二维 TimedArray 已接入两种 GPU：表值在同一 owner 内去重后放入现有
参数/突触值 buffer，GPU 自行按时间及列索引查表。支持动态 Poisson 速率和
延迟突触的投递时刻查表。时间、列索引表达式采用 f32；时间首尾钳制，
列越界会在发布结果前失败。表值须为有限 f32，epsilon 须为正的 normal f32。
不保证任意 f64 输入边界一致。见 [TimedArray 验证](execution-plan-evidence/gpu-timed-array/README.md)。

fixed-total、fixed-indegree 和 binary CSR 已接入 GPU 前的 Rust 主机初始化。
连接顺序、f64 参数采样及整数延迟沿用 reference 算法，模拟 tick 由原生 GPU 执行。
初始化以紧凑二进制数组传递，内容哈希写入计划及结果绑定，原始 B2IR 保持不变。
默认初始化导出数据上限 512 MiB，不代表主机 RSS 上限；初始化耗时另列。
现有过程化拓扑的一次 activation 与不可变 per-edge 参数约束仍保留。
GPU 常驻拓扑构造属于后续优化。见 [初始化验证](execution-plan-evidence/gpu-initialization/README.md)。

GPU 稀疏事件投递已作为可选路径落地，见 [四类负载实测](execution-plan-evidence/metal-sparse/README.md)。
延迟环形历史与 pending continuation 已落地，见 [延迟验证](execution-plan-evidence/metal-delays/README.md)。
稀疏延迟分组投递及相邻源端融合已落地，见 [改进实验](execution-plan-evidence/metal-delay-sparse/README.md)。
性能仍需与原 Rust CPU 配对评估；CUDA 已增加 executor 内缓冲区复用与 Graph 重放，
跨 Device activation 的输入准备和调度成本仍需优化。塑性已接通
canonical 串行 GPU 路径；耦合多 clock、整数状态与 linked variables 已接通。独立 monitor clock 与
完整表达式契约等 GPU 语义仍待实现。阈值之前的延迟 pathway 已补齐，
见 [提前 pathway 验证](execution-plan-evidence/gpu-pathway-order/README.md)。
新增 [CUDA / Modal 实验入口](CUDA_MODAL.md) 将受支持的计划生成 CUDA kernel，
在 NVIDIA GPU 上对照本地 f32 执行结果。现在另有 `engine="cuda"` 的正式 Device
入口，支持受限模型的运行、结果传输和 pending/store/restore；不能将这些
通过的用例等同于完整 GPU 功能覆盖。后续在同一 GPU 上比较 Brian2CUDA、
Brian2GeNN 和直接 GeNN 的模型，同时保留同机 Rust CPU 对照。
完整待办见 [GPU 实现状态](GPU_IMPLEMENTATION_STATUS.md)。
稀疏目标队列与并行前缀叠加延迟塑性、多时钟、类型/链接状态、命名事件和续跑恢复的
组合验证已在 M3、L4、A100 完成，各 20 项通过；具体数值门槛、原始失败及
离线数组复核见 [组合策略验证](execution-plan-evidence/composed-policies/README.md)。
另有 [位图投递实验](execution-plan-evidence/target-bitset/README.md)，在保留当前缓冲区和
调度的条件下省去当前事件队列排序，三端密集模型改善约 36–37%；静默收益混合，
现在可通过 `gpu_synapse_sparse="bitset"` 显式选择，`False` 默认和 `True` rank 队列
保持原行为。正式选项与缓存/计划验证见 [位图 API](execution-plan-evidence/bitset-api/README.md)。
现有突触/归约和多 kernel DAG 是受限的可执行路径。
显式逐 activation 的策略校准已接入 Device，支持完整结果门槛、编译前去重和激活内
内核缓存，见 [GPU 自动调优](GPU_AUTOTUNE.md)。默认关闭，调优总成本单列；可选精确输入决策缓存会校验新执行器的完整重放。
同次调优的语义校验已集中到不可变快照，每个候选仍独立推导计划、准备模型并校验完整结果，
见 [准备阶段优化](execution-plan-evidence/preparation/README.md)。
跨不同输入的策略选择、带成本模型的自动选择、NUMA 消融实验、生命周期 buffer alias reuse 与 MPI
仍为后续工作。研究阶段的论证保留在 [研究记录](EXECUTION_PLAN_RESEARCH.md)。

### GPU logical Tick 与突触时间表达式

`tick_offset` 与私有 Tick 临时变量已使用 int64 原生 lowering，防止超过
2²⁴ 后 `+1` 被浮点临时量吞掉。范围检查对齐参考执行器的 ±2⁵³ 合约及其
f64 边界舍入；时间输入和 `timestep()` 商仍属于显式 float32 模式。
突触直接表达式及 portable Function 内的时间运算选择 canonical GPU lane；
标量语句在每次调度时执行，即使没有到达事件也不会漏掉数值错误。
前端浮点 `v += timestep(...)` 现在显式插入 `tick_to_f64`，同时保留 reference/AOT
兼容性。详见 [Tick 验证记录](execution-plan-evidence/gpu-ticks/README.md)。

### HH 跨后端比较

`hh-ionic-v0` 已接入 32/4,096 细胞、512 tick 的同 L4 六路比较，另有 Metal
同输入验证。独立 f64 oracle 校验六个状态、逐 tick 轨迹和全部脉冲；各后端的
116/15,239 个脉冲时刻与索引一致，Metal/CUDA/CPU float32 的结果逐位一致。
所有 float32 实现仍未通过原定 f64 状态/轨迹误差门，不能据此声称全面 f64 兼容。
Brian2GeNN 旧工具链需要显式记录的 `exprel` 与整数 `exp` 兼容绑定。
原始 NPZ、源文件清单、编译器信息、失败记录及分范围计时已归档；统一计时和
重复 warm 性能评估仍待完成。详见 [HH 比较报告](execution-plan-evidence/gpu-hh/README.md)。


### CUDA 常驻 DAG 与 Graph 重放

同一 `CudaExecutor` 现可保留不可变输入和设备分配，每次重放恢复所有可写初值。
上一阶段的 `dag_execution` 支持 `direct`、`resident`、`graph` 或默认 `auto`；
当时默认首次使用 resident，重复使用且不超过 65,536 次 kernel launch 时再捕获 Graph。
当前默认策略已由下节的单次 activation 小图复用扩展。
这保留原始绝对整数 tick、多 clock 合并顺序和 f32 算术，不改变 B2IR/plan 身份。
Graph 构建、上传和冷启动成本单独报告，不能只以热运行计时衡量单次 run。
Device 的 `cuda_dag_execution` 可显式指定策略；不同 `Network.run` 仍建立独立
executor，跨 activation 复用尚未实现。详见 [CUDA Graph 证据](execution-plan-evidence/cuda-graphs/README.md)。

最终同源测试：L4 88 项通过、35 项 Apple 测试跳过；A100 159 项通过、73 项
Apple 测试跳过。两规模 CUBA 的冷/预热/测量输出在两 GPU 上逐位一致。
Graph 热重放 wall time 相对直接提交提升约 L4 1.34–3.68 倍、A100 1.09–2.39 倍；
首次捕获仍更慢，不能据此声称单次 Device run 或跨后端性能已经解决。


### CUDA 单次 activation 内的小图复用

`chunked` 路径把现有 f64 主机调度转换为 int64 tick 表，捕获最多 64 种、每种
最多 64 条指令的小图，再由设备游标依序选择实际 tick。每段末尾增加一个游标
更新 kernel，完整保留原 DAG 顺序、多 clock 合并与绝对 tick；不要求时钟周期整除。
附加表、游标和错误位计入数据缓冲区预算；生成的 tick-table kernel 与游标 ABI
绑定到 CudaPlan 源码和缓存身份。

默认 `auto` 在捕获节点数不超过总逻辑 launch 数四分之一、表长不超过 1,000,000
且预算允许时选用该路径。否则沿用 resident/完整 Graph 策略；显式 `chunked`
超界会报错。这样可以在首次 activation 内摊薄捕获成本，但尚未实现跨
`Network.run` 的常驻复用，也不改变 f32 与 f64 的差异。首次运行和计划/模块准备
成本分别测量，见 [分块 Graph 证据](execution-plan-evidence/cuda-chunks/README.md)。
