# Gate 0：Rust Standalone Device PoC

## 用户级安装

从当前 checkout 安装可供所有本地实验调用的开发环境：

```bash
./tools/install_user.sh
```

安装器在 `~/.local/bin` 创建 `brian2-atlas`、
`brian2-atlas-python` 和 `b2-runner`，依赖隔离在
`~/.local/share/brian2-atlas`，并在 `install.json` 记录 Brian2、Git、
Rust 1.98.1、Cargo 和 runner 的精确身份。Python 端继续链接当前 checkout，
因此前端修改立即可用；Rust 源码修改后须重新运行安装器，以更新固定的 runner
快照。

这是 [V4 方案](../BRIAN2_RUST_ARCHITECTURE_V4.md) 的第一个执行探针：

```text
import brian2_rust → set_device('rust_standalone')
  → Brian2 NeuronGroup / Synapses / Monitor → run() 或 Network.run()
  → Brian2 单位检查、变量解析、state-updater abstract code、Statement
  → CodeObjectSpec（标量/神经元/突触语句、类型、调度、读写集合）
  → 稳定 B2IR v1 JSON
  → reference：独立 Rust runner，预解析指令 + 分块 SoA 执行
    或 AOT：Rust 校验 → ExecutionPlan → 模型专用 Rust 源码 + 紧凑实例 → rustc
    或 Metal：显式 float32 → 时间融合 kernel / 整数延迟突触 target-owned DAG
    或 CUDA：显式 float32 → CudaPlan → nvcc/CuPy → 同一 Device 生命周期
    或 MPI：reference-f64 → DistributedPlan → canonical Rust AOT + MPI spike 交换
  → versioned little-endian results.bin → NumPy memmap 视图 → 回填 Brian2 state / Monitor / clock
```

它验证 Python 建模、Device 集成、语义导出、独立执行及结果读取的最小闭环。`RustStandaloneDevice` **直接继承 Brian2 的 `Device`**，导入可选模块 `brian2_rust` 后注册到 `all_devices`，通过正式的 `network_run` 扩展点执行。不改 Brian2 主执行循环、不继承 C++/Runtime Device、不执行其他后端的 CodeObject。reference 引擎将通用算术表达式编译为整数索引指令；AOT 引擎生成逐模型原生 Rust 循环。LIF/CUBA 方程没有硬编码进任一执行器。当前仍是受限的 Gate 0 PoC，不代表整个 Gate 0 已完成。

MPI 现支持按 rank 混用 CPU 与 GPU：
`rank_backends=["cpu", "metal"]` 或 `["cpu", "cuda:0"]`，显式设置
`numeric_mode="mixed-f32"`。GPU 卸载神经元状态更新；事件、突触和通信仍
使用 CPU MPI 调度。使用方法及已验证边界见 [MPI_GPU.md](MPI_GPU.md)。

CPU MPI 已提供 `engine="mpi", ranks=4`：静态显式、binary CSR 或 fixed-total 网络、
单共享 clock、非负整数 tick 延迟。可变状态、外部输入与入边按 rank 存储，
保留全局 RNG 身份与事件顺序；只读突触前状态仍复制。
显式/binary topology 还支持 target-owned 可变 f64 突触状态、clock-driven
积分、延迟 `on_pre` 和 post `summed`；Skaar 2025 explicit NMDA 的 640 神经元
固定输入诊断已在 1/2/4/8 ranks 逐字节通过，论文首个 2,560 神经元规模也完成
逐字节一致的单次 1/2/4-rank 探针。mutable fixed-total、pre summed、
多 clock 和分段续跑仍会在规划阶段拒绝。
23/24 Linux 完整 FlyWire（139,255 神经元、15,091,983 加权边）的四条件、
1/2/4 ranks 均与独立 reference 逐字节一致。本轮 33 次运行全部通过；
相同工具链下，优化前后静息仿真中位数改善 2.86–3.26 倍，4 ranks 每进程峰值
从 400.5 降到 76.5 MiB。4 ranks 相比优化后的 1 rank 仅快约 4%，且波动约 7.5%，
尚未证明稳定的强扩展收益。支持范围见 [MPI.md](MPI.md)，
[分片优化实测](mpi-evidence/rank-local/README.md) 与
[多脑区模型评估](MULTI_AREA_MPI.md) 记录证据及下一阶段限制。fixed-total 的分布式生成、权重/逐边延迟初始化与
资源受限试跑见 [MPI_PROCEDURAL.md](MPI_PROCEDURAL.md)。

完整 Potjans–Diesmann 2014 微回路的原生可行性里程碑、复现命令和 Apple
Silicon 实测见 [PD14.md](PD14.md)。当前 `pd14` CLI 已能在 Rust 内构建并运行
77,169 神经元、298,880,968 条突触的完整网络。普通 Brian2 `Synapses`
也可通过 `brian2_rust.connect_fixed_total(...)` 导出 procedural B2IR，并由
reference/AOT 在 Rust 中确定性构建 source-major 拓扑。B2IR v20 进一步支持有单位的
逐边 `(constant)` clipped-normal 参数和 heterogeneous delay initializer；完整的八组、
55 投影 PD14 因此已经通过普通 Brian2 `Network`、统一 scheduler、binary dump 和
monitor 回填路径运行。`examples/pd14_device.py` 同时生成固定总数 C++ standalone
基线，`examples/nest_pd14_benchmark.py` 运行语义对齐的 NEST 3.10 基线，
`examples/pd14_compare.py` 对照官方十个 PyNEST realization 计算三后端 KS 距离；
完整同机结果见 [PD14_COMPARISON.md](PD14_COMPARISON.md) 和
[PD14_LINUX_COMPARISON.md](PD14_LINUX_COMPARISON.md)。与 NEST 的 Apple Silicon、
等线程及 96-core NUMA 对比见
[PD14_NEST_COMPARISON.md](PD14_NEST_COMPARISON.md)。16 GB M3 MacBook Air
上的内存可行性复测见 [PD14_LOCAL_16GB.md](PD14_LOCAL_16GB.md)：Rust 完成
完整 10 秒仿真，而 C++ standalone 在进入编译前即超出物理内存预算。

浏览器执行器现可从同一逻辑 ExecutionPlan 构建 WASM 计划，在 Worker 中分批运行，并导出原生格式结果。构建、API 与浏览器验证见 [WASM.md](WASM.md)。

## 运行

在仓库根目录执行，使用现有的 `.venv`（需要 Brian2 和 NumPy）及 Rust/Cargo：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/minimal.py
```

`run()` 自动调用 Cargo 构建 runner，再执行模型并回填结果；已有构建会增量复用。最小示例只选择 Device、建模、运行和读取 Brian 的公开结果 API。需要与 NumPy 自动差分对比时执行：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/lif.py
```

这一轮新增 **100 个神经元、两个耦合状态变量** 的验收例子，使用不同初值和逐神经元驱动参数，分别通过 Rust Device、NumPy 和 C++ standalone 执行：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/population.py
```

默认进行三方差分，需要可用的 C++ 编译器和 make。仅运行 Rust 时：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/population.py --backend rust --output brian2-rust/output/my-population
```

仓库中的两个官方 standalone 多进程示例也可以直接切换到 Rust Device；
Device 模块会在每个 `multiprocessing` 或 joblib/loky worker 内单独注册：

```sh
PYTHONPATH=brian2-rust/python:. \
BRIAN2_STANDALONE_DEVICE=rust_standalone \
BRIAN2_STANDALONE_MODULE=brian2_rust \
B2_RUNNER=brian2-rust/target/release/b2-runner \
MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib \
.venv/bin/python examples/multiprocessing/03_standalone_joblib.py
```

安装官方示例用到的可选 Python 包可执行
`uv sync --extra examples`（或 `pip install -e '.[examples]'`）。OpenCV
示例在 C++ standalone 下仍使用有状态原生函数；Rust Device 下会改用周期
Python callback 和 `cv2` 解码，因此只需 examples extra。也可以用
`BRIAN2_OPENCV_VIDEO=/path/to/video.avi` 指定本地视频，跳过下载；或设置
`BRIAN2_OPENCV_CAMERA=0` 使用摄像头。摄像头默认处理 120 帧，可用
`BRIAN2_OPENCV_FRAMES` 调整，并建议用 `BRIAN2_OPENCV_SIZE=64x36` 控制模型
规模。结果窗口支持空格/点击暂停与恢复、`r` 重播、`q`/Escape 关闭；无人值守
运行可设置 `BRIAN2_OPENCV_GUI=0`。

`population.py` 的模型是：

```python
import numpy as np

group = NeuronGroup(
    100,
    '''dv/dt = (drive - v - w) / tau : 1
       dw/dt = (0.1*v - w) / tau_w : 1
       drive : 1 (constant)''',
    threshold='v > 1', reset='w += 0.02; v = 0', method='euler',
    dt=0.1*ms, namespace={'tau': 10*ms, 'tau_w': 5*ms},
)
group.v = np.linspace(0, 0.8, 100)
group.w = np.linspace(0, 0.1, 100)
group.drive = np.linspace(1.1, 1.8, 100)
state = StateMonitor(group, ['v', 'w'], record=True)
spikes = SpikeMonitor(group)
Network(group, state, spikes).run(100*ms)
# state.v.shape == state.w.shape == (100, 1000)
```

新增静态 `Synapses`。最小例子是 `0 → 1 → 2` 的三神经元传播链：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/synapses.py
```

该例使用普通 Brian2 API：

```python
synapses = Synapses(group, group, 'w : 1', on_pre='v_post += w',
                    delay=1*ms, clock=group.clock)
synapses.connect(i=[0, 1], j=[1, 2])
synapses.w = 1.2
```

初始只有神经元 0 超过阈值，dt=1 ms，运行 5 ms 后得到 spike index `[0, 1, 2]`、spike 时间 `[0, 2, 4] ms`。到期的突触事件在当前 tick 的阈值检查之后投递，因此接收神经元最早在下一个 tick 放电。`connect` 通过正式的 `synapses_connect` 扩展点创建 device 所有的数组，不调用 Runtime/C++ CodeObject。

双 population 跨组投递的最小 AOT 示例同时使用不同组大小、逐神经元参数、独立 monitor、异质 delay 和 clock-driven 突触状态：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/two_populations.py
```

双组路径中的每个 population 都拥有独立的状态/参数 schema、`CodeObjectSpec`、monitor schema 和 `dt`；`NeuronGroup` 还可独立配置 threshold/reset 和 fixed refractory，constant-rate `PoissonGroup` 可作为无状态放电源。跨组 `Synapses(source, target, ...)` 显式引用 source/target population，并在各自的局部状态 schema 与局部 neuron index 上解析 pre/post alias。reference 与 AOT 都按各 population 的 clock 激活 phase，最后分别回填各组公开数组。

两个独立 population（80 个 `Exc`、20 个 `Inh`）、四个投影、800 条兴奋/抑制静态连接和 0.3 ms 固定延迟的 CUBA 验收：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/cuba.py
```

默认在独立进程中对比 Rust AOT、Rust reference、NumPy 和 C++ standalone；也支持 `--backend aot|reference|numpy|cpp --output <新目录>`。模型使用 `dv/dt=(drive-v+I_syn)/tau`、`dI_syn/dt=-I_syn/tau_syn` 和 `on_pre='I_syn_post += w'`，原单组拓扑确定性拆为 `Exc→Exc`、`Exc→Inh`、`Inh→Exc`、`Inh→Inh`，权重与连接均显式给定，不涉及 RNG。

固定不应期和 `(unless refractory)` 已支持。最小例子与带 2 ms 不应期的 CUBA 三方验收：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/refractory.py
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/cuba.py --refractory-ms 2
```

使用 `NeuronGroup(..., refractory=2*ms)` 开启固定不应期；在需要冻结的 ODE 上标记 `(unless refractory)`。CUBA 只冻结 `v`，`I_syn` 在不应期内仍持续衰减、接收突触输入。`refractory.py` 的 3 ms 不应期例子在 `[1, 5, 9] ms` 放电，冻结电压期间第二个状态 `x` 继续积分。

第一次构建需要下载 Cargo 依赖。Device 尊重已有 `CARGO_HOME`，未设置时使用仓库内 `.cache/rust-cargo`。可通过 `set_device(..., runner='/绝对路径/b2-runner')` 使用已构建的二进制并跳过编译。无需 PyO3、MPI 或额外 Python 包。`rust-toolchain.toml` 将本项目的 Rust 构建固定为 **1.98.1**；在 `brian2-rust` 目录内运行 `rustc -vV` 和 `cargo -V` 应先显示该版本。从外部 Brian2 工作目录启动的 Device runner、AOT 和 MPI 编译也会检查实际 Rust 1.98.1 原生目标；旧版本或非原生编译器会直接报错。若 1.98.1 仅安装在隔离路径，使用已有 runner 并设置 `B2_RUSTC=/绝对路径/rustc` 可指定 Device AOT 编译器；版本与架构门禁仍然生效。跨主机正式性能基准必须记录实际编译器身份；以前其他版本的诊断数据不进入当前基准结论。

自己的脚本需要先将 `brian2-rust/python` 加到 `PYTHONPATH`（附带示例已自动处理）。模型写法：

```python
from brian2 import *
import brian2_rust  # 注册可选 Device

set_device('rust_standalone', engine='aot')  # 或 engine='reference'（默认）
group = NeuronGroup(
    1, 'dv/dt = (drive - v) / tau : 1',
    threshold='v > 1', reset='v = 0',
    method='euler', dt=0.1 * ms,
    namespace={'drive': 1.5, 'tau': 10 * ms},
)
group.v = 0
state = StateMonitor(group, 'v', record=True)
spikes = SpikeMonitor(group)
network = Network(group, state, spikes)
report = brian2_rust.capability_report(network, 100 * ms)
assert report.supported, report.format_text()  # 只检查/降级，不构建或运行
network.run(100 * ms)  # 也可以直接使用 run(100 * ms)
print(state.v.shape)  # (1, 1000)，Brian 公共 API 的 neuron × time 布局
print(spikes.t / ms)
print(group.v[:])
```

`engine='aot'` 先调用同一个 Rust validator 检查 B2IR，再生成 `native/main.rs`、紧凑的 `native/instance.bin` 和 `native/manifest.json`，最后用 `rustc -C opt-level=3 -C codegen-units=1` 编译模型专用可执行文件。编译失败会直接终止，不会回退到 reference、NumPy 或 C++。manifest 保存源文件/实例 SHA-256、rustc 版本、参数和分阶段构建计时。

`lif.py` 先通过 Rust Device 运行，再切回 runtime 创建独立的 Brian2 NumPy reference，比较每一步状态、采样时间、spike tick/index/time/count、最终状态及网络时间。每次执行会生成独立的 `brian2-rust/output/device-<suffix>/`，末尾打印实际路径。也可以用 `set_device(..., directory='<新目录>')` 或差分示例的 `--output <新目录>` 指定目录；已有目录会被拒绝，避免覆盖结果。

Device 拥有自己的初始数组和结果数组；读结果不需要手动解析 CSV。切回其他 Device 后，已有的 Rust 结果仍通过原所属 Device 读取。`build_on_run=True` 支持同一 Network 连续多次 `run()`：绝对时间、神经元/突触/refractory 状态、State/SpikeMonitor 历史及跨分段 delay 事件都会延续。`Network.store()/restore()` 在该模式下同时保存/恢复 counter RNG 和原生 pending delay queue，支持内存及 pickle 文件 checkpoint；显式排队 build 暂不支持 checkpoint。每次用户发起的 run 默认产生独立 artifact；显式 `directory` 的后续 run 保存在 `run-0002` 等子目录。对重复 restore/replay 工作负载可显式设置 `retain_run_artifacts=False`，设备会在后续 run 成功且结果已载入后删除被取代的 artifact，从而只保留最新的成功 run。`NetworkOperation` 内部 continuation 将绝对 clock tick/final time 作为 instance 输入，相同长度且不含外部 C ABI Function 的 AOT 分段可在当前 Device 生命周期内复用已编译二进制；同时只保留最后一个成功分段，其已被取代的中间 artifact 会及时删除，避免磁盘占用随 callback 次数线性增长。`build_on_run=False` 会将连续的 run 排队为一条时间轴，随后由 `device.build()` 一次生成并执行；目前明确拒绝在这些尚未执行的排队 run 之间修改模型。`reinit(); activate()` 开始新的模型所有权周期，旧模型或不同 Network 会被拒绝。建模阶段的字符串初始化使用一次性 NumPy frontend CodeObject，可使用 `i/j`、单位、`rand()` 和 `randn()`；它只写 Device 自有初值数组，不参与模拟。自动执行支持 `report='text'/'stdout'/'stderr'` 或 Python callback 的开始/结束边界通知；独立子进程运行中暂不提供周期回调。`profile=True` 和 Device 默认 profile 会向 `Network.profiling_info`/`profiling_summary` 发布 simulation+recording 与二进制 dump 两项粗粒度计时。AOT 性能诊断可额外设置 `B2_AOT_PROFILE_PHASES=1`，在 `summary.json.phase_profile` 中记录 scheduler、groups/monitor、neuron/synapse state、threshold、主/附加 event pathway、PoissonInput、post 和 reset 的累计秒数；主 event 计时还用嵌套的 `primary_enqueue`/`primary_apply` 分解 delay queue/owner 分类与事件 kernel。默认关闭，正式性能基线不承担 `Instant` 采样开销。显式排队 build 的 report/profile 和 checkpoint 暂不支持。

本机 2026-09-03 实测：

- 100 ms，dt=0.1 ms，共 1,000 个采样点。
- 9 次 spike，tick 为 `109, 219, 329, 439, 549, 659, 769, 879, 989`。
- 所有 spike tick 完全一致；状态最大绝对误差 `1.3877787807814457e-17`。
- 最终状态 `v = 0.14342688748679328`。
- 状态比较容差 `rtol=1e-12, atol=1e-14`，时间比较 `atol=1e-15 s`；不承诺跨后端 bitwise 相等。

100 神经元 / 两变量验收结果：

| 对照后端 | 最大绝对 v 误差 | 最大绝对 w 误差 | Spike 数及顺序 |
| --- | --- | --- | --- |
| NumPy | `2.220446049250313e-16` | `2.7755575615628914e-17` | 738，tick/index/count 完全一致 |
| C++ standalone | `2.220446049250313e-16` | `2.7755575615628914e-17` | 738，tick/index/count 完全一致 |

三方各自使用独立进程和同一模型定义。完整轨迹、最终状态、Monitor 时间、spike 时间及网络/clock 时间均通过上述容差。C++ 对照使用单线程，Unix 编译参数为 `-O2 -std=c++17 -fno-fast-math -ffp-contract=off`；这是正确性验证，不是性能 benchmark。每次验收生成 `output/population-<suffix>/comparison.json`、各后端的 `results.npz`、环境元数据和日志。

双 population、四投影 CUBA 验收结果（80 Exc + 20 Inh，800 synapses，1000 ticks，delay=3 ticks）：

| 对照后端 | 最大绝对 v 误差 | 最大绝对 I_syn 误差 | Spike / 突触投递 |
| --- | --- | --- | --- |
| Rust reference | `0` | `0` | 702 / 5592 |
| NumPy | `0` | `0` | 702 / 5592 |
| C++ standalone | `0` | `0` | 702 / 5592 |

以 Rust AOT 为基准，四个后端的 spike tick/index/count/顺序、四投影重组后的 i/j/权重、完整轨迹和最终状态完全一致。验收产物保存在 `output/cuba-<suffix>/`。这是显式固定拓扑的双 population CUBA 验收模型，尚不覆盖依赖 RNG 的完整 Brian2 CUBA benchmark。

同一 CUBA 模型加入 `refractory=2*ms`、电压 `(unless refractory)` 后：

| 对照后端 | 最大绝对 v 误差 | 最大绝对 I_syn 误差 | Spike / 突触投递 |
| --- | --- | --- | --- |
| Rust reference | `0` | `0` | 620 / 4944 |
| NumPy | `0` | `0` | 620 / 4944 |
| C++ standalone | `0` | `0` | 620 / 4944 |

除完整轨迹、spike 和时间外，还比较最终 `lastspike` 与 `not_refractory` 数组；均通过。Device 在校验这些数组和 spike 不应期间隔后才发布结果，`group.lastspike[:]`、`group.not_refractory[:]` 可正常读取。

## 不使用 Python 独立重放

reference 重放只需要通用 runner 和 `model.json`；AOT 重放只需要生成的模型二进制和 `instance.bin`。两种输出目录都必须不存在：

```sh
brian2-rust/target/release/b2-runner <输出目录>/model.json <新的重放目录>
<输出目录>/native/b2-native <输出目录>/native/instance.bin <新的 AOT 重放目录>
```

输出：

| 文件 | 含义 |
| --- | --- |
| `rust/results.bin` | `b2-result-dump-v3` little-endian 类型化二进制 Dump；保存所有 population 的轨迹、spike、最终状态、不应期状态，以及每个 Synapses 的最终状态和投递数 |
| `rust/summary.json` | 固定大小的运行元数据：Dump 字节数、模型规模、总 spike/突触投递数和模拟/Dump 写出计时；不再承载数组 |
| `comparison.json` | Python 示例额外生成的差分结果，不是 runner 的依赖 |
| `native/main.rs` | 从已校验 B2IR 生成的逐模型 Rust 源码 |
| `native/instance.bin` | little-endian 紧凑实例：dt、形状、SoA 状态/参数、refractory 和突触拓扑 |
| `native/manifest.json` | 输入/源码哈希、模型形状、rustc/编译参数和校验/生成/编译计时 |
| `native/b2-native` | 不依赖 Python 和 Cargo 的模型专用重放程序 |

`b2-result-dump-v3` 的整数与 IEEE 浮点均为 little-endian；名称、dtype 和单位继续由已校验 B2IR 提供，Dump 只保存定长数值布局。每个监控变量及最终状态保持其 f32/f64 原生宽度，因此 NumPy 可直接建立 memmap 视图：

```text
B2DMP001 | u32 version | u32 endian marker | u64 population_count
          | u64 neuron_count | u64 file_bytes
repeat population_count:
  u64 count, steps, record_count, variable_count, state_count,
      spike_count, last_spike_count, flags
  repeat variable_count: dtype samples[steps, record_count]
  i64 spikes[spike_count, 2]                 # tick, local neuron index
  i64 spike_counts[count]
  i64 last_spikes[last_spike_count]
  repeat state_count: dtype final_state[count]
  if flags & 1: f64 lastspike[count] | u8 not_refractory[count]
u64 synapses_count
repeat synapses_count:
  u64 synaptic_state_count | u64 edge_count
  f64 final_synaptic_states[synaptic_state_count, edge_count]
  u64 delivered_events
f64 final_time_seconds | B2END001
```

模拟热循环只记录内存：状态样本保存为原始 f64，spike 保存为 tick/index pair，不做浮点转文本、文件写入或 flush。所有 tick 成功后，runner 顺序写出一个 `results.bin`。Dump 头包含 magic、协议版本、endianness marker、总字节数及 population 数量/形状，文件尾包含结束标记；Python 还会依据 B2IR 校验数组长度、有限性、spike 顺序/count、不应期状态和小型 metadata，然后通过 `np.memmap`/NumPy view 直接读取 payload，避免 `loadtxt`、逐行 CSV 解析和大型 JSON 数组物化。`timings.dump_write_seconds` 单独统计二进制写出成本。

默认模式的状态样本缓冲在模拟前预留，spike 缓冲按需增长且不超过每神经元每 tick 一次的事件预算。现有预算分别限制最多 10,000,000 个状态值和 spike。长时程任务可在激活 Device 时设置 `recording_window_steps=N`：runner 只保留每个 population 最后 N 个本地 clock tick，Dump 只写该窗口，分段 run 回填后继续裁剪 StateMonitor/SpikeMonitor。该模式让原生和 Python 记录内存不再随总生物时间增长，但属于显式 rolling-window 扩展，`count/num_spikes` 也只描述当前窗口；不设置时仍保持 Brian2 的完整历史语义。

需要保留完整历史时，可在 reference/AOT 自动执行模式设置显式目录和 `monitor_streaming_steps=N`。Device 会按最小 active clock 的 N 个 tick 分段执行，把每个 State/Event/Spike/PopulationRateMonitor 写到 `<directory>/monitor-stream/chunk-*`，而 Brian Monitor 对象只保留最后一个 chunk。`open_monitor_stream(path)` 返回不预载数组的 reader，`iter_monitor_chunks(path, monitor_name)` 每次只读取一个有界 chunk；StateMonitor 数值的磁盘布局为 `(time, recorded_index)`。该模式当前不与 NetworkOperation、排队 build、GPU/MPI 或 profile 组合。

reference 对每个算术指令检查有限性；AOT 校验所有输入，并在输出前检查最终状态。失败会非零退出且不会写出结果文件。运行后 I/O 失败可能留下不完整的 Dump，但长度/结束标记校验会阻止 Python 向 Brian 发布部分结果；只有完整 Dump 成功写入和 flush 后才写小型 `summary.json`。

## Compatibility Matrix v0（仅此探针）

下面是实现边界的紧凑摘要。面向当前 Brian2 用户 API、结果对象和运行语义的完整逐项对照，以及每项的兼容优先级，见 [COMPATIBILITY.md](COMPATIBILITY.md)。

| 能力 | 当前边界 |
| --- | --- |
| 模型 | 任意数量 population：拥有独立 schema、方程、threshold/reset、dt 和 refractory 的 NeuronGroup，constant-rate PoissonGroup，或静态/周期 SpikeGeneratorGroup；合计 1..1,000,000 neurons，NeuronGroup 每组 1..32 个可变状态；ODE 为 f32/f64，参数状态可为 f32/f64/i32/i64/u32/u64/bool |
| 方程 | 显式 `method='euler'`、`'rk4'` 或 `'exponential_euler'`；执行 Brian2 展开的求解器语句；支持动态及 `(constant over dt)` subexpression、1D/2D `TimedArray`、浮点四则/幂/模/向下整除、全部 Brian 内置确定性数学函数、`int`/`timestep`、数值/布尔比较与逻辑运算，以及带静态单位的单表达式纯 user-defined `Function`；f32 存储通过显式 `f32_to_f64`/`f64_to_f32` 边界定义逐次写回舍入；多个方程保留每个 stage 所需的旧状态快照和提交顺序，允许有证明条件的临时量融合 |
| 参数 | 普通 `x : unit` 可变逐神经元参数；namespace 标量常量；方程中 `(constant)` 逐神经元参数和 `(constant, shared)` 标量参数；另支持由唯一 `run_regularly` 无条件标量赋值的可变 `(shared)` 参数（含 scalar `rand()`，reference/AOT 同 seed 一致）；均以 SI 值导出 |
| 索引与整数 | 神经元 `i/N` 与突触 `i/j/N/N_pre/N_post` 为 `index`，`timestep()` 与受界偏移为 `tick`；subgroup 的 i/j 保持局部坐标；进入浮点算式时显式 `index_to_f64`/`tick_to_f64`；公开 i32/i64/u32/u64 状态保留定宽位值（包括 >2^53），整数加减乘/整除/取模与 cast 具有明确语义 |
| linked variable | reference/AOT 支持跨 NeuronGroup 可变状态的 identity、固定数组/subgroup 和本地整数状态动态映射；reference 另支持 Synapses 从固定 source index 读取确定性 NeuronGroup subexpression（含 1→N 广播）及 StateMonitor 重建。链接读进入全局 effect DAG，动态越界明确失败；暂不支持 Synapses 动态映射/AOT/GPU、non-identity self-link 或 constant source |
| Device 生命周期 | 注册/选择、独立数组、连续多段自动执行、排队 run + 显式 build、结果追加回填、reinit；一个初始化周期绑定一个 Network |
| 时间 | 每个 population 独立 clock/dt；绝对时间跨 run 延续，duration 是各 dt 的整数倍；固定/异质 delay 可跨分段边界 |
| 阈值/reset | threshold 可选，reset 与 SpikeMonitor 均可独立省略；`== != > >= < <=`、`and/or/not` 或布尔常量；多语句 reset 和 Brian 模型代码支持的 `+= -= *= /= %=`，按源顺序执行 |
| refractory | 固定标量时间，非负且有限，默认 Brian2 timestep 语义；可选 `(unless refractory)`，每个变量独立冻结，支持公开 lastspike/not_refractory 结果 |
| 突触拓扑 | 任意数量独立 Synapses（对象数受 IR/数组预算约束），可自连接、跨任意已导出 population，或使用其连续 subgroup 作为 endpoint；reference 另支持单层 Synapses edge domain 的 population↔edge 及双侧 edge→edge post-summed reduction；显式数字局部 i/j、`connect()`、标量 `p=0..1`、受限字符串 p/condition，以及 postsynaptic `j` 的 `range`/`sample` generator（含条件、`rand()`、`skip_if_invalid`）；允许追加连接、非负整数 n 和显式重复 pair，使用 source clock；默认每个 Synapses 最多生成 50,000,000 条边，最多扫描 500,000,000 个候选 pair |
| 随机性与输入 | `seed(integer)` 可复现概率拓扑、建模阶段字符串初始化及运行期 `rand()/randn()/poisson()`/`BinomialFunction`；运行期采用 `splitmix64-counter-v1`，按 seed/draw-site/绝对 tick/逻辑 index 定位，reference/AOT 及连续/分段 run 逐位一致；支持表达式/TimedArray `PoissonGroup.rates`、constant `N/rate` 的 `PoissonInput` 与 `SpikeGeneratorGroup`；与 Brian2 NumPy/C++ 不承诺相同随机位流 |
| 突触参数 | 无 flag 的逐边 `w : 1` 是可变状态；`(constant)`、`(shared)`、`(constant, shared)` 及 namespace 标量为运行期只读参数；支持 f32/f64 |
| 突触状态 | 最多 32 个逐突触 f32/f64 状态；支持普通可变变量、显式 `(clock-driven)` ODE，以及 Brian 展开的独立一维 `(event-driven)` ODE/`lastupdate`；clock-driven 支持 `method='euler'`、`'rk4'` 或 `'exponential_euler'` |
| on_pre/on_post | 多个命名 pre pathway（order=-1）及 post pathway（order=1），按默认 Brian schedule/name 顺序执行；每条 pathway 有独立标量或逐突触 delay queue；pre 可读写逐突触状态并写 postsynaptic ODE state，post 当前只写逐突触状态；二者可读 pre/post 状态、参数、i/j/N/N_pre/N_post/t/dt |
| delay | 每条边可设置独立的有限非负延迟；按 `floor(delay/dt + 0.5)` 量化，零延迟在放电当期投递；统一 delay 以长度 1 紧凑保存 |
| 观测 | 每个 NeuronGroup 可有多个 StateMonitor；physical plan 合并变量/索引后一次采样，再按各 monitor 的顺序与重复索引回填；支持 linked 状态。reference 还支持以 Synapses 为 source，记录逐突触可变状态、按边映射的 pre/post neuron state（含 summed destination alias），或重建受限 fixed-index linked subexpression，并可使用独立 monitor clock；该路径在 AOT/GPU 明确失败。每个 spiking population 一个默认 SpikeMonitor；无状态 PoissonGroup/SpikeGeneratorGroup 不需要 StateMonitor |
| 数组预算 | 默认 B2IR ≤2 GiB；初值、参数、refractory 状态、拓扑 i/j 和 delay 合计 ≤100,000,000 个元素；显式突触默认每投射 ≤50,000,000 条 |
| 运行预算 | 每个 population ≤10,000,000 ticks；所有投射合计 ≤50,000,000,000,000 synapse-ticks；每 population 的物理 StateMonitor 样本组合 ≤100,000,000；spike 记录 ≤10,000,000；delay/refractory 各 ≤1,000,000 ticks |
| 代码预算 | 神经元/突触各 ≤128 parameters，每 CodeObjectSpec ≤128 statements；每个执行计划的临时寄存器缓冲 ≤4,000,000 个 f64（32 MB） |
| 明确拒绝 | 嵌套 edge-domain endpoint、读取旧值/条件或多 runner 写入的可变 shared、linked non-identity self/constant source、Synapses 动态 linked 映射或其 AOT/GPU 执行、on_post 写 pre/post 神经元状态、耦合/非独立 event-driven ODE、动态 PoissonInput `N/rate`、presynaptic `i` generator、受限子集以外的 connect 表达式、legacy refractory timing、未显式提供 `b2ir-c-abi-v1` 的捕获 Python 状态/控制流/随机调用 Function（native-only 仅 AOT）、其他模型 dtype、AOT 非默认 event 与 run_regularly slot、Effect Algebra 无法证明等价的 custom schedule、callback |

大规模显式 `Synapses.connect()` 的**进程容量**另行受限：默认每个投射最多
50,000,000 条边、每次条件连接最多 500,000,000 个候选对、
全模型最多 100,000,000 个初始值、B2IR 文件最多 2 GiB。
经主机内存与磁盘预检后，可分别设置
`B2_MAX_EXPLICIT_SYNAPSES`（上限 1,000,000,000）、
`B2_MAX_CANDIDATE_PAIRS`（上限 1,000,000,000）、
`B2_MAX_INITIAL_VALUES`（上限 8,000,000,000）和
`B2_MAX_IR_BYTES`（上限 64 GiB）。Python 前端和 Rust 独立校验器
对显式边数、初始值数、IR 字节数使用相同的有效预算；这些配置只改变
允许构建的规模，不改变方程、连接、逐边状态、延迟、求解器或精度。
二维或一维 `TimedArray` 默认每张表最多 10,000,000 个有限 f64 值；
经主机容量预检可设置 `B2_MAX_TIMED_ARRAY_VALUES`（每表上限
1,000,000,000），Python 前端和 Rust 独立校验器同样执行该限制，
整个 B2IR 仍受 `B2_MAX_IR_BYTES` 限制。

确定性函数白名单为 `exp/log/log10/expm1/exprel/log1p/sqrt`、`sin/cos/tan`、`sinh/cosh/tanh`、`arcsin/arccos/arctan`、`abs/sign/ceil/floor/clip`。Brian state updater 展开的 dimensionful `clip(x, 0, inf*unit)` 会把 cast 后的零继续作为多态量纲下界，并将内置 `inf` 饱和为 finite-only B2IR 可表示的最大 f64。输入、每个中间结果和最终状态仍必须是有限 f64；例如负数 `sqrt`、非正数 `log`、零除法或溢出的 `exp` 会让 standalone 运行明确失败。

### COBA/HH 表达验收

当前表达层已经覆盖经典 COBA/HH 方程需要的 conductance/voltage 单位、状态乘积与整数幂、`exp`/`exprel`、alpha/beta 确定性 subexpression、`exponential_euler`，以及只有 threshold＋fixed refractory、没有 reset 或 SpikeMonitor 的 HH population。`tests/test_coba_hh.py` 分别对 conductance-based 突触状态更新和经典六状态 COBAHH 方程做 AOT/reference/NumPy 全轨迹、最终状态与 refractory 差分：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m unittest discover -s brian2-rust/tests -p 'test_coba_hh.py' -v
```

这表示 COBA/HH 的方程表达与求解器闭环已经接通；subgroup、多个 Synapses、`connect(p=...)`、`seed(...)` 和 `rand()/randn()` 初始值也已具备。新增的 seeded COBA 风格测试使用 excitatory/inhibitory subgroup、两条概率投影和随机初态，并验证 AOT/reference 得到完全相同的拓扑、初值和轨迹。运行预算保持有界并提升到 50 亿 synapse-ticks，足以覆盖仓库原始 `examples/COBAHH.py` 的 1 秒 workload；`report='text'` 已支持开始/结束边界通知，但尚无独立子进程运行中的周期进度。

`cobahh_throughput.py` 将原始 4,000-neuron、80/20 subgroup、`p=0.02`、六状态 COBAHH workload 固化为 AOT/C++ 性能硬门。脚本使用显式 seeded 初态和拓扑保证两端输入一致，只记录 3 条电压轨迹，预热后交错 replay；最终六个状态、轨迹、spike 和连接数必须一致，且 AOT 热循环中位数不得慢于单线程 C++：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/cobahh_throughput.py --output /tmp/cobahh-throughput
```

本机 2026-09-04 完整性能套件中的 1 秒、10,000 ticks、5 次中位数为 AOT 2,699.594 ms、C++ 3,317.780 ms，AOT 快 1.23×；319,232 条连接产生 142,039 个 spike，结果一致。优化前的 100 ms AOT 为 456.561 ms；RHS 局部公共子表达式消除和整数幂 `powi` 专化后为约 271 ms，降低约 40%，同长度 C++ 为约 331 ms。

`poisson_throughput.py` 固化 constant-rate `PoissonGroup` 的 counter RNG 性能门。默认 100,000 neurons、100 Hz、dt=0.1 ms、100 ms，共执行 100,000,000 次随机 draw，并记录约一百万个 spike；两端检查总数落在声明 Bernoulli 分布的六个标准差内，而不错误要求不同 RNG 算法产生相同位流：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/poisson_throughput.py --output /tmp/poisson-throughput
```

本机 2026-09-04 五次热循环中位数为 AOT 89.885 ms、单线程 C++ 421.050 ms，AOT 快 4.68×；AOT/C++ 分别产生 1,000,325/999,173 个 spike，每神经元 count variance 分别为 9.934/9.894，均符合期望均值 10 的统计检查。

`poisson_input_throughput.py` 单独约束 PoissonInput 的中心二项/normal 路径。默认 100,000 neurons、每神经元 100 个 1,000 Hz 输入、dt=0.1 ms、100 ms，共执行 100,000,000 次输入采样；两端只比较声明分布及六标准差范围，不比较随机位流：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/poisson_input_throughput.py --output /tmp/poisson-input-throughput
```

本机 2026-09-04 五次热循环中位数为 AOT 909.456 ms、单线程 C++ 1,272.020 ms，AOT 快 1.40×。初版逐元素三角 Box–Muller 为 1,337.998 ms、比 C++ 慢 6.3%；改为按逻辑索引配对的 counter Marsaglia polar sampler 后降低约 32%，同时保持 reference/AOT 及连续/分段逐位一致。

`stdp_throughput.py` 固化双 population 的 event-driven pair-STDP 性能门：默认 4,000+4,000 neurons、319,232 条显式同拓扑连接、20/17 Hz 错开放电、0.3 ms pre delay 和 1 秒模拟。它比较最终权重、Apre/Apost、lastupdate、稀疏轨迹及全部 spike，并要求 AOT 热循环不慢于单线程 C++：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/stdp_throughput.py --output /tmp/stdp-throughput
```

本机 2026-09-04 七次热循环中位数为 AOT 102.190 ms、C++ 110.009 ms，AOT 快 1.08×；两端共执行 11,809,718 次 pre/post 突触事件且结果一致。初版巨型 AOT 函数为 156.807 ms：target/source CSR edge ID 改为验证后的 u32、跨独立赋值保留安全 CSE，并将只触及逐边状态的 pathway 提取成 no-alias 批内核后，消除了 `exp` 调用周围的指针寄存器溢出；加载期仍校验 topology、数组长度和 pending event 范围，热循环中的 unchecked access 只用于已证明有界的索引。

`on_pre` 允许用 `v_post` 或无后缀的 `v` 访问目标状态，同一代码块对同一状态只能选择一个别名。`v_pre` 等源状态在投递时读取；目前拒绝在同一个 on_pre 中读取某个 presynaptic 状态又写该状态的 postsynaptic 版本，避免跨边读写依赖造成后端顺序差异。无 flag 的逐边变量（典型 `w : 1`）可由 pre/post 修改，带 `constant/shared` flag 的参数和 presynaptic 神经元状态仍只读。多个命名 pre/post pathway 按默认 order 与对象名稳定排序，每条 pathway 独立量化 delay、维护队列并跨连续 `run()` 保存 pending event；`on_post` 使用 target CSR 按目标 spike/连接创建顺序执行，当前只允许写逐突触状态，尚未开放对神经元状态的写入。

固定 refractory 的执行约定：

- fixed refractory 在每个 tick 的状态更新之前用预计算截止 tick 更新可用性；duration/boolean expression refractory 则由同一已验证 CodeObject 在 reference/AOT 中求值。非负时间的 `timestep` 为截断 `(time + 1e-3*dt)/dt`；它与 delay 的最近整数取整不同。零或不足一个 tick 的不应期仍最多每 tick 放电一次。
- 阈值只接受当前 `not_refractory=True` 的神经元；放电时立即写 `lastspike=t`、`not_refractory=False`，随后进入 synapses 和 reset 阶段。
- `(unless refractory)` 同时约束状态更新和 on_pre 对该目标状态的赋值。条件为假时跳过整条赋值，后续语句读取未被改写的值；reset 明确覆盖该条件，仍然执行。投递但被条件屏蔽的事件仍计入投递数。
- 未标记的 ODE 在不应期内继续更新。`lastspike` 和 `not_refractory` 是独立的 f64/bool SoA 数组，在用户模拟代码中只读；初始 lastspike 可设置为有限且不晚于当前 run 起点的时间，timestep 必须在 Brian2 int64 范围内。当前 StateMonitor 仍只记录 ODE 状态。

Brian2 前端保留单位检查；Rust 校验 schema、数组形状/索引、全部公开 dtype 与 index/tick 表达式类型、SI 单位维度、临时变量作用域、标量/逐神经元访问范围、参数只读、局部和全局读写集合及 schedule。不支持的 Brian 功能在导出时抛错；非法 IR 在创建模拟输出前拒绝。v23 会从输入符号开始逐表达式重新推导七维 SI dimension，并与每条 assignment 的目标 dimension 核对；v22 引入 Effect Algebra，v27 将 threshold/reset/pathway/EventMonitor 绑定到命名 EventStream；v29 增加可移植 Function Contract（现含 f64/i64/bool 签名）；v30 分离 index/tick；v31–v34 完成 f32/bool/定宽整数状态与类型化结果；v35 增加 canonical layer hash 和显式 migration；v36 将 linked read 及其跨 population ownership 纳入 schema 和 DAG；v37 增加按 CPU/CUDA/Metal/WGSL backend 键控、内容寻址的 Function implementation 表，并落地 `b2ir-c-abi-v1` CPU adapter。

## B2IR 探针约定

当前稳定输入协议为 `schema = b2ir-v1`，规范见 [B2IR.md](B2IR.md)，冻结决策见 [ADR 0001](docs/adr/0001-freeze-b2ir-v1.md)。v20–v32 建立 procedural topology、滚动 monitor、全模型 schedule/effect DAG、Rust 单位复核、一等 Clock/EventStream、expression refractory、portable Function、index/tick、f32 storage 与可替换 PoissonInput 参数；v33/v34 增加公开 bool 与定宽整数状态；v35 增加 `b2ir-canonical-json-v1`、Definition/Instance/Run 独立 SHA-256 及显式 v34 migration；v36 增加跨 population linked-variable descriptor、索引策略与 effect ownership；v37 增加通用 backend Function implementation 表、各 backend ABI/source/entry-point hash，以及已可执行的 CPU C ABI AOT 静态链接。当前 reader 会先验证再显式迁移 v34/v35/v36/v37，其余旧 JSON 明确拒绝：

- `definition`：一等 `clocks[]`（名称与 dt）；状态/参数的 dtype、单位维度和 scalar/neuron/synapse 索引域；population 的名称、offset、长度、clock 引用及可选 monitor 分区；pre/post 状态别名；可选 refractory 冻结状态列表；带 clock 引用的 CodeObjectSpec；带 semantic version/ABI/参数与返回类型和单位/effects/portable body/implementation hash，以及按 CPU/CUDA/Metal/WGSL 键控的可选 backend ABI、entry point、source hash 的 `functions[]`；numeric profile 和 RNG algorithm。
- `instance`：neuron_count、u64 RNG seed、每个神经元和突触 ODE 的初始数组、参数数组；`explicit` topology 携带 source/target 数组，`fixed_total` topology 携带 edge_count/seed/initializers；materialized delay 长度 1 表示标量、长度 E 表示逐边值，procedural delay 则只携带 initializer；可选 refractory period/period_ticks、初始 lastspike/not_refractory 数组。
- `run`：本段绝对起始时间和 duration，以及每个一等 Clock 的 `start_tick/steps`；population 内的 dt/steps 在冻结迁移期间作为一致性冗余字段保留。
- f64 值/dt 使用 16 位、f32 值使用 8 位小写十六进制 IEEE-754 位模式，避免导出时十进制转换改变位值。
- 单位维度按 Brian 的七个 SI 基本维度顺序保存，数值已规范化到 SI。
- `definition.schedule` 保存 `base_slots`、canonical `before_/base/after` 展开 slots，以及严格按 `(slot, order, name, id)` 排序的节点。每个节点引用唯一 executable object，声明全局资源 reads/writes 及保守 RAW/WAR/WAW 依赖；Rust 从被引用对象重新推导 slots、effects 和依赖图，拒绝伪造、遗漏、重复与乱序节点。
- 每条 statement 显式保存目标 dtype 与七维 SI dimension；Rust 独立推导加减、乘除、幂、比较、clip、TimedArray、transcendental function 及 Function call 的结果维度，验证 assignment/threshold/rate 等表达式，单位验证只发生在加载期。portable Function body 同样由 Rust 独立验算，reference 可执行且 AOT 内联；用户显式注册的 `b2ir-c-abi-v1` 源码作为独立 C translation unit 编译并静态链接，Rust 调用点只使用 f64/i64/bool 的固定 C ABI，不执行 Python，也不依赖 Rust ABI。native-only Function 在 reference 中明确失败。
- JSON 使用 `b2ir-canonical-json-v1`（UTF-8、排序 key、紧凑分隔、禁止 NaN），Definition/Instance/Run 分层独立 SHA-256；reader 先验证旧版本 envelope 再执行显式 migration。`b2ir-v1` 字段、默认值、数值编码和执行语义已经冻结；不兼容变化必须使用新 schema 与显式迁移。
- reference 使用进程内串行分块执行计划；AOT 生成隔离 artifact 和 manifest。AOT 固定 phase 顺序与 canonical 节点顺序之间的每个逆序对都必须由 Effect Algebra 证明无 RAW/WAR/WAW 冲突，证明失败即拒绝，不以硬编码 phase 猜测语义。Definition/Instance/Run 的 canonical identity 与内容 schema 已冻结；native artifact Manifest 仍保留独立版本，不属于 B2IR v1 输入兼容承诺。
- 临时变量仅在单个 code object 内可见，标量局部值在各 neuron/synapse lane 共享读取，vector 临时变量每个 lane 独立。
- reference executor 逐 tick 直接遍历 v22 的全局节点顺序：`before_start` CodeObject → StateMonitor → `groups` CodeObject/summed variable → threshold/event source → SpikeMonitor → pre/PoissonInput/post pathway → reset。refractory 刷新与 tick 提交仍是所属状态/时钟的生命周期操作。AOT 当前由同一已验证 CodeObject 元数据生成等价固定 phase；下一步会让其以 DAG 合法性证明驱动融合。
- Python adapter 允许自定义基础 `Network.schedule`；reference executor 按 active Clock 与全局节点顺序执行。AOT 通过 ExecutionPlan 选择已证明等价的 compact/general 优化路径，或按 canonical 顺序生成 `slot-v1` Rust 代码，承接前端已导出的合法 `when/order`、独立 Clock、命名 EventStream/EventMonitor 和 delay continuation。显式 `numeric_mode="float32"` 的 Metal 支持独立 population 和受限的突触/summed DAG，现含固定/逐边整数延迟及 pending continuation；延迟 pathway 要求在源 threshold 之后。`event_delivery="sparse"` 支持零延迟源端投递和 source/delay 分组投递，并可融合相邻的完整源 population 更新与入队；默认仍为 scan。API、限制与验证见 [ExecutionPlan v0](EXECUTION_PLAN.md)。
- Metal/CUDA 的 `gpu_max_buffer_bytes` Device 选项控制一次执行允许的总工作缓冲字节数，默认 512 MiB，并要求显式正整数。该限制同时作用于直接执行、跨 activation buffer adoption、autotune 及 tuning-cache replay；提高它只放宽主机已确认的设备容量门，不改变模型、精度或执行计划语义。Skaar 2025 的 5,120 神经元外部验证需要 1.87 GB resident buffers，使用了记录在结果清单中的 8 GiB 上限。
- Rust 构建 Source CSR（offsets + 原始 edge index），保留每个 source 的连接创建顺序。统一 delay 的环形队列保存 source 事件，到达时展开出边；异质 delay 在放电时按 edge 的 delay 分别入队。到达顺序按源 spike 时间、source index 和连接创建顺序稳定保留；队列槽复用容量。零延迟事件在同一个 tick 投递；超出本段终点的事件保存为待投递事件，并在同一 Network 的后续 run 中继续执行。

`python/brian2_rust/spec.py` 定义的 CodeObjectSpec 统一用于 state_update / threshold / synapses / synapses_post / poisson_input / reset，保存名称、when/order、all_neurons/spiking_neurons/active_synapses 执行域、scalar/vector 语句、dtype、条件写入和 reads/writes。阈值代码产生用户条件 bool `_cond`，refractory 刷新、阈值抑制和放电记账是显式的执行器生命周期操作。Brian state updater 生成的 `int(not_refractory)` 降为受限的 bool→f64 算子；intrinsic index、timestep 和 tick offset 则保留逻辑类型直到显式转换；`exponential_euler` 生成的纯私有整数常系数可精确提升为 f64，但不开放一般整数运算。

Rust 从代码重新推导类型和访问集合，校验拓扑、别名、只读输入、delay、不应期及调度。语句条件必须与冻结状态列表和访问域完全匹配，reset 不得带冻结条件；非法 IR 在创建输出前拒绝。模型方程没有针对 LIF/CUBA 的硬编码。

Euler 的 `_v`/`_w`、RK4 的 `k1..k4` 以及 Exponential Euler 的 affine coefficient 等中间值保留 Brian2 生成的 stage 与快照语义；producer/consumer guard 相同、只使用一次且不会跨越相关状态写入时，AOT 前的 B2IR 优化会将临时表达式安全融合。reset 仍按赋值顺序读取最新局部值。此 CodeObjectSpec 是可选 adapter 内的受限后端无关协议，Brian2 核心的通用 CodeObjectSpec 扩展仍待后续实现。

## 当前执行计划与 SoA

`src/main.rs` 保留 IR、类型、effect、索引域和 schedule 校验；`src/executor.rs` 在校验成功后构建 reference 执行计划并写出结果 Dump。两种 Rust 引擎都使用 SoA：每个 population 的神经元状态、参数、`lastspike` 和 `not_refractory` 各是一段连续数组，跨组 source/target 保持各自的局部索引。每个突触状态/参数是独立的连续 edge 数组，配合 source CSR、target/delay 数组和环形事件队列。统一 delay 只存一个值。AOT 的 `instance.bin` 和输出 `results.bin` 也按这些 SoA 数组顺序写入。

每条赋值和中间表达式使用独立寄存器版本，指令在最多 256 个神经元的连续分块上运行。缓冲复用，标量代码每阶段执行一次并广播；保留耦合 ODE 的 integrator-stage 快照、赋值顺序、每条冻结赋值的条件和每个算术中间值的有限性检查。条件为假时不会计算 RHS。阈值、突触和 reset 保留逻辑阶段边界；AOT 可以融合已证明局部的相邻循环，但事件进入下一阶段前仍有 barrier，同一目标的累加顺序不变。

reference 仍是通用指令执行器。AOT 已有逐模型代码生成：合法标量外提、单用途临时量融合、只绑定实际读取的输入，以及不应期更新整合；`not_refractory` 使用字节数组，避免 Rust `Vec<bool>` 的位压缩访问。固定不应期在初始化时换算为每个神经元的 `refractory_until` 整数 tick，热循环只做整数比较；放电时同步更新截止 tick、公开的 `lastspike` 和 `not_refractory`。突触使用 Source CSR。统一 delay 继续使用自适应 delayed batch：小批次保存扁平 edge ID，较大批次保存 source ID、到期时展开 CSR；统一零延迟从当期 `fired` 直接投递，不创建环形队列。异质 delay 在初始化时按量化 tick 建立分组 Source CSR，delay 组按降序投递。各路径都保持 source index 和连接创建顺序。

当前没有手写 SIMD，也没有引入 Rayon。AOT 的 `threads=N` 使用常驻 `std::thread` worker，并把 N 视为池上限：独立 neuron state update 按连续 neuron lane 分片；满足局部性证明时 threshold 与该 worker 批次融合，并以 lane 顺序合并 spike，从而保持全局 index 顺序。Linux 默认 `thread_affinity="auto"`：从进程允许的 CPU 集合读取 NUMA/package/core/SMT 拓扑，优先在一个 NUMA node 的不同物理 core 上固定 main lane 与 workers；`"required"` 在无法固定时失败，`"off"` 保留系统调度。procedural fixed-total topology、逐边 weight 和 delay tick 的大数组由实际消费它们的 worker 并行写入，实现 first-touch placement，避免先由主线程把 3 亿条边全部放到单一节点。

对 primary `on_pre`，初始化阶段按 target population 汇总所有安全 projection 的实际入度，再以连续 target 区间近似等分 edge work；同一 target 在跨 source、跨 projection 的路径中始终归同一个 owner，同一 owner 内仍按 projection、source、连接创建顺序执行。uniform delay 共用 routed spike batch；heterogeneous delay 为每个 owner 建立 incoming CSR 和本地 delay queue，source spike 只读广播，worker 不再跨 owner/NUMA 写队列，也不需要 producer×owner 原子同步。兼容且无 pre-side 依赖的 heterogeneous route 合并为一次 target-owner apply dispatch；同 clock population 的 state update/threshold 也共享一次 dispatch。单 population 快路径和通用多 projection 路径复用同一个 `TargetParallelPlan` 资格判定。只有写 postsynaptic/synaptic state、没有 pre-side 写入，且递归路径不存在“读取 pre 状态并写同一 post 状态”的数据依赖时才启用 target-owner event 路径；拓扑小于 2,048 条边、跨 run 待投递 heterogeneous 事件或无法证明安全的模型自动走原串行路径。Event-driven pre/post plasticity 按互不重叠的 active endpoint/edge 分片，post-summed variable 按 target-owner 分区归约；同一 target population 上相互独立的多个 reduction 共用一次 worker dispatch。IR effect analysis 若能证明 summed destination 在运行中既不被方程读取也不被 Monitor 观察，则只在本段最后一个 endpoint tick 求值；每个 `run()`/checkpoint 边界仍发布完整状态。上述路径保持 1/N worker 逐位一致，任何可观测性或独立写入证明不成立时自动保留逐 tick 串行语义。

跨线程调用只在受控 lane dispatcher 内使用 `unsafe`：全池和部分 lane 都使用 cache-line 隔离的 per-worker wake/completion 信号，只发布并等待实际参与者。不参与当前 phase 的 worker 会 park，任务数随后增加时才 unpark。release/acquire barrier 约束借用闭包的生命周期；结果元数据公开 `thread_affinity`、`thread_cpus`、`parallel_state_update`、`parallel_on_pre`、静态 route 数、并行 route 数和融合 event dispatch 数。Source CSR 的运行时 offset 暂用 `usize`，edge ID 与磁盘拓扑使用 u32；事件队列尚未实现正式容量池和完整内存估算器。

## 验证

先运行示例自动构建，或在仓库根目录手动构建，再执行测试：

```sh
CARGO_HOME="$PWD/.cache/rust-cargo" cargo build --release --locked --manifest-path brian2-rust/Cargo.toml
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m unittest discover -s brian2-rust/tests -v
CARGO_HOME="$PWD/.cache/rust-cargo" cargo clippy --offline --locked --manifest-path brian2-rust/Cargo.toml -- -D warnings
cargo fmt --check --manifest-path brian2-rust/Cargo.toml
```

174 个测试及 121 个参数化 subtest 通过。Device 测试覆盖正式注册、`Network.run()`、magic `run()`、默认/独立 clock、namespace、结构化 capability report、report/profile、多 StateMonitor、EventMonitor、分段执行、内存/磁盘 checkpoint、RNG 与 pending-delay 精确恢复，以及跨 run rolling-window State/SpikeMonitor。已知数值验收覆盖 Brian 默认 deterministic method selection、Euler/RK4/Exponential Euler、reset/refractory、Monitor、参数和结果回填。COBA/HH、异构 population、三 population/九投影、subgroup、linked variables、counter RNG/Poisson、TimedArray/SpikeGeneratorGroup、命名 EventStream、动态/存储 subexpression、多 pathway/delay、event-driven STDP、summed variable、f64/i64/bool portable Function Contract、CPU native C ABI 和未来 GPU implementation descriptor 均有 AOT/reference/NumPy 差分或完整性验收。新增突触测试还要求 parallel plasticity 与 target-owned summed reduction 在 1/N worker 下逐位一致；大型 fixed-total、heterogeneous-delay、release/LLVM/host triple 门继续保留。

新增不应期测试覆盖可选冻结、未标记状态继续积分、零/短于 dt/非整数周期/长运行边界、逐语句条件写入、reset 覆盖条件、初始 lastspike、公开结果数组，以及非法 IR/错误结果不发布部分状态。执行测试会阻断 RuntimeDevice 的 CodeObject 创建，检查不会静默回退。负例包括单位、数组/拓扑形状、表达式类型、虚假 effects、错误迭代域、delay、别名、条件写入及跨 phase 临时变量泄漏。三方验收由 `examples/population.py` 和 `examples/cuba.py`（含 `--refractory-ms 2`）单独执行。

## 性能快照

### Litwin-Kumar 派生 plasticity 与长时程门槛

`examples/litwin_kumar_short.py` 是机制/吞吐门，不宣称复现论文：4:1 E/I、event-driven triplet-style E-E plasticity、homeostatic I-E plasticity及两条 post-summed incoming-weight reduction。脚本先校验 Rust 1/8 worker 逐位一致和 Rust/C++ 数值一致，再分别预热 3 次，以交替顺序各 replay 15 次并比较中位数；并行 plasticity、summed reduction 和两个 final-only reduction 必须实际激活，Rust 串行或 8 worker 任一不快于 C++ 单线程都会直接返回失败。2026-09-05 本机 B2IR v21 结果：1,000 E + 250 I、`p=0.3`、10 ms 为 Rust 1 worker 11.026 ms、8 worker 4.320 ms、C++ 1 thread 23.973 ms，即 Rust 串行/并行分别快 2.17×/5.55×；默认 2,000/500、`p=0.1`、20 ms 为 15.388/6.897/49.225 ms，Rust 串行/并行分别快 3.20×/7.14×。提升来自通用 IR 可观测性分析和共享 reduction dispatch，而非 Litwin-Kumar 名称特判；被方程读取或 Monitor 观察的 summed state 仍保持逐 tick 更新，并有 AOT/reference/NumPy 反例测试。

`examples/litwin_kumar_long.py` 是长时程系统门：默认 40 E + 10 I、`p=0.1`、dt=1 ms，两段各 500 生物秒，保留最后 1,000 ticks。2026-09-05 本机 B2IR v21 结果为 1,000 生物秒完成；中点磁盘 checkpoint 为 28,965 bytes，三次 500 秒执行（首跑、续跑、恢复后重放）分别为 0.075/0.075/0.078 s，三个结果 Dump 均为 15,778 bytes，恢复重放逐位一致。它证明长时间轴、bounded recording 和 checkpoint 生命周期成立，不代表该小网络的论文性能或科学结论。

### 统一串行与并发基准

`performance_suite.py` 是统一验收入口：依次运行规模矩阵、异构双 population、事件投递、CUBA、COBAHH、STDP、Litwin-Kumar 派生机制门、PoissonGroup 和 PoissonInput，再以相同线程数比较 Rust AOT 内部 worker 与 Brian2 C++ standalone OpenMP。`--strict` 会在任一正确性门或 AOT≥C++ 性能门失败时返回非零；Litwin-Kumar 项在调用者要求少于 3 次时仍强制至少 3 次稳定 replay。即使中途某项失败，汇总器仍继续执行并保留其他报告：

入口会在创建输出目录或启动子测试前强制执行工具链门禁：解析并记录 Rustup launcher、实际 `rustc` 路径、release、LLVM 和 host triple，默认要求 rustc ≥1.98.0、LLVM ≥22，并要求完整 host triple 与本机原生目标精确一致。通过检查的实际编译器目录会固定在所有子进程 `PATH` 首位，防止检查与实际构建使用不同工具链。可用 `--rustc /absolute/path/to/rustc` 或 `B2_BENCHMARK_RUSTC` 显式选择编译器；降低 `--minimum-rustc`/`--minimum-llvm` 或覆盖 `--expected-rustc-host` 只应用于有意进行的实验性复测，不应写入正式基线。

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/performance_suite.py \
  --output brian2-rust/output/performance-suite-$(date +%Y%m%d-%H%M%S) \
  --repeats 5 --strict
```

本机 2026-09-04 的串行基线为 23/23 个场景通过数值/分布检查和性能门，AOT 相对单线程 C++ 的循环加速范围为 1.06×–4.67×。代表值为 CUBA 1.17×、COBAHH 1.23×、pair-STDP 1.08×、PoissonGroup 4.67× 和 PoissonInput 1.39×。

真正的单进程内部并行覆盖 neuron state-update lane、可证明局部的 threshold 融合，以及安全 uniform-delay `on_pre` 的 target-owner 路由；没有引入 Rayon。AOT 使用常驻 `std::thread` worker，C++ 使用 Apple Clang 21 + Homebrew libomp。2026-09-04 的五次交错统一基准使用 1/2/4 线程；CUBA 为 100,000 neurons、固定 fan-out 8，COBAHH 为 4,000 neurons、约 319k 连接和 10,000 ticks：

| 模型 | 线程 | AOT | C++ OpenMP | AOT 自身加速 | C++ 自身加速 | C++ / AOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUBA 100k | 1 | 115.557 ms | 229.033 ms | 1.00× | 1.00× | 1.98× |
| CUBA 100k | 2 | 55.504 ms | 181.456 ms | 2.08× | 1.26× | 3.27× |
| CUBA 100k | 4 | 39.233 ms | 184.340 ms | 2.95× | 1.24× | 4.70× |
| COBAHH | 1 | 3014.453 ms | 3574.600 ms | 1.00× | 1.00× | 1.19× |
| COBAHH | 2 | 1455.518 ms | 2292.300 ms | 2.07× | 1.56× | 1.57× |
| COBAHH | 4 | 1392.711 ms | 2353.510 ms | 2.16× | 1.52× | 1.69× |

所有线程档位的 AOT 结果 Dump SHA-256 完全一致，C++ OpenMP 的完整状态、轨迹、spike 和连接也通过严格差分；AOT 在 6/6 个绝对性能比较中快于 C++。统一采样后半段主机负载或热状态持续变化，两个 workload 都被标记为 spread 超过 15%；对同一 AOT 二进制的无重建连续复测更适合观察内核扩展：CUBA 的 1/2/4-thread 中位数为 114.078/56.372/44.938 ms，4 线程自身加速 2.54×；COBAHH 的 1/4-thread 中位数为 2753.294/827.778 ms，自身加速 3.33×。CUBA 改造前的 4-thread 基线为 104.366 ms，新路径约快 2.32×。默认仍为单线程，用户通过 `set_device(..., engine="aot", threads=N)` 显式启用；工作量不足或安全条件不成立时自动保留串行路径。8 线程跨性能核/能效核的拓扑感知调度尚未在本轮重新调优。完整统一报告位于 `output/target-owner-threading-20260904/report.md`；独立多进程吞吐仍可用 `concurrency_benchmark.py` 单独测量，但不再冒充内核并行。

### Degree-balanced target-owner A/B

`target_partition_benchmark.py` 使用同一个 20,000-neuron、1,000,000-edge AOT 实例生成两个二进制，只把新入度均衡 owner map 替换回旧的 `target * threads / N`，然后执行五次交错 replay。该压力拓扑把 97,000,000 次事件投递集中到前 1/8 target；所有策略和线程档位的 Dump SHA-256 完全一致：

| 线程 | 旧 index 分区 | degree-balanced | 相对旧分区 | 新策略自身扩展 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 77.052 ms | 77.601 ms | 0.99× | 1.00× |
| 2 | 65.104 ms | 42.438 ms | 1.53× | 1.83× |
| 4 | 72.789 ms | 38.673 ms | 1.88× | 2.01× |
| 8 | 71.695 ms | 43.274 ms | 1.66× | 1.79× |

标准均匀 CUBA100k 的 4-thread 热循环为 43.627 ms，与改造前同机 42.770 ms 的差异落在高噪声测量范围；COBAHH 为 818.057 ms，对应改造前 944.644 ms，但同样不能把主机状态差异归因为分区策略。因而可归因结论限定为：degree balancing 对均匀拓扑无明确热循环回归，在事件主导的偏斜拓扑上提升 1.53–1.88×；8 线程仍受性能核/能效核调度、barrier 与内存带宽限制。可复现报告位于 `output/target-partition-benchmark-20260904/report.md`。

`dispatcher_benchmark.py` 在同一份生成模型和实例上编译旧 shared-counter 与新 per-worker/idle-parking 调度，并按线程档位做 ABBA block replay。报告使用每个 block 的配对加速比；配对 spread 超过 15%，或两个实际上等价的 1-thread 串行 control 相差超过 5%，都会把本轮标为无效，避免把主机负载和热状态误记成调度收益；原始 min/max spread 仍会保留作诊断：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/dispatcher_benchmark.py \
  --output brian2-rust/output/dispatcher-barrier-$(date +%Y%m%d-%H%M%S) \
  --repeats 5 --strict
```

本机首轮 8-thread A/B 处于明显外部负载下，整体 paired spread 超过 15%，因此未作为正式性能基线。诊断结果中，CUBA 的聚合 replay 原始中位数基本不变；COBAHH 的 8-thread 五个配对 block 均有改善，配对中位加速 1.45×（单 block 为 1.13×–1.67×）。这只能作为 idle-worker parking 有效的强信号，仍需在安静主机上通过上述 strict 门后才能接受。

运行完整串行基准：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/benchmark.py --repeats 5 --cpu-label 'Apple M3; 1 thread; paired AOT/C++'
```

8 个场景均为 1000 ticks、dt=0.1 ms、固定 2 ms refractory，并记录 8 个神经元；CUBA 每个源有 8 条静态连接和 0.3 ms 固定延迟。四后端使用 float64、单线程、同一模型和监测输出。C++ 使用 `-O3`、禁用 fast-math/FMA contraction，`openmp_threads=0`；构建后自动断言生成源码没有 OpenMP pragma、makefile 没有 `-fopenmp`。`make -j1` 只控制首次编译，本身不影响模拟循环。表中 AOT/C++ 为两种原生二进制预热后交错执行、每轮反转顺序的五次循环中位数；reference/NumPy 为各自五次中位数。循环包含内存监测，不包含结果 Dump、进程启动和结果读取；`C++ / AOT ≥ 1` 表示 AOT 不慢于 C++。

| 场景 | Rust AOT | Rust reference | NumPy | C++ | C++ / AOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| LIF 100 | 0.082 ms | 0.939 ms | 21.356 ms | 0.193 ms | 2.37× |
| LIF 1,000 | 0.669 ms | 6.103 ms | 27.885 ms | 1.153 ms | 1.72× |
| LIF 10,000 | 6.793 ms | 57.807 ms | 59.043 ms | 8.392 ms | 1.24× |
| LIF 100,000 | 68.866 ms | 564.341 ms | 334.843 ms | 81.753 ms | 1.19× |
| CUBA 100 | 0.121 ms | 1.372 ms | 27.134 ms | 0.263 ms | 2.17× |
| CUBA 1,000 | 0.921 ms | 9.883 ms | 38.059 ms | 1.464 ms | 1.59× |
| CUBA 10,000 | 9.824 ms | 89.036 ms | 87.323 ms | 12.104 ms | 1.23× |
| CUBA 100,000 | 108.243 ms | 887.856 ms | 565.573 ms | 120.839 ms | 1.12× |

所有场景的完整状态轨迹、最终状态、spike tick/index/count、`lastspike` 和 `not_refractory` 都通过差分。AOT 在 8/8 场景快于 reference、NumPy 和 C++，通过当前支持范围“至少不慢于 C++”的中位数性能门；相对 C++ 的加速为 1.12–2.37×。取消小模型 update/threshold 融合的交错 A/B 提升为 1.23–1.32×；固定 refractory 预计算为整数截止 tick。编译器自动向量化属于单线程 SIMD，不是 OpenMP 并行。

优化前对 CUBA 100k 的 AOT 热循环做 7 次分阶段诊断，中位数 112.3 ms：state update 51.7 ms、threshold 42.1 ms、spike 记录 2.6 ms、edge enqueue 6.4 ms、delivery/on_pre 8.5 ms、reset 0.4 ms，其余 0.7 ms。约 533 万次突触投递的 enqueue+delivery 合计占 13.3%；这组数据直接促成了整数 refractory 截止 tick 优化。当前基准的事件路径不是主导项；更高 fan-out、发放率或更复杂 on_pre 仍需单独压力测试。

首次运行包含 B2IR 校验、源码/实例生成、rustc 编译、执行和结果读取。AOT 的逐模型编译约 0.38–1.05 s，C++ 单任务编译约 4.69–6.71 s；例如 100k CUBA 首次总计 AOT 3.71 s、C++ 7.75 s。输出阶段已经与模拟循环分离：每次先记录原始 f64/spike 到内存，成功后顺序写出二进制 Dump。历史表格仍反映改造前的 CSV/大 JSON 结果协议，不应直接用于评价当前结果载入开销。

完整报告：[report.md](output/benchmark-event-adaptive-20260903/report.md)，原始逐次计时、交错执行顺序、误差、编译信息和源文件 SHA-256：[report.json](output/benchmark-event-adaptive-20260903/report.json)。benchmark 产物不随源码提交；命令会生成新的 `output/benchmark-<suffix>/`。

### 双 population 性能基线与优化

`two_population_benchmark.py` 比较两个独立状态 schema 的 population 和一个跨组 `Synapses`。每组 10,000 neurons，80,000 条 source→target 边，运行 100 ms；source dt=0.1 ms，target 分别使用 0.1 ms 和 0.2 ms。两组拥有不同方程、threshold/reset 和 refractory，突触包含 clock-driven 状态、固定 delay，并分别记录 8 个神经元：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/two_population_benchmark.py --repeats 5
```

初始通用 edge queue 路径分别为 56.276 ms 和 46.602 ms。profile/A-B 定位出两个独立问题：统一 delay 仍逐次复制 edge ID；默认 panic unwind 边会阻止 mixed-clock threshold 循环优化。AOT 现在对统一 delay 只排队 source ID、到期后用 CSR 展开，refractory 使用预计算整数截止 tick，standalone 使用 `panic=abort`。结果读取的逐 spike Python refractory 校验和逐元素位模式解码也已改成 NumPy 批处理。

本机 2026-09-04 完整性能套件的单线程 warm-loop 中位数（5 次）：

| 场景 | AOT | Reference | NumPy | C++ | C++ / AOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 同 clock，0.1/0.1 ms | 30.258 ms | 1522.372 ms | 212.141 ms | 31.659 ms | 1.05× |
| 异构 clock，0.1/0.2 ms | 25.213 ms | 1491.547 ms | 174.855 ms | 26.858 ms | 1.07× |

四后端的完整状态、spike、refractory 和最终突触状态均通过差分；两个场景均完成 538,928 次突触投递，分别产生 83,098 和 83,201 个 spike。AOT 分别比同轮 C++ 快 5% 和 7%。AOT 进程 wall 中位数为 38.5/34.3 ms，C++ 为 50.7/60.3 ms；AOT 峰值 replay RSS 为 9.9–11.3 MiB，C++ 为 11.2 MiB。AOT Dump 写出中位数为 1.1–1.3 ms，Python memmap 载入及完整语义校验中位数为 11.6–14.1 ms。历史优化结果仍保存在 [report.md](output/two-pop-optimized-20260903/report.md) 和 [report.json](output/two-pop-optimized-20260903/report.json)。

`cuba_throughput.py` 是真实双 population、四 projection 的性能硬门：4000 neurons（80/20）、`p=0.02`、固定 0.3 ms delay、1000 ticks，只记录每组 4 个神经元。AOT 与 C++ 使用完全相同的显式 seeded 拓扑、严格 float64 和单线程配置；脚本先比较最终状态、稀疏轨迹、spike 及连接数，再预热并交错 replay，若 AOT 热循环中位数慢于 C++ 就返回失败：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/cuba_throughput.py --output /tmp/cuba-throughput
```

本机 2026-09-04 本轮回归的 7 次中位数为 AOT 4.742 ms、C++ 5.546 ms，AOT 快 1.17×；319,232 条连接的结果完全一致。代码生成器会将具有相同 source 范围、clock 和统一 delay 的相邻 projection 融合为一条 spike route，共享 source 过滤和 ring queue，同时保持各 projection 的 Brian 调度顺序。结果 Dump 使用 8 KiB 分块编码和 64 KiB buffered write，避免每个标量一次 `write_all`；该优化不计入热循环门，但降低了含输出 replay 的固定成本。

### 事件投递压力测试

`event_benchmark.py` 对统一 delay 分别强制 flat edge、source CSR 和自适应布局；异质 delay 使用按 delay 分组的自适应布局。两类场景都加入同模型的单线程 C++，原生二进制预热后交错执行。每种布局都从同一初态运行，并要求最终状态、完整轨迹和 spike 一致。脚本还记录 adaptive AOT 和 C++ 子进程的峰值 RSS：

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/event_benchmark.py --repeats 5
```

| 场景 | 投递数 | Adaptive | C++ | C++ / AOT | AOT RSS | C++ RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 稳态、fan-out 8、简单 on_pre | 539,904 | 9.355 ms | 11.231 ms | 1.20× | 8.4 MiB | 8.7 MiB |
| 同步 burst、fan-out 128、简单 on_pre | 25,600,000 | 18.172 ms | 35.449 ms | 1.95× | 43.5 MiB | 50.7 MiB |
| 同步 burst、fan-out 128、两条 on_pre | 25,600,000 | 28.367 ms | 41.060 ms | 1.45× | 43.5 MiB | 50.6 MiB |
| 长尾/hub out-degree burst | 3,872,000 | 3.939 ms | 5.222 ms | 1.33× | 8.8 MiB | 13.5 MiB |
| fan-out 32、delay 12.7 ms、多次 queue wrap | 7,040,000 | 8.073 ms | 11.651 ms | 1.44× | 17.1 MiB | 39.7 MiB |
| 异质 delay、稳态、fan-out 8 | 534,826 | 10.890 ms | 11.726 ms | 1.08× | 9.4 MiB | 10.4 MiB |
| 异质 delay、同步 burst、fan-out 128 | 24,160,000 | 28.853 ms | 47.272 ms | 1.64× | 44.8 MiB | 77.7 MiB |
| 80,000 条突触、clock-driven trace ODE | 538,928 | 20.527 ms | 21.846 ms | 1.06× | 8.5 MiB | 9.9 MiB |

统一 delay 且没有可写突触状态时，AOT 会按 source CSR 顺序预打包不可变 source/target/parameter 数组，消除热循环中的 edge-ID 间接访问，同时保持每个 source 的 Brian edge 顺序。两个 2560 万投递 burst 场景相对 flat 分别提升 1.47× 和 1.27×；AOT 峰值 RSS 虽升至 43.5 MiB，仍低于 C++ 的约 50.6 MiB。异质 delay 和含 clock-driven 突触状态的路径不启用该专化，8/8 压力场景仍全部快于 C++。历史压力报告保存在 `output/event-*.md/json`；C++ 使用相同的 `-O3`、严格浮点和 OpenMP=0 设置。

## Gate 0 接下来要完成

1. AOT 低扰动 phase timing、全局 target ownership、本地 heterogeneous delay queue 和同 clock population/event dispatch 融合已完成；下一步补 NUMA-local owner CSR first-touch、压缩跨 NUMA spike exchange，并根据当前 RSS 数据形式化事件队列的最坏内存上界。
2. 任意数量异构 population/Synapses、并行 event-driven plasticity 与 target-owned summed reduction、二进制结果、counter RNG/Poisson、常用静态输入、完整 COBAHH/STDP 门、多段运行、checkpoint、rolling-window monitor、命名 EventStream、f64/i64/bool 可移植纯 Function Contract，以及 CPU `b2ir-c-abi-v1` Function implementation 已完成。`examples/litwin_kumar_short.py` 验证短窗机制及 worker 一致性；`examples/litwin_kumar_long.py` 完成 1,000 生物秒、有界结果和 checkpoint 精确重放。CUDA/Metal Function implementation、周期 progress、全历史 streaming sink 和细粒度逐 CodeObject profiling 仍待完成。
3. B2IR v1 schema、Definition/Instance/Run 分层及 schedule/effect/time/RNG/hash ADR 已冻结，并由 golden/migration corpus 约束；下一步单独冻结 native Artifact Manifest、补内存估算器和更多隔离环境 workload。
4. 许可审查及 native artifact threat model。本目录沿用仓库 CeCILL 2.1 声明，不代表正式许可审查已经完成。

## Empirical FlyWire benchmark

The complete unthresholded FlyWire v783 weighted graph can now be imported with bounded batch processing and executed through a generic binary CSR topology API. See [FLYWIRE.md](FLYWIRE.md) for provenance, model assumptions, limits and the Rust/C++ comparison commands.

Experimental browser GPU execution and measured boundaries: [WEBGPU.md](WEBGPU.md).

Frontend-only editable equations, AdEx / Quadratic IF and larger browser populations: [EQUATION_LAB.md](EQUATION_LAB.md).

Load and execute exported Brian2 networks entirely in the browser: [WASM_IMPORT.md](WASM_IMPORT.md).
