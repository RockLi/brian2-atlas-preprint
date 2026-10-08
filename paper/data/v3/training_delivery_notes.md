# 2026-10-02 训练验收记录

交付范围及 API：[MPI_TRAINING.md](../../MPI_TRAINING.md)、
[NATIVE_TRAINING.md](../../NATIVE_TRAINING.md)。
所有运行使用工作树 Python 包和隔离 Cargo target，没有安装共享运行时或重启平台服务。
本机 Apple Silicon、原生 Rust 1.98.1、MPICH、真实 Metal；没有 CUDA 或跨主机训练验收。
此目录是功能及确定性证据，不是训练吞吐、公开数据集准确率或扩展性基准。

## 最终结果

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| MPI STDP、分段、pending、checkpoint、结构代际保护及窄队列 | 30 passed；其中 19 个训练测试和 11 个窄队列测试 | [XML](mpi-final.xml)、[log](mpi-final.log) |
| Native CPU/Metal surrogate BPTT、训练及恢复 | 20 passed；真实 Metal 四种 reset/detach、更多隐藏层、mask/TBPTT、fresh-process replay | [XML](native-final.xml)、[log](native-final.log) |
| 既有 Device MPI 续跑入口 | 1 passed，79 deselected | [XML](mpi-lifecycle-final.xml) |
| 平台六种 LIF 模型的训练/冻结/候选重连/恢复 | 六种均 passed，最终保存契约之后重跑 | [JSON](platform-families.json) |
| 其他平台动力学与结构策略探针 | custom/feedforward/weight，AdEx/reservoir/fixed，Izhikevich/recurrent/random-rewire 均 passed | [custom](platform-custom-weight.json)、[AdEx](platform-adex-fixed.json)、[Izhikevich](platform-izhikevich-rewire.json) |
| CPU 与 CPU/Metal MPI 交付示例、新进程恢复 | time/weights/spikes/pending 完全相同；混合末段 dispatch `[0,7]` | [JSON 与实际 MPI runtime](mpi-examples.json) |
| CPU 与 Metal 原生分类示例、60 Adam steps、新进程恢复 | loss 1.136871→0.061968；held-out `[0,0,1,1]`；weights/optimizer/loss 完全恢复 | [JSON](native-examples.json) |

CPU STDP 的 1/2/4 ranks 与独立 Rust reference 比较数组原始位模式，而非 allclose。
分段比较包括绝对时间、refractory、神经元状态、w/apre/apost/slow/lastupdate、bool/i32
突触状态、counter RNG、monitor 历史及 pending。两种延迟布局还覆盖长度 1 tick 的运行段。
改变 rank、设备顺序、engine 后的继续执行或保存被拒绝；损坏或不匹配快照在数组回填前拒绝。

Metal 监督训练直接运行 MSL forward/backward kernel；结果声明
`native-metal-forward-backward-f32-host-optimizer-f64`，没有把 host STDP 或外部读出更新
计为 GPU BPTT。CPU/Metal 之间对受控输入用显式容差；同一后端新进程重放比较完整 optimizer
状态。MPI 的 Metal 范围是神经元 state update，STDP/threshold/reset/队列仍在各 rank CPU。

六种平台模型为 diehl-cook、reservoir、local-receptive、feedforward、convolutional、recurrent。
探针只读调用平台实际 builders/present/consolidate/readout/freeze/structural/store/restore，
并将独立 Device 切换为 MPI；平台当前自己的 engine 配置门未更改。
其他动力学探针在最后的 store/engine guard 前完成；最终六模型矩阵在该 guard 后完成。

## 回归与开发中发现的问题

保存初次失败和重测证据，避免把多次重叠运行累加为一个虚假的通过总数。

| 检查过程 | 初次结果 | 修复后结果 |
| --- | --- | --- |
| CPU MPI / MPI GPU / timing / 当时的 training 联合运行 | 105 passed，17 failed；procedural initializer 错读 pending，另一次在快照期间重建 validator 触发预期版本拒绝 | [17 项定向重测全部通过](legacy-mpi-retest.xml)；最终训练矩阵另见上表 |
| Device / projection/queue compaction / additive ring / training | 92 passed，9 failed，1 deselected，17 subtests passed；post 工作计数改变旧 source anchors，空 pending initializer 与 u32 类型冲突 | [9 项失败重测全部通过](legacy-compaction-retest.xml)；最终完整窄队列另见上表 |
| 旧 f32 STDP / parameter sweep / precompiled Rust 与 C++ replay | 38 passed，12 failed；baseline shared clock/target Device 生命周期污染 | [整套 50 passed](legacy-stdp-final.xml) |

初次 XML：[MPI](legacy-mpi-initial.xml)、[Device/compaction](legacy-device-compaction-initial.xml)、
[旧 STDP](legacy-stdp-initial.xml)。这些回归阶段早于最后保存契约，未将一次定向重测描述成
所有历史测试在最终源码上重新全跑。旧 STDP 50 项含 CPU f32 和 C++ 验证，其名称中的 GPU
不意味着这 50 项都跑了真实 GPU。真实 GPU 证据来自最终 MPI/Native 测试及示例。
部分 XML 把 pytest subtests 记录为独立 testcase；与顶层测试计数的区别保存在 manifest。

## 复现

仓库根目录，已有 Brian2/Numpy/pytest 的虚拟环境、Rust、MPI、clang：

```bash
CARGO_TARGET_DIR="$PWD/brian2-rust/target-mpi-training" \
  cargo build --offline --release --manifest-path brian2-rust/Cargo.toml --bin b2-runner
CARGO_TARGET_DIR="$PWD/brian2-rust/target-native-training" \
  cargo build --offline --release --manifest-path brian2-rust/Cargo.toml --bin b2-train
export PYTHONPATH="$PWD/brian2-rust/python"
export B2_RUNNER="$PWD/brian2-rust/target-mpi-training/release/b2-runner"
export B2_TRAIN_RUNNER="$PWD/brian2-rust/target-native-training/release/b2-train"
export MPLCONFIGDIR=/tmp/atlas-training-mpl
B2_TEST_MPI=1 B2_TEST_GPU=1 B2_TEST_MPI_GPU=metal \
  .venv/bin/python -m pytest -q \
  brian2-rust/tests/test_mpi_training.py brian2-rust/tests/test_mpi_queue_compact.py
B2_TEST_GPU=1 .venv/bin/python -m pytest -q brian2-rust/tests/test_native_training.py
```

新 checkpoint 契约禁止恢复期间改变编译器、runtime 二进制或包源码。
先完成构建再开始测试；不要在同一次 checkpoint/restore 回归中重建 runner。
测试里的重复 Device 编译使用 opt-level 0；独立 reference parity 与既有优化路径同时保留。
MPI 初始 Pair/Triplet parity 直接编译入口仍使用默认优化级别。

平台兼容性探针（用自己的只读平台 checkout 路径）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  brian2-rust/tools/verify_next_brain_mpi_training.py \
  --platform-root /path/to/next-brain.ai --output /tmp/platform-mpi.json
```

完整示例和恢复命令见两个交付文档。平台探针采用 opt-level 0 与小规模候选连接。
大型 production 网络、跨主机资源隔离及性能测量不在本次测试范围内。

[verification.json](verification.json) 保存工具链、runner hash、MPI 包源码 hash、交付文件
hash、测试 XML 摘要与平台只读 reference commit。工作树原有未提交/未跟踪工作保留，
所以这些 hash 标识当前交付文件，并不宣称存在一个干净的交付 commit。
