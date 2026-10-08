# Atlas 原生 LIF 监督训练 v1 / v2

2026-10-02。新增 `b2-train` 执行器：Rust CPU 和真实 Metal 都执行硬脉冲前向、
声明的 surrogate VJP 和时间反传；SGD/Adam 在原生 Rust 中更新隐藏层和输出层权重。
Python 仅传输计划/输入/状态、构建 Metal 动态库和保存快照，未使用 Python autodiff
或外部深度学习运行时。`b2-lif-training-plan-v1` 保留 dense 前馈入口；新增
`b2-lif-training-plan-v2` 支持显式循环投影和共享参数，包括卷积核。
它们都是限定的原生训练计划，不是一般 B2IR 的自动可微编译器。
v2 API 和验收见 [循环与共享权重训练](NATIVE_TRAINING_GRAPH.md)；
v1 首阶段记录见 [验收证据](mpi-evidence/training-20261002/README.md)。

后续 v5 的运行时 Poisson 采样与似然梯度见
[Poisson 训练契约](NATIVE_TRAINING_POISSON.md)。当前 CPU／Metal 动态训练已接入；
CPU 新增跨 action 的共享抽样缓存、只计一次的似然梯度及全别名零 rate 回放，
缓存随 carry／checkpoint 保留。Metal／本地 MPI 已接入 GPU 持久抽样缓存、
carry／恢复与设备 count 校验，使用显式后端 f32 profile；持久缓存阶段回归 1756 passed／658 NVIDIA-only skipped／0 failed，
见[持久缓存验证记录](mpi-evidence/training-poisson-gpu-checkpoint-20261005/README.md)。
CPU／实际 Metal 已支持零 rate 的完整轨迹边界梯度；CPU／动态 GPU 的无关奇异导数裁剪见 [VJP 活动路径契约](NATIVE_TRAINING_VJP_ACTIVITY.md)；静态 v4 CPU／Metal Poisson 已实现，见[本轮静态 GPU 验证](mpi-evidence/training-poisson-static-gpu-20261005/README.md)。
静态 v4 TimedArray、SDE 时间表扩散系数和 CPU／Metal Poisson 表值率的当前范围见[时间输入契约](NATIVE_TRAINING_TIME.md)。
静态 v3 Poisson、更多 Brian 组合和 NVIDIA 实机验收仍未完成。

## 最小调用

```python
import numpy as np
from brian2_rust import NativeLIFTrainer, lif_training_plan

plan = lif_training_plan([2, 2, 2], backend='cpu', beta=0.8,
                        learning_rate=0.04, surrogate_slope=3, logit_scale=3)
trainer = NativeLIFTrainer(plan, runner='/path/to/b2-train',
                          weights=[[1.2, .4, .4, 1.2], [.2, .8, .8, .2]])
x = np.zeros((4, 24, 2))        # batch × time × input
x[:2, :, 0] = 1; x[2:, :, 1] = 1
labels = [0, 0, 1, 1]
for _ in range(60):
    result = trainer.step(x, labels)
validation = trainer.evaluate(x, labels)  # 不更新已初始化的参数/optimizer
trainer.store('/tmp/native-training.checkpoint')
```

每个矩阵按 source-major 展平，索引 `source * target_count + target`。输入是确定的脉冲
幅度序列；当前没有隐式 Poisson 编码、数据下载或平台数据管线。标签为输出类索引。
`gradients` 返回权重及 initial membrane 的 VJP，保留参数；首次无显式权重时会初始化。
`step` 返回更新后的 `state`，同一次输出的 loss/spikes/gradients 属于更新前前向。

隔离构建与完整示例，从仓库根目录运行：

```bash
CARGO_TARGET_DIR="$PWD/brian2-rust/target-native-training" \
  cargo build --offline --release --manifest-path brian2-rust/Cargo.toml --bin b2-train
export B2_TRAIN_RUNNER="$PWD/brian2-rust/target-native-training/release/b2-train"
export PYTHONPATH="$PWD/brian2-rust/python"
.venv/bin/python brian2-rust/examples/native_supervised_training.py \
  --backend cpu --steps 60 --checkpoint /tmp/native-cpu.checkpoint
.venv/bin/python brian2-rust/examples/native_supervised_training.py \
  --backend metal --steps 60 --checkpoint /tmp/native-metal.checkpoint
```

`--resume` 在新进程中用相同计划和二进制恢复，并再执行 `--steps` 个 optimizer steps。
Metal 需要 macOS、Metal GPU 和 clang；在临时目录编译动态库，没有 CPU 静默回退。
CLI 也可直接接受 request.json/result.json；Rust serde 拒绝未知字段、后端和操作。

Python 协调器为每次调用启用 `B2_TRAIN_ERROR_RESULT=1`，原生失败时在 stderr／
MPI abort 前原子写入 `b2-native-training-error-v1` 结果，错误文本限制为
4,096 个 UTF-8 字节。读取端严格核对 schema、字段和大小；缺失或无效时沿用
stderr 错误。静态 Poisson 的 collective peer 使用专用失败类型，避免覆盖 owner
的具体原因。本机 serial／2／8 rank 丢弃 stderr 的重复失败已通过专项验证，
失败仍完整回滚。直接 CLI 默认不写失败结果；成功结果格式保持不变。
MPI launcher 退出后，peer 仍可能完成错误文件写入；请求目录清理会有上限地
退避重试，不重跑训练。持续清理失败时保留已有原生错误并附上残留目录信息。

## 前向、反向与时间语义

层数包含输入、至少一个隐藏层和输出；最多 17 个宽度，每层宽度 1..65,536。
每个非输入层独立 `beta`、`threshold`，`beta = 1 - dt/tau` 是离散 Euler 衰减。
同一 tick 先更新全部层、判定全部阈值，再加突触输入，最后 reset，与当前 Brian
groups/thresholds/synapses/resets 顺序一致。新输入影响下一 tick 的阈值。

对每层，`u[t] = beta*v[t-1]`，硬脉冲 `s[t] = int(u[t] > threshold)`，
`a[t] = u[t] + x[t] @ W`；第一层 x 是外部输入，后续层 x 是上游同 tick 的 s。
subtract reset 为 `v[t] = a[t] - threshold*s[t]`；zero reset 为
`v[t] = a[t]*(1-s[t])`，因而发放时也清除同 tick 的新输入。

损失为输出平均发放率乘 `logit_scale` 的 batch-mean softmax cross entropy。
硬阶跃反向使用显式 `phi(margin) = scale/(1+slope*abs(margin))²`，
`margin=u-threshold`。`detach_reset=True` 只停止 reset 操作中的脉冲梯度，
输出 loss 与上游传播仍使用 surrogate。attached subtract 的 reset 导数为
`1-threshold*phi`；attached zero 为 `1-s-a*phi`；detach 时分别为 1 与 `1-s`。
权重梯度、初始膜电位梯度和跨 tick adjoint 都由原生循环计算。

独立测试固定基准硬脉冲与 phi，构造局部平滑 surrogate 线性化，再做有限差分核对
VJP。它不是对不可微硬阶跃的普通有限差分；文档不将 surrogate 当作硬阶跃真导数。
另外两种 reset 的前向与由 Brian 模型导出的独立 Rust canonical reference 比较。

`tbptt_window=None` 做完整 BPTT；正整数 W 在每个 W 的时间边界截断 adjoint，
不截断前向膜电位。W 大于等于序列长度与完整 BPTT 相同。
当前仍保存整个获准序列的 tape，TBPTT 不降低 tape 分配，没有流式分窗 optimizer。
一次请求更新一次 optimizer；手工把序列切成多次 `step` 会改变优化语义。

默认每个样本从零膜电位开始。可传 `initial`，形状 batch × 所有非输入层宽度之和，
或 `initial='carry'` 使用上一次 `step` 的 final membrane；batch 身份和大小需保持一致。
carry 跨请求保留前向状态，但不保留跨请求反向图。`evaluate`、`gradients` 不改变
trainer 的 carry/tick 计数；tick 计数每次 step 增加序列长度，与物理秒无隐式映射。

## 参数、优化器、结构与快照

提供 SGD、Adam（beta1=.9、beta2=.999、epsilon=1e-8；显式计划可配置），无隐式
weight decay 或梯度裁剪。Native SplitMix64 用 seed 初始化权重并保存 RNG 状态。
`trainable=[False, True, ...]` 冻结对应矩阵；v2 对应投影的参数组。
mask=0 的权重/梯度/moments 为零。
`update_mask(new_masks, growth_weight=...)` 仅用于 batch 边界，形状必须不变；裁剪边归零，
生长边重新初始化，变化边的 Adam moments 清零。此接口没有可微结构搜索或新增矩阵。

`store` 原子写入校验 JSON，包含完整 plan、weights、moments、step、RNG、carry membrane、
elapsed_ticks、runner SHA256 和 Metal source SHA256。`restore` 在赋值前核对 plan、
运行时和 checksum；不支持跨 backend、precision、拓扑或版本迁移。改变 mask 后恢复需
先重建快照中的同一计划。真实 Metal 动态库在每次调用前检查 hash，损坏或 GPU 错误会失败。
快照是运行边界快照，没有保存活跃 backward tape，也没有恢复请求中途执行。

默认 `max_tape_bytes=64 MiB`，native 上限 1 GiB；完整请求 JSON 最大 64 MiB。
预检包含 admitted forward/backward tape、参数/moments/梯度/mask、输入、状态与 scratch；
Python helper 在默认 mask 分配前检查参数预算。Metal 还计入 GPU tape、batch 私有梯度、
CPU/GPU 副本与 scratch。估计不是进程 RSS 上限：JSON 解析、编译器、GPU 驱动、allocator
开销仍可能增加内存。形状/预算不合格、非有限输入/中间值/梯度或 optimizer 溢出明确报错。

## 后端与尚未实现范围

| 范围 | CPU | Metal |
| --- | --- | --- |
| 多层 dense 或固定掩码 LIF；v2 循环/共享投影 | 原生 f64 前向/反向 | 真实 MSL f32 前向/反向 |
| 隐藏层及输出层权重梯度 | 支持 | 支持 |
| initial VJP/full BPTT/TBPTT | 支持 | 支持 |
| zero/subtract、detach/attached reset | 支持 | 支持 |
| SGD/Adam | 原生 Rust f64 | 原生 Rust f64 host optimizer |
| 归并 | 固定 batch/time/edge 顺序 | 每 sample GPU lane，host 按 batch 顺序 f64 合并 |
| 设备证据 | `native-cpu-f64` | `native-metal-forward-backward-f32-host-optimizer-f64` 与 `gpu_dispatches` |

Metal kernel 以每 sample 一个 lane 的循环执行完整序列与反传，用 batch 私有梯度避免
浮点 atomics。当前以正确性为目标；CPU/GPU 权重复制、JSON 往返和 kernel JIT 有开销，
不宣称加速或大模型吞吐。GPU 与 CPU 精度不同，硬阈值附近可能产生不同脉冲；测试在
指定输入上以明确容差核对梯度和 logits，同一后端快照重放检查一致性。

v1 不支持 recurrence 或卷积参数共享；这两项现在通过显式 v2 投影图提供。
当前仍不支持可学习延迟、refractory、一般 Brian 方程/CodeObject、池化、
可训练 beta/tau/threshold、一般 reduction VJP、CUDA 或 MPI BPTT。
v2 不会自动把既有 MPI STDP 模型转换成可微训练模型。
未来一般 backward-plan 需定义可微 op/VJP 注册、共享 parameter id、tape layout/预算、
状态边界、优化器 ownership 以及确定性通信；相关 MPI 约束见
[MPI_TRAINING.md](MPI_TRAINING.md)。这些是后续里程碑，当前没有未声明的 CPU fallback。

## 验收

新增 20 项测试包括独立 surrogate VJP、时间截断、非零 initial、两种 reset 的 canonical
前向、隐藏层学习、冻结权重、独立验证数据、SGD/Adam/RNG 状态、磁盘和 fresh-process
恢复、mask/moments、预算及损坏拒绝，以及真实 Metal 的四种 reset/detach 组合、
更多隐藏层的 mask/TBPTT、前向/反向/更新和新进程恢复。
60 步小型分类示例，CPU loss 约 1.136871→0.061968，Metal 约 1.136871→0.061968，
两者 held-out predictions `[0,0,1,1]`。这是功能验收数据，不是公开数据集准确率指标。

```bash
B2_TEST_GPU=1 .venv/bin/python -m pytest -q brian2-rust/tests/test_native_training.py
```

## 方程与分布式后续实现

当前 CUDA、CPU MPI BPTT，以及可训练神经元系数/阈值的标量方程编译入口见
[NATIVE_TRAINING_V3.md](NATIVE_TRAINING_V3.md)。本文先前的阶段边界描述为 v1 交付时状态。

最新多状态与 Metal MPI 能力见 [v4 / GPU MPI](NATIVE_TRAINING_V4.md)。


旧 scalar/vector GPU VJP 也已接入 requested-leaf 活动路径裁剪：masked sqrt(0)
等有限 forward 不再被无关奇异导数污染，v4 共用数学/惰性解释器使用独立 metadata
模式，保留 v5 物理状态/detach 语义。17 完整模块本地验收 1,038 passed / 486
NVIDIA-only skipped / 0 failed，另 23 Rust 项通过。证据和剩余原目标范围见
[本阶段验收](mpi-evidence/training-static-vjp-activity-20261005/README.md)及
[NEXT](mpi-evidence/training-static-vjp-activity-20261005/NEXT.md)。
