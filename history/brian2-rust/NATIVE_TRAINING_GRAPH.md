# 原生循环与共享权重训练 v2

2026-10-02，第二阶段。CPU 和真实 Metal 现可训练同层循环、自连接、跨层反馈，以及
一组权重被多条连接共享的 LIF 网络。卷积 helper 把空间位置映射到共享核参数编号；
原生执行器对共享参数汇总梯度，并且每个参数只保存一份权重、Adam moments 和更新。
前向、反向及 optimizer 均由 Atlas 原生代码执行。入口仍为 `NativeLIFTrainer`。

## 计划与最小示例

给 `lif_training_plan` 传 `projections` 会选择 `b2-lif-training-plan-v2`。
不传时保留 v1 dense 前馈入口；相同 dense 网络的 v1/v2 输出、梯度、optimizer 状态
在 CPU/Metal 分别做了逐值一致性比较。

```python
from brian2_rust import (lif_training_plan, NativeLIFTrainer,
                        dense_training_projection, conv2d_training_projection)

# 层 0 是外部输入；后续层保存 LIF 状态；最后一层用于分类损失。
conv, shape = conv2d_training_projection(0, 1, (2, 2, 2), 2, 1)
recurrent = dense_training_projection(1, 1, 8, 8)
readout = dense_training_projection(1, 2, 8, 2)
plan = lif_training_plan([8, 8, 2], projections=[conv, recurrent, readout],
                         backend='metal', beta=.8, learning_rate=.02)
trainer = NativeLIFTrainer(plan, runner='/path/to/b2-train')
# trainer.step(inputs, labels)；inputs 的 shape 为 batch × time × 8。
```

每个投影有六个字段：

| 字段 | 含义 |
| --- | --- |
| `source_layer` | 来源层编号，0 代表外部输入 |
| `target_layer` | 目标层编号，必须大于 0 |
| `sources` / `targets` | 等长整数数组，端点为对应层内索引；数组顺序就是前向边顺序 |
| `parameter_ids` | 与端点数组等长，指定每条边使用该投影参数组中的哪一个参数 |
| `parameter_count` | 该投影独立参数的数量 |

允许重复端点；重复 `parameter_ids` 使连接共享参数。编号只在同一投影内共享，
不同投影拥有不同的参数组。`weights`、`masks`、`trainable`、梯度及 optimizer 数组
均按投影排列，内部长度为 `parameter_count`。参数组数可以与非输入层数不同。
`beta` 和 `threshold` 仍按非输入层排列。

`dense_training_projection` 生成 target-major 边序，参数按 source-major 编号，
与 v1 每个目标累加来源的顺序一致。将 source_layer 与 target_layer 设成相同编号即可
建立循环；也可从输出层连接回隐藏层。

## 卷积约定

`conv2d_training_projection` 接收 CHW 输入形状，返回 `(projection, CHW_output_shape)`。
核参数按 OIHW 展平，执行普通不翻转核的 cross-correlation；每个空间位置使用相同核参数。
支持矩形核、整数 stride、对称整数 zero padding；没有 bias、dilation、grouped convolution
或 pooling。需要将输入图像按 CHW 展平为每个 tick 的脉冲幅度。

边顺序为输出通道、输出 y/x、输入通道、核 y/x。零填充区域不生成输入边。
kernel 的梯度是所有空间使用、时间和 batch 的贡献之和；损失仍使用 batch mean 和
输出时间平均，没有额外对空间梯度取平均。

## 循环前向与反向

每 tick 先更新全部神经元并确定硬阈值脉冲，再按投影及边顺序累加突触作用。
同 tick 的来源脉冲影响目标的膜电位，目标下一次阈值判定才能看到该作用；
因此循环没有同 tick 的代数方程求解，也没有额外可配置的突触延迟。
zero reset 在累加后执行；subtract 保持 v1 的浮点执行次序，先减阈值再累加输入。
任意投影图的事件顺序由本计划定义，不承诺与任意 Brian 连接创建/投递顺序逐位一致。

反向先汇总全部投影对参数和来源脉冲的 VJP，再计算全部神经元对上一 tick 的 VJP。
这一步屏障对跨层反馈是必要的：不能在其他投影还会增加该层脉冲梯度时提前完成该层反传。
zero reset 的 attached 导数使用记录的完整突触累加结果，涵盖所有循环和共享投影。
硬阶跃仍仅在反向使用声明的 fast-sigmoid surrogate。

完整 BPTT、TBPTT、非零 initial membrane、`initial='carry'`、SGD/Adam 和冻结参数
沿用 [v1 的定义](NATIVE_TRAINING.md)。冻结一个参数组只阻止 optimizer 更新，
仍允许梯度通过该投影传播。TBPTT 截断跨窗口 adjoint；当前仍保留整个获准序列的 tape。
跨请求 carry 不保留反向图。返回 `initial_gradients`，当前不返回外部输入序列的 VJP。

## 掩码、快照和资源

v2 的 mask 是参数掩码。一个共享参数被 mask=0 后，它关联的所有空间/图连接同时停用。
`update_mask` 裁剪/生长时重置该参数的 optimizer moments；不会创建额外的逐边 moments。
如果应用需要独立裁剪某一条共享边，需另行设计 edge mask；本版本不隐式解除参数共享。

同一个 Python trainer 绑定 schema/backend/sizes/投影图。原地改变 endpoint 或 parameter ID
后，执行和保存都会拒绝；创建新 trainer 才能选择新的拓扑。快照保存完整图计划和唯一
参数/optimizer 数组，新进程 restore 在回填状态前核对整个计划及 runner/Metal source hash。
没有拓扑迁移、共享关系迁移、v1→v2 快照迁移或跨精度迁移。runner 或 Metal 执行源码
变化后旧快照会被拒绝。
这个生命周期绑定属于 Python trainer；低层 CLI 每次接收一个完整且独立校验的请求。

保持 3..17 个层宽度、每层最多 65,536、最多 256 个投影；每个投影至少一条边。
helper 默认最多构造 1,000,000 条边，在生成索引列表之前检查上界，`max_edges` 可显式设置。
conv 的上界包含被 padding 排除的候选位置，所以会保守拒绝某些实际边数较少的超大配置。
原生预算额外计入投影三组索引、完整 pre-reset tape、GPU 打包索引及副本。
默认 64 MiB、native 上限 1 GiB、request JSON 上限 64 MiB；这些不是进程 RSS 上限。

Metal 使用独立的 graph kernel：每个 sample 一个 GPU lane，梯度在该 lane 内按固定边序
累加到唯一参数编号，再由 Rust 按 batch 顺序做 f64 合并和 optimizer 更新。
`b2_train_metal_v2` 对动态库 metadata ABI 显式版本化，旧动态库会报 symbol missing，
不会把投影图误当作 dense 网络。保持 f32 GPU/f64 host optimizer 的数值契约和无 CPU 回退。

## 运行与验收

隔离构建：

```bash
CARGO_TARGET_DIR="$PWD/brian2-rust/target-native-training" \
  cargo build --offline --release --manifest-path brian2-rust/Cargo.toml --bin b2-train
export B2_TRAIN_RUNNER="$PWD/brian2-rust/target-native-training/release/b2-train"
export PYTHONPATH="$PWD/brian2-rust/python"
export MPLCONFIGDIR=/tmp/atlas-training-mpl
.venv/bin/python brian2-rust/examples/native_graph_training.py \
  --backend cpu --steps 80 --checkpoint /tmp/graph-cpu.checkpoint
.venv/bin/python brian2-rust/examples/native_graph_training.py \
  --backend metal --steps 80 --checkpoint /tmp/graph-metal.checkpoint
.venv/bin/python brian2-rust/examples/native_graph_training.py \
  --backend metal --steps 0 --checkpoint /tmp/graph-metal.checkpoint --resume
B2_TEST_GPU=1 .venv/bin/python -m pytest -q \
  brian2-rust/tests/test_native_training.py brian2-rust/tests/test_native_training_graph.py
```

交付示例同时训练共享卷积核、循环权重和读出。CPU/Metal 的 80 步损失均从约 1.223445
降到 0.061968；四个从训练输入中额外去除部分时间脉冲和空间位置的 held-out 样本均分类正确。
卷积的 16 条连接只有 4 个独立参数；示例的循环参数组保留 64 个候选槽、激活 8 条自连接。
另有含非对角循环和输出→隐藏反馈的梯度测试，使用非零初态保证反馈参数梯度确实非零。
示例是功能验证，不是公开数据集准确率或 GPU 加速比结论。

测试包含独立局部平滑 surrogate 有限差分（四种 reset/detach × 完整/TBPTT）、
初始状态 VJP、独立空间卷积计算、共享梯度等于未共享逐边梯度之和、一次 SGD 更新、
冻结/裁剪/生长、CPU/Metal 新进程恢复、v1/v2 dense 一致性及失败边界。
完整日志、运行时/源码指纹及例子输出见
[第二阶段验收](mpi-evidence/training-graph-20261002/README.md)。

后续已增加 CUDA BPTT、CPU MPI 目标分区 BPTT，以及 v3 标量更新方程、可训练 neuron
系数/阈值，见 [第三阶段](NATIVE_TRAINING_V3.md)。任意 Brian 方程自动转换、refractory、
可学习延迟、pooling 和 GPU MPI BPTT 仍未实现。显式训练计划不会自动转换平台模型或改变其 engine 配置。

最新多状态与 Metal MPI 能力见 [v4 / GPU MPI](NATIVE_TRAINING_V4.md)。
