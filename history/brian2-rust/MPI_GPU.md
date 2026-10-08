# MPI 中混用 CPU 与 GPU

2026-09-11，`codex/mpi-cpu`。

原有 MPI 实现只有 CPU Rust AOT，单机 CUDA/Metal 执行器没有接入 MPI。
现在可以在同一个 MPI communicator 内，显式为每个 rank 选择 CPU、Metal
或 CUDA。GPU rank 将神经元 `state_update` 节点放到真实 GPU 执行；阈值、
重置、突触运算、延迟队列、监视器及 MPI 通信仍由该 rank 的 CPU 执行。
这是神经元更新卸载版本，不是全网络 GPU 驻留版本。

2026-10-02：同一 rank/设备/数值政策下增加分段 STDP、pending 恢复和边界 checkpoint，
详见 [MPI_TRAINING.md](MPI_TRAINING.md)。单机原生 Metal surrogate BPTT 的不同执行范围
见 [NATIVE_TRAINING.md](NATIVE_TRAINING.md)；MPI GPU 更新卸载不包含分布式 BPTT。

## 使用

Python 程序只启动一次，Device 自己调用 `mpiexec`：

```python
import brian2 as b
import brian2_rust

b.set_device("rust_standalone", engine="mpi", ranks=2,
             rank_backends=["cpu", "metal"],
             numeric_mode="mixed-f32", directory="mpi-mixed")
# 正常创建 NeuronGroup / Synapses / Monitor，然后 Network.run(...)
```

Linux NVIDIA 示例：`rank_backends=["cpu", "cuda:0", "cuda:1"]`、`ranks=3`。
`cuda` 等价于 `cuda:0`。设备 ordinal 是各 rank 所在节点、进程可见的 CUDA
编号；不同节点可以都使用 `cuda:0`。没有自动设备发现、负载均衡或隐式 CPU
回退。Metal 使用系统默认 GPU，只接受 `metal`。

可直接运行的循环网络示例：

```bash
python brian2-rust/examples/mpi_heterogeneous.py \
  --rank-backends cpu,metal --directory /tmp/mpi-mixed-example
```

导出模型的底层 API 同样支持设备选择：

```python
plan = brian2_rust.build_execution_plan(
    model, backend="mpi", ranks=2,
    rank_backends=["cpu", "cuda:0"], numeric_mode="mixed-f32")
brian2_rust.write_mpi_project(model, "mpi-project", ranks=2, plan=plan)
brian2_rust.compile_mpi_project("mpi-project")
report = brian2_rust.run_mpi_project("mpi-project", "mpi-result")
```

`write_mpi_project` 的 `population_owners` 可继续将整个 population 指派给
某个 rank；默认仍按神经元数量连续等分。设备不同不改变入边归属或全局索引。

## 精度与运行契约

- GPU 更新使用现有 GPU 表达式生成器的 float32 运算，输入从该 rank 的
  f64 状态暂存为 f32，写回时扩展为 f64；CPU rank 保持原来的 f64 运算。
  必须显式指定 `mixed-f32`。改变设备分配可能改变舍入及后续脉冲。
- 所有状态更新遵守 canonical 节点顺序。GPU 完成、状态回传之后才执行
  阈值及 MPI 脉冲交换。现有重复边次序、跨 rank 延迟、固定 refractory
  和全局神经元/边 RNG 身份保留。
- CPU-only 默认计划、计划 hash 和生成路径保留原样，不需要 GPU 工具链。
- `execution-plan.json` 记录设备顺序和数值契约；manifest 包含 GPU 源码
  hash；`build.json` 包含 GPU 动态库 hash；启动前验证这些文件。
- `mpi-runtime.json` 中 `rank_backends`、`gpu_scope`、`numeric_profile` 和
  `rank_gpu_dispatches` 可检查实际卸载范围及各 rank 的成功 GPU 调用次数。
  空分片不会发起零长度 GPU dispatch。
- CPU rank 不加载 GPU 动态库。GPU 初始化/执行错误进入现有 `MPI_Abort`
  路径，不会静默回退；其他 rank 不会一直等待缺失的 collective。

## 环境与当前边界

需要原有 MPI/Rust 工具链。Metal 项目还需 macOS、Metal 和 clang；CUDA
项目需 Linux NVIDIA 驱动及 `nvcc`，此路径直接使用 CUDA Runtime，不需要
CuPy。编译生成 GPU 动态库，CPU/GPU 节点须使用兼容的 OS/CPU ABI、MPI
实现，并在同一路径访问项目文件；本轮未验证跨主机运行。

此版本继承 [CPU MPI 支持范围](MPI.md)。此外，GPU 状态更新暂不支持
TimedArray、scalar RNG、直接写 refractory 标记；不支持的表达式在项目
生成前拒绝。向量 counter RNG 已接入。原有 source compaction/prebuild
选项暂不能与 GPU 卸载同时开启。

暂存的状态/参数载荷每 rank 最多 512 MiB，另有每神经元 4 字节错误标记；
此上限不包含原 f64 状态及 GPU 副本。Metal 每节点调用仍会分配/复制设备
缓冲；CUDA 复用设备缓冲但仍按更新节点传输。首次使用节点会初始化 kernel，
这部分目前计入 simulation 时间。尚无混合模式加速比结论。

## 验证

本机 Apple Silicon 真实 MPI + Metal 测试：

```bash
B2_TEST_MPI=1 B2_TEST_GPU=1 PYTHONPATH=brian2-rust/python \
  python -m pytest -q brian2-rust/tests/test_mpi_gpu.py
```

Linux NVIDIA 上使用同一测试集：

```bash
B2_TEST_MPI=1 B2_TEST_GPU=1 B2_TEST_MPI_GPU=cuda:0 \
  PYTHONPATH=brian2-rust/python python -m pytest -q brian2-rust/tests/test_mpi_gpu.py
```

测试覆盖设备配置校验及绑定、1/2/3/4 ranks、CPU/GPU 顺序交换、整 population
指派、空分片、双向循环网络、重复突触、不同延迟、refractory、监视器、
负零保留、混合精度的独立预期值、全局 RNG 索引、Brian2 Device 入口及
GPU 动态库缺失后的进程组退出。二进制可精确表示的循环网络与独立 Rust
reference 逐位比较；非精确 f32 网络分别验证 CPU 和 GPU 的数值契约。
CUDA 已实现源码生成和动态库执行路径；本机没有 NVIDIA GPU/nvcc，尚未做
CUDA 编译或硬件验收，不能把 Metal 的实测结果视作 CUDA 验收。

本轮结果：新增混合模式及完整性测试 25 项通过；CPU MPI/计时回归 70 项
通过；计划/Device/压缩相关回归 23 项通过、12 项环境门跳过。合计 118 项
通过。交付示例实际记录 82 个脉冲，两个 rank 的 GPU dispatch 分别为
`[0, 32]`。测试 XML、运行报告及交付源码 hash 见
[验证记录](mpi-evidence/heterogeneous-20260911/verification.json)。
