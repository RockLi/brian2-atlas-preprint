# 原生均匀随机数与混合随机定时代码

原生 v4/v5 SSA 现在接受 `uniform_noise`，Python 显式绑定为
`brian2_rust.training_equations.UniformNoise(stream)`。同步 `run_regularly` 前端
把 `rand()` 转成均匀调用点，`randn()` 继续使用正态调用点；两类调用共同占用该
runner 的最多 16 个流，标量和向量段有各自的调用点编号。

## 数值和执行契约

- 均匀序列使用独立于正态序列的 domain magic `0x4232554e49463031`，依次混合
  seed、sequence、batch、domain/layer、entity/neuron、tick、stream；没有可变全局
  draw order 或 MPI rank。取最终计数哈希的高 53 位除以 2^53，得到 `[0,1)`。
- 旧正态算法及其计数地址保持不变。不能通过正态近似生成均匀值，也不能让两种
  分布在一个执行上下文中共用同一个 stream；原生验证会拒绝这类冲突。
- CPU 使用 float64 均匀值。GPU 主机原生层预生成值，再以原有噪声表接口上传；
  转成 float32 时若舍入为 1，则降到最大的、严格小于 1 的 float32，保持半开区间。
  shader 读取已有噪声算子，因此 GPU ABI 仍为 v5r7，无需伪造新的设备数学实现。
- 随机值自身不求导；乘性幅度、状态与参数沿固定抽样路径反传。独立 full/TBPTT
  数值差分覆盖更新和 reset，以及混合 Gaussian/uniform 的路径导数。
- carry/store/restore 保留序列及 tick；MPI 分区变化不改变地址。重新执行梯度不会
  重新抽取不同样本。声明复用的 `UniformNoise(stream)` 表示有意复用同一次抽样；
  前端每个不同的 `rand()` 调用点则分配不同流。
- 突触向量临时量纳入结构 mask 的 owner/restart 描述，保证定时随机代码在剪枝、
  regrowth、Adam 和 checkpoint 后仍能继续。共享标量更新仍按时钟执行。

## 接口与验收

```python
from brian2_rust.training_equations import UniformNoise, compile_training_equation
program = compile_training_equation(
    '.8*v + gain*u', states=['v'],
    parameters={'gain': (0, 0), 'u': UniformNoise(0)},
)
# 原生计划仍须声明 noise_streams 和 clock。
# Brian 同步前端则可直接使用：
# group.run_regularly('u=rand(); v += .03*(v+1)*u', when='end')
```

本机的新版 runner 位于 `mpi-evidence/training-uniform-final-20261004/b2-train`，
通过 `runner=` 或 `B2_TRAIN_RUNNER` 显式选择。它是本机编译产物，跨平台应使用
相同源代码重新编译；旧 v5r7 验收 runner 保留在原路径，不能用旧二进制测试新算子。

正式结果见 [最终验收](mpi-evidence/training-uniform-final-20261004/README.md)：
207 passed / 95 NVIDIA-only skipped / 0 failed，302 个身份、283 个冻结文件；
父进程 session 82114 exit 0，另有 2 项 Rust 单元测试通过。
新增模块测试原生大批量分布／逐地址结果、混合抽样、真实 Cython、冷／暖快照、
独立权重／初态 full/TBPTT VJP、CPU／Metal／2/8 MPI、检查点和结构掩码迁移；
Rust 单元测试检查极值转换和独立分布。旧正态及动态 GPU 模块完整回归。
中间诊断保留于 `mpi-evidence/training-uniform-20261004/`，不并入最终统计。

## 仍有缺口

这完成了原生显式算子及同步定时代码的均匀随机数支持。一般 Brian 方程、threshold、
reset、pre/post 路径中的任意显式随机调用仍需前端处理，尤其同一 tick 内同一边的
多次事件投递不能复用一次抽样。异步执行、int64/unsigned/位运算、外部回调及
Synapses-to-Synapses 端点亦尚未全部覆盖。新版 NVIDIA 实机验证仍待完成，跨主机／
多物理 GPU 继续暂缓。本阶段只使用本地资源。


### 随机突触调用更新（2026-10-04）

标准 pre/post 路径现已接入显式 rand/randn，并通过稳定的发射身份区分同一 tick
同一条边的多个到达事件。连续突触 Euler 与 summed 的显式随机调用也已接入。
执行契约、独立参照和最终验收入口见 [随机突触事件](NATIVE_TRAINING_EVENT_NOISE.md)。
一般神经元 equation/threshold/reset 的任意显式随机调用仍有前端缺口。
