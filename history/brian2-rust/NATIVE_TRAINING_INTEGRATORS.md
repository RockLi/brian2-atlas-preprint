# Brian RK2/RK4 与一般单状态复位

后续扩展：[固定时长 refractory](NATIVE_TRAINING_REFRACTORY.md)，以及
[RK2/RK4 与 refractory 组合](NATIVE_TRAINING_RK_REFRACTORY.md)；
本页 RK 验收仍对应本页记录的源码版本。

2026-10-03。`lower_brian_training` 扩展内置显式积分方法和单状态复位，复用现有
原生 v4 前向/VJP、CPU/MPI 和 Metal/CUDA 算子，不引入 Python 训练回退。

## 使用与语义

Brian `NeuronGroup(..., method='rk2')` 和 `method='rk4'` 现在可以直接转换。
不同层可使用不同的内置 `euler`、`rk2`、`rk4`，仍须相同 dt 和默认调度。
可训练共享参数和全部初始状态的导数穿过所有 RK 积分步骤，支持 full/TBPTT。
转换器调用 Brian 内置积分器生成阶段表达式，再把顺序赋值编译为不可变表达式图；
每个阶段读取当时的状态快照，最终状态写回仍保持同时更新语义。

RK 编译复用重复阶段与表达式的 SSA 节点；每个输出程序独立，最多 128 节点。
原生反向会累加共享节点的所有使用位置，算术次序不作代数重排。
现有三状态 RK4 测试各输出为 99/90/87 节点。更大或更复杂模型可能仍超出预算，
会明确拒绝，不静默降阶。被用户替换的积分器注册项、自动方法列表、自定义方法选项
和其他未支持的方法继续拒绝。

单状态模型现在也支持 v4 的有界顺序复位，例如：

```python
hidden = NeuronGroup(
    2, 'dv/dt=-v/tau : 1\ntau : second (constant, shared)\n'
       'theta : 1 (constant, shared)',
    threshold='v>theta', reset='v=.4*v\nv+=.1*theta', method='rk4',
)
# 构造其余层、Synapses、Network 后，照常调用 lower_brian_training。
```

同一网络可混合零复位、减阈值复位和一般复位。一般复位或混合复位自动选择
单状态 v4；多状态继续使用 v4。原有统一标准复位的 scalar Euler 网络保留 v3。
只能写微分状态；复位写入系数、控制流和自定义事件仍拒绝。
`bundle.provenance['integrators']` 记录每层所选方法。
v4 继续使用完整 `bundle.initial_state`、`final_state` 和 `initial_state_gradients`。

## 验证

新增 `tests/test_training_brian_integrators.py`：

- 与 Brian 独立运行时逐脉冲比较，并比较最终状态；覆盖 RK2/RK4、单/多状态及 SI volt 单位。
- 一般顺序复位、混合复位，纯 Euler 和不同层混合积分方法的前向对照。
- 独立 NumPy 教科书 RK 算法和局部 surrogate 线性化，通过数值差分检查全部权重、
  神经元参数和初态梯度；覆盖 detach 开关和 full/TBPTT。
- CPU MPI、Metal、Metal MPI 与串行 CPU 比较两次更新、全部状态/VJP，并检查 checkpoint 恢复。
- CUDA 与 CUDA MPI 的 RK2/RK4 对应 4 项测试已在后续单 L4 作业中全部通过。

本轮验收日志和源码指纹见 [本轮证据](mpi-evidence/training-integrators-20261003/README.md)。
新增 RK 前端的 CUDA 端到端实测见后续 [L4 验收](mpi-evidence/training-refractory-20261003/README.md)，
其中本套件 27 passed、4 skipped（Metal），包含 4 项 CUDA；此前旧版 L4 证据仍单独保留。

## 剩余边界

这仍是确定性 Brian 子集：随机微分方程、条件式 refractory、任意事件/调度、异质系数、
动态突触、可学习延迟、自定义函数和任意控制流尚未覆盖。
阈值仍须 `v >` 正标量或单个共享参数，突触仍只写 `v_post`。
同一主机共享一张 GPU 的 MPI 验证不能替代跨主机、多物理 GPU 验证。
跨主机 GPU 验证仍需明确的测试资源、相同二进制/动态库及真实 rank/GPU 身份记录；
后续已使用获授权的单张 L4 完成测试，未连接既有生产主机。
