# 神经元显式随机调用

`lower_brian_dynamic_training` 支持神经元微分方程、threshold 和 reset 中的标准
`rand()`／`randn()`。已有命名 `xi` SDE 流保持原来的位置；显式调用点使用随后
的独立流，每层合计最多 16 个。`provenance['neuron_draws']` 记录 update、threshold、
reset 的调用类型和流编号，`noise_names` 记录完整流布局。

例如，以下神经元定义可以由动态前端转换：

```python
group = NeuronGroup(
    2, 'dv/dt=(-v + .1*rand() + .01*randn())/ms : 1',
    threshold='v > .5 + .01*rand()',
    reset='u = rand(); v = .1*u', method='euler', dt=.2*ms,
)
```

## 执行与梯度

- 每个显式调用点以 seed、sequence、batch、layer、neuron、tick、stream 定址。
  reset 中 `u=rand()` 后重复使用 `u` 只抽样一次。RK 更新依照 Brian 生成代码
  中的调用点分配流；不同 stage 的调用不会被折叠。
- 更新、阈值 margin、reset 保持现有 Brian 调度；refractory 的状态写入限制和
  spike 门控继续生效。阈值随机 margin 必须由无 mask、无 trigger 的动作准备，
  原生校验仍拒绝条件准备的阈值。
- 随机值自身不求导；固定噪声路径上的状态、可训练幅度及 spike surrogate
  参与 full BPTT／TBPTT。检查点和分段 carry 使用同一 sequence 与逻辑 tick。
- Brian 的语句分析器物化 reset 子表达式和临时量。浮点、Boolean、int32 局部
  计算保持类型；int32 运算采用环绕语义。v4 的物理状态输出仍必须是浮点。
- v4 GPU 需要新版 `v4r7` ABI，通过已有动态表达式内核执行 typed locals；
  v5 GPU ABI 保持 `v5r7`。旧 GPU 库不能静默执行未知的 v4 指令。

不保证和 Brian 全局随机生成器在相同 seed 下逐样本一致。测试将独立计算的
固定样本注入实际 Brian Cython 随机缓冲区，比较调用次数、spike 和物理状态。

## Brian 限制与当前边界

转换前直接检查原始方程，即使 SymPy 替换缓存已经建立，也拒绝 Brian 不允许
的同一微分表达式内重复调用同一种 stateful 函数，以及缺少 `constant over dt`
标记的随机子表达式，避免代数展开错误地合并随机样本。

`constant over dt` 需要单独的 SubexpressionUpdater 调度与持久状态，目前仍未
接入。直接在 refractory 表达式内调用随机函数也尚未接入；本阶段 refractory
测试使用确定性时长。异步连续积分、外部 Python 回调、int64／unsigned／位运算和
Synapses-to-Synapses 端点也仍有缺口。跨主机／多物理 GPU 验证按用户要求暂缓。
本轮 CUDA 仅检查共享源转换，尚未进行 NVIDIA 实机验收。

## 验证入口

新增 `tests/test_training_neuron_noise.py` 覆盖 Euler、混合 xi、RK2、RK4、
冷／暖快照、refractory、独立权重和初态有限差分、整数溢出 reset、检查点及
CPU／Metal 本机 2/8 MPI。静态 v4 前端另与独立公式和动态 v5 梯度交叉比较。

正式验收：**575 passed / 164 NVIDIA-only skipped / 0 failed**，739 身份、12 完整
模块、302 冻结文件；新增模块为 51 passed／23 NVIDIA-only skipped。另有 4 项
独立 v4 边界检查通过，不并入 pytest 计数。父进程 session 24468 exit 0 已观察，
verify.py 通过。完整记录见 [验收报告](mpi-evidence/training-neuron-noise-final-r4-20261004/README.md)。

新版 runner 位于 `mpi-evidence/training-neuron-noise-final-r4-20261004/b2-train`，
请用 runner= 或 B2_TRAIN_RUNNER 明确选择；旧 runner 保留。测试保留 Python
默认编译选项并追加 -g0，避免改变 Cython int32 溢出语义。
