# RK2/RK4 与固定时长 refractory

后续新增 [异质神经元系数和参数](NATIVE_TRAINING_HETEROGENEOUS.md)；本页 v4r3 的硬件证据仍单独保留。

2026-10-03，接续 [Euler refractory](NATIVE_TRAINING_REFRACTORY.md) 与
[RK 前端](NATIVE_TRAINING_INTEGRATORS.md)。现在固定非负时间 Quantity 的 refractory
可与内置 `euler`、`rk2`、`rk4` 组合，也允许不同层使用不同积分方法。

## 语义与实现

对带 `(unless refractory)` 的方程，Brian 的 RK 中间阶段也包含
`int(not_refractory)`。转换器保留内置 RK 的阶段表达式，把该项降低为
`refractory_active` SSA 节点：读取本时间步开始时的计数器，为零时返回 1，否则返回 0。
整个 RK 步使用同一个离散开关；所有阶段都对被保持状态的导数应用该开关。
只在最终赋值时恢复被保持状态是不正确的，因为中间值已可能影响其他未保持的状态。

该节点在 CPU、Metal 和 CUDA 共用 shader 中实现，VJP 为零。物理状态和参数仍按
实际 RK 阶段传播梯度，非活动时被保持状态具有恒等 VJP。发放、同一步突触条件写入、
复位、计时递减、full/TBPTT 和恢复的语义沿用固定时长 refractory 实现。

原生校验只允许在已声明 refractory 的层的物理状态更新中使用活动节点，索引必须
指向该层最后一个计时状态；复位程序、无 refractory 的层、任意其他状态索引均拒绝。
直接读取计时器参与可微算术仍拒绝。条件式时长、可学习时长和 legacy timing 仍拒绝。

GPU SSA 新增 opcode 16；动态库 ABI 更新为 `b2_train_metal_v4r3` /
`b2_train_cuda_v4r3`，避免旧库将新 opcode 当作其他值。旧二进制和检查点不能无条件
混用；既有指纹校验继续生效。此前 v4r2 的 L4 成功报告不覆盖本次新增 opcode。

## 验证

新增 `tests/test_training_rk_refractory.py`，复用独立 Brian 前向运行和独立 NumPy
教科书 RK 算法。覆盖四种保持组合、零/整步/非整步时长、SI 单位、已运行网络快照、
混合积分方法、detach、full/TBPTT、全部参数及物理初态梯度。另有 CPU/Metal/CUDA
及 2/8 MPI ranks 的连续训练和新 Python 进程恢复，重点覆盖只保持一个耦合状态。
反例测试明确证明去掉中间阶段开关会改变未保持状态，非法节点测试检查失败不提交状态。

本轮执行结果及范围见 [验收记录](mpi-evidence/training-rk-refractory-20261003/README.md)。
本地完整回归为 411 passed、86 skipped（CUDA）。随后获授权的单 L4 作业完成
238 passed、45 skipped（Metal）、0 failed；实际 45 项 CUDA（26 项 MPI），包括
本套件的 12 项 CUDA。资源已停止，无遗留运行容器。详细分项见验收记录。

后续单 L4 回归入口：

```sh
.venv/bin/python brian2-rust/examples/modal_native_training_v4.py \
  --suite rk-refractory --output /absolute/new/evidence/directory
```

保留 CPU MPI 镜像预检、单张 L4、函数 900 秒、pytest 720 秒和跨容器单次执行锁。
本次单次云资源授权已执行完毕；未来重跑需要新的资源授权。
跨主机与多个物理 GPU 的方案另见 [资源与验收方案](NATIVE_TRAINING_CROSSHOST_GPU.md)。
随机方程、动态突触、异质系数、自定义事件等任意 Brian 语义仍不在支持范围内。
