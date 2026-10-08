# v4 GPU / CUDA MPI 验收记录

2026-10-02。本轮源码接入 CUDA MPI、Metal/CUDA 多状态 BPTT，以及多状态 GPU MPI。
当前已完成本机 Metal / CPU / MPI 验证，以及修复 MPI 镜像后的单 L4 验证。
最新 L4 结果：**100 passed, 64 skipped, 0 failed**，单机 CUDA MPI 与多状态验收通过。
最初失败作业的证据保留在下文，不能与最新结果相加。

本机：**192 passed, 64 skipped, 0 failed**，134.49 秒。
64 项均因本机没有 NVIDIA CUDA 硬件而跳过。该结果包含前一轮 169 项，不能相加。

- [本机完整日志](local.log)
- [本机 JUnit](local.xml)
- [源代码、二进制指纹与分组计数](verification.json)
- [实现与 API](../../NATIVE_TRAINING_V4_GPU.md)
- [L4 完整报告](cuda-l4/report.json)、[JUnit](cuda-l4/junit.xml)、[计数与指纹](cuda-l4/verification.json)
- [修复后 L4 完整报告](cuda-l4-mpich5/report.json)、[JUnit](cuda-l4-mpich5/junit.xml)、[验收汇总](cuda-l4-mpich5/verification.json)
- [有界 L4 入口](../../examples/modal_native_training_v4.py)

新增实际 Metal 覆盖 23 项：2/3 状态混合层的全部初态和参数梯度、2 次更新、full/TBPTT、
两种复位与 detach 开关、8 rank 空 owner、freeze/mask、新 Python 进程精确恢复、
16 状态上限与非线性、中间域错误不被后续有限状态掩盖、Brian 多状态端到端。
原有 CPU 独立数值差分、Brian 运行时对照以及 Metal v2/v3 MPI 测试继续通过。

本机 native binary SHA-256：
`a9362eedcf6c7520cc8d4d33679d74116ad8582aa39e3730247eabb8bbf60260`。

## 首次 L4 结果（失败记录）

[Modal 作业](https://modal.com/apps/insfocus/main/ap-RiD0zHaIztfN2DF40kirEU)
已结束：**58 passed, 42 failed, 64 skipped**。其中实际单进程 CUDA **33 项通过**，
包括多状态 v4 **11 项**；其他通过项为 CPU 测试。CUDA 12.8.93，NVIDIA L4。
作业总耗时 533.20 秒（包含 CPU 镜像构建），测试耗时 403.36 秒。
只运行一次，1 张 L4、函数上限 900 秒、无重试；退出后无持续 GPU 作业。

42 项失败全部为 `MPI training plan/launch rank mismatch`，在训练开始前被拒绝。
MPI 启动后的进程均表现为独立 rank 0，未形成预期通信域，故包括 CPU MPI 在内均失败。
v4 CUDA 新进程 checkpoint 恢复测试要求 MPI，仍未通过；不能用旧版 CUDA 恢复测试替代。
本次 CUDA native binary SHA-256：
`4f48f72fa9d15917f37442f5504bc31cfb9ac68d8ef292a062ad9b8080f31150`。
上传源文件在作业结束时与本地一致；后续镜像配置修改另记于 `verification.json`。

## 失败后修复与待验收部分

原镜像使用 Ubuntu 24.04 `mpich 4.2.0-5build3`，现象与
[MPICH 上游记录的 PMIx / Hydra 打包问题](https://github.com/pmodels/mpich/issues/7064)
一致。尚未通过独立 Linux 复现确认完整因果链。
入口现改用固定 `mpich==5.0.1` 发行包，并在 CPU 镜像构建阶段运行
[`verify_native_training_mpi_bootstrap.py`](../../tools/verify_native_training_mpi_bootstrap.py)：
用实际 MPI shim 校验 2/4 rank 身份、通信域和归约，再对比原生 MPI 与串行训练结果。
预检失败会中止镜像构建，不进入 GPU 函数。

该预检先在本机 MPICH 5.0.1 下通过，见 [本机预检报告](mpi-preflight-macos.json)。
随后用户明确授权新的单次有界验证；新 Linux 镜像与 L4 结果见下一节。
两次作业之间只改变 MPI 镜像配置并添加预检，运行时代码和训练测试未改动。

本地 Ubuntu 容器复现因磁盘仅余约 142 MiB、依赖安装 I/O 错误和 Docker 断连中止，
尚未执行 MPI 探针，见 [日志](mpi-linux-repro.log)。容器 `113b19b6be34` 原以 `--rm`
启动，但断连后无法确认清理；未重启 Docker 或删除用户文件。

## 修复后 L4 结果（当前验收）

[Modal 作业](https://modal.com/apps/insfocus/main/ap-tmicGNrxRJi97ejQe2LE1T)
已结束：**100 passed, 64 skipped, 0 failed**。64 项跳过均为 Metal 用例。
通过项包括 **64 项实际 CUDA 测试（33 项单进程、31 项 MPI）**和 36 项 CPU 测试。
其中 GPU MPI 套件 19 项、多状态 GPU 套件 23 项全部通过。

覆盖 2/4 rank 目标分区梯度和 optimizer 对照、8 rank 空 owner、full/TBPTT、
两种 reset、detach 开关、多状态全部初态梯度、freeze/mask、故障原子性，
以及单状态和多状态 MPI checkpoint 的新 Python 进程精确恢复、Brian 多状态端到端。
[Linux CPU 预检](cuda-l4-mpich5/mpi-preflight.json)也确认 2/4 rank 身份、通信域、
归约和原生训练与串行一致。验证使用配套 MPICH 5.0.1，原生二进制与首次作业相同。

测试耗时 **622.72 秒**，包含镜像准备的总耗时 **682.78 秒**；1 张 L4、最多一个容器、
函数上限 900 秒、测试超时 720 秒、无自动重试。容器已结束，上传的 131 个文件指纹
与完成时本地内容一致。本次验收只证明同一主机上多个 MPI rank 共享一张 L4 的正确性；
跨主机、多物理 GPU、性能和内存扩展性、任意 Brian 语义仍不在已验收范围内。

本机多个 MPI rank 共享一张 Metal GPU。尚无跨主机 GPU、多物理 GPU、性能加速比
或内存扩展性验收。生产执行器和远端服务未替换。
