# 神经元显式随机调用验收

正式结果：**575 passed / 164 NVIDIA-only skipped / 0 failed**。739 个不重复测试身份，
12 个完整模块，302 个冻结文件。父进程 session 24468 的 exit 0 已观察；run.py 末尾
verify.py 已通过。源码、runner、分片终态、JUnit 身份及汇总逐一核对，所有跳过
均为 CUDA 硬件用例。没有云调用；跨主机／多物理 GPU 按用户要求暂缓。

| 模块 | 通过 | NVIDIA 跳过 |
| --- | ---: | ---: |
| test_training_neuron_noise.py | 51 | 23 |
| test_training_brian.py | 23 | 0 |
| test_training_brian_multistate.py | 7 | 0 |
| test_training_brian_integrators.py | 27 | 4 |
| test_training_stochastic.py | 69 | 9 |
| test_training_dynamic_thresholds.py | 53 | 9 |
| test_training_rk_refractory.py | 125 | 12 |
| test_training_event_noise.py | 31 | 12 |
| test_training_uniform.py | 36 | 17 |
| test_training_linked.py | 63 | 15 |
| test_training_dynamic_gpu.py | 45 | 45 |
| test_training_integer_ir.py | 45 | 18 |

## 实现与证据

神经元微分方程、threshold、reset 的 rand/randn 调用点流，保留命名 xi 地址；
reset 临时量及子表达式重用；v4 精确 int32/Boolean 局部表达式和 v4r7 GPU ABI。
新增模块含实际 Brian Cython 固定样本注入、Euler／混合 xi／RK2／RK4、冷／暖
快照、refractory、独立 full/TBPTT 权重及浮点初态差分、批次、检查点、实际
Metal 与本机 CPU/Metal 2/8 MPI。详见 ../../NATIVE_TRAINING_NEURON_NOISE.md。

另有 check_v4_boundary.py 的 4 项独立检查全部通过，session 72502 exit 0：
拒绝 v4 整数物理输出／integer_state，CPU 与实际 Metal 验证 lazy 未选中除零分支、
-1 整数载荷及零伴随。结果和脚本摘要见 v4-boundary*.json，不并入上表 pytest 计数。
CUDA 只进行共享源转换检查（prototype、definition 和 6 个 kernels），未运行 NVIDIA。

## 复现与环境

本机 runner：`b2-train`，SHA256 `46be8add60b0b5671d34819dff83a40347b774a7683edca4b7d04074190b5a92`。
用 runner= 或 B2_TRAIN_RUNNER 明确选择此文件；旧二进制没有覆盖。
`runtime.json`、`source-hashes.json`、`environment.json` 保存对应信息。
`run.py` 为不可覆盖已有目录的首次运行入口；`verify.py` 可复核已保存证据。
新模块每 4 项一进程，其余每 32 项；Cython 临时缓存随分片释放。
CFLAGS/CXXFLAGS 从 Python sysconfig 默认配置追加 -g0，保留 -fno-strict-overflow。

## 未纳入验收的运行

- training-neuron-noise-final-20261004：父进程 exit 120，未完成首片，原因未确认。
- final-r2：第一片通过后发生 errno 28 磁盘写满，父进程 exit 1。
- final-r3：仅设 CFLAGS=-g0 替换了 Python 默认溢出语义选项，Cython 整数 reset
  与原生对照不一致，父进程 exit 1。保留默认选项后的定向复测为 2 passed／1
  NVIDIA skip，session 50332 exit 0；随后从头执行本次 r4。

这些运行及 pilot 的计数均不累计到正式结果。清理了本轮已结束的 Cargo 缓存和
uv 认定未使用的包缓存（uv 报告移除 831.6 MiB），没有删除源码、环境或验收记录。

## 下一阶段

见 NEXT.md。constant-over-dt 子表达式缓存更新和直接随机 refractory 表达式仍未
接入，异步连续积分、Python 回调、int64/unsigned/位运算、Synapses-to-Synapses
端点及新版 NVIDIA 实测仍有缺口。整体目标保持 active。
