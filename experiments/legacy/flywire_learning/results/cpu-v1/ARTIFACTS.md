# CPU v1 验收与复现索引

完成日期：2026-09-11。独立分支 `codex/flywire-learning`，基线
`0bc46a102290846836c5d1c76e19e6c027c34d2c`。工作目录
`/private/tmp/brian2-flywire-learning`。未合并或修改 flywire-vision。
最终科学结论及全部 25 行结果见 [RESULTS.md](RESULTS.md)。

交付前的身份、登记及图表检查见 [delivery_audit.json](delivery_audit.json)；
[artifact_manifest.json](artifact_manifest.json) 保存本次小型产物与相关源码的 SHA-256
（不包含清单自身，路径相对仓库根目录）。

## 逐项目标与证据

| 要求 | 已完成内容 | 证据 |
| --- | --- | --- |
| 1. 数据及文献依据 | 精确双侧 MBON01 注释，合并 γ5/β′2a 输入；教学映射和学习规则明确标为人工假设 | [冻结协议](../../PROTOCOL.md)、[研究依据](../../RESEARCH.md)、[连接组审计](../connectome_audit.json) |
| 2. 局部可塑性 | delivered KC spike 资格迹、教学门控、有界 gain、独立学习开关；固定拓扑和符号 | [模型](../../spiking.py)、[真实网络入口](../../experiment.py)、[测试禁学审计](completion_audit.json) |
| 3. CPU 正确性 | 独立逐 tick 参考、Brian NumPy/C++、Rust reference/AOT 1/4 线程 | [小回路验证](../spiking_validation.json)、[续跑验证](../continuation_validation.json)、[全脑误差](correctness.json) |
| 4. 真实子图及全脑 | 2,558 节点子图；139,255 节点全脑；15,089,423 静态边 + 2,560 可塑边逆向还原原图，无重复传递 | [分区清单](../prepared_manifest.json)、[子图验证](../real_subgraph_validation.json)、[全脑误差](correctness.json) |
| 5. 固定对照矩阵 | 5 条件 × seeds 11/23/47/83/131 = 25/25；900 次试验读出从完整脉冲重算；所有测试/休息教学与学习关闭 | [最终汇总](final_report.json)、[逐条件审计](completion_audit.json)、[输入身份](input_identity_audit.json) |
| 6. 恢复、重放及成本 | 33/33 受监督作业退出 0；全脑连续/分段、fresh-process restore、原生结果重放精确相同；保留配置、源码、资源与数据身份 | [执行登记](execution_registry.json)、[checkpoint 审计](checkpoint_audit.json)、[CPU 环境](../cpu_environment.json)、[运行身份](study_identity.json) |

全脑跨后端门覆盖 1.1 生物秒；完整连续/分段和保存恢复覆盖 **seed 11 paired 的
19.1 生物秒**，没有声称 25 个条件逐一恢复。比较包含所有神经元的最终 v/ge/gi、
不应期状态、完整脉冲、全部 2,560 gain/资格迹及 6 个预选神经元的轨迹。
没有记录所有神经元每个 tick 的电压。冻结分区轨迹最大差 2.22e-16，其他全脑
门保存数组误差为 0。小回路 C++ 不含中间突触样本；历史字段 false 表示该项未采样，
由 NumPy/Rust 完成逐段禁学检查，详见最终报告。

## 原始文件保留位置

本机本研究的 `brian2-rust/output` 是指向
`/atlas-storage/0002/codex-flywire-learning/output` 的软链接，`target` 同样位于该卷。
大图、原生制品、checkpoint 和全部脉冲不放入 Git；小型结果、审计、图表和代码已归档。
这些路径依赖本机 T7 挂载；跨机器复现需使用下列原始数据并重新准备。

| 内容 | 本机路径或结构 |
| --- | --- |
| 只读原始图 | `/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/output/flywire-v783-ei-sensory` |
| 只读原始注释 | `/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/output/flywire-data/annotations-v2.1.0.tsv` |
| 派生分区 | `/atlas-storage/0002/codex-flywire-learning/output/learning-prepared` |
| 完整正式运行 | `/atlas-storage/0002/codex-flywire-learning/output/learning-study-compact` |
| 每个正式条件 | `seed-<seed>-<condition>/configuration.json`, `input.npz`, `report.json`, `snapshot.npz` |
| 五个块的可塑状态 | 每个条件下 `pre/acquisition/post/second/final-plastic.npz` |
| 原生每段证据 | 每个条件下 `project/` 及 `project/run-0002` 至 `run-0005`，含模型、Rust 源、原生结果、摘要和 `native_metrics.json` |
| 保存及恢复 | 正式目录下 `full-save-seed11/acquisition.checkpoint`、`full-restore-seed11`、`full-continuous-seed11` |
| 原生重放 | 正式目录下 `native-replay`、`native-replay-metrics.json` |
| 作业和日志 | 正式目录下 `progress.json` 与各作业 `.log`；已复制登记至本目录 |

checkpoint 为 110,164,425 bytes，在 tick 81,000 保存。它包含完整神经和可塑状态、
时钟、RNG、源事件进度、监控历史、二进制图身份契约及跨保存边界的待投递事件。
两条递归投影各保存 308 个源事件，投递 tick 为 81,000–81,017；这是源事件数，
不是 308 条解剖突触，也不意味着静态/可塑边重叠。恢复后通过 tick 191,000 的
完整数组与连续运行精确一致。checkpoint、payload 和原生重放 SHA-256 见
[checkpoint_audit.json](checkpoint_audit.json)。

数据身份见 [prepared_manifest.json](../prepared_manifest.json)。原始 CSR SHA-256 为
`b84a19b5c6d899181e6ade3a914eba89d53cb3a4cc36789702785b4cfa35ec38`，
注释 SHA-256 为 `30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6`。
科学输入为 `static.b2csr`, `plastic.npz`, `groups.npz`, `static_edge_ids.npy`,
`subgraph.npz`。本次冻结 manifest 还包含外部卷的 macOS `._*` 附属文件，其散列
在审计中仍被检查；后续 prepare 只列五个科学文件。五个科学 payload 散列相同，
但新 manifest 的整体散列会不同，不应当作旧运行身份复用。

## 复现入口

从该分支根目录运行；Brian2 Cython 扩展、NumPy、C++ 编译器和 native Rust runner
需按仓库说明安装。当前 CPU/OS、Python 包、编译器版本及运行中源码散列已记录在
[cpu_environment.json](../cpu_environment.json)。图表另需 Matplotlib。

```sh
cargo build --release --locked --manifest-path brian2-rust/Cargo.toml
python brian2-rust/experiments/flywire_learning/validate_spiking.py --output /path/to/new-spiking
python brian2-rust/experiments/flywire_learning/validate_continuation.py --output /path/to/new-continuation
python brian2-rust/experiments/flywire_learning/prepare.py \
  --graph /path/to/flywire-v783-ei-sensory \
  --annotations /path/to/annotations-v2.1.0.tsv \
  --output /path/to/new-prepared
python brian2-rust/experiments/flywire_learning/study.py \
  --prepared /path/to/new-prepared --output /path/to/new-study --amplitude 40
python brian2-rust/experiments/flywire_learning/report.py \
  --prepared /path/to/new-prepared --study /path/to/new-study --output /path/to/new-report
python brian2-rust/experiments/flywire_learning/plot_results.py --report /path/to/new-report
```

使用新的输出目录。研究脚本会核对身份并只跳过已完整成功的任务；运行状态不明时
先核对进程及日志，不能盲目重启。`report.py` 独立重数完整脉冲、核对时窗和权重，
再重算各行及汇总。最终报告中的研究解释为人工撰写，图表由机器可读结果确定性生成。
本次 40/80 mV 校准及选择 40 mV 的预设回退见 [input_calibration.json](../input_calibration.json)；
复现既定结果不需要重新搜索参数。

## 失败历史和解释边界

系统盘不足后仅把本研究产物移至 T7。旧 `learning-study` 保留密集控制表导致的
编译中断现场；改成 50 ms 表后逐点核对与原 .1 ms 控制相同，积分仍为 .1 ms。
首个 compact paired 的 10.6 s 浮点时钟校验误拒已修为严格整数 tick 校验，失败
保存在 `failed-attempts/seed-11-paired-1789125585689261000`。只有 supervisor
确认非零退出后才重试；旧/新已完成前缀原生结果 SHA 相同，见
[clock_fix_prefix_equivalence.json](../clock_fix_prefix_equivalence.json)。没有改变科学协议。

耗时来自共享主机：原生循环 70.95–847.32 s/条件，端到端 119.78–986.82 s；
最大原生 RSS 350.3 MiB，前端 1303.4 MiB，二者不是并发进程树内存之和。
5 个种子和每块每种刺激仅 2 次测试限制推断；首次小幅配对差异与后续静默均完整报告。
机制与工程验收通过不等于稳定关联、生物机制拟合、长时记忆或真实拓扑优越性。
