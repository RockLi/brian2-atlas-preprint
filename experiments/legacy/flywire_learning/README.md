# FlyWire 局部学习研究

研究启动日期：2026-09-11。分支：`codex/flywire-learning`。

基线为 `0bc46a102290846836c5d1c76e19e6c027c34d2c`，从
`gate0/minimal-rust-backend` 的已提交状态建立独立 worktree。
未引入 `codex/flywire-vision`、MNIST 或其他分支的代码和未提交修改。

**CPU 第一版已完成：真实全脑固定网络、局部 KC→MBON01 塑性、25 组预设实验，
以及全脑连续/分段、保存恢复与重放验收均已完成。**
首次训练出现小幅配对差异，但第二轮训练后配对条件在 4/5 个 seed 中对 A、B
同时静默；本次没有建立稳定可读出的奖励关联或可靠反转证据。
详见 [最终研究报告](results/cpu-v1/RESULTS.md) 和
[逐项验收及原始产物索引](results/cpu-v1/ARTIFACTS.md)。

研究问题：保持全脑拓扑及其余连接固定，仅允许指定 KC→MBON 权重变化，
能否获得依赖奖励配对、可反转、在撤掉教学信号后仍可测得的刺激关联？
第一项任务为两个合成嗅觉刺激 A/B 的差异性奖励关联。
成功不等于真实果蝇行为复现，也不预设真实拓扑优于随机拓扑。

## 本目录可运行内容

需要 Python、NumPy；测试另需 pytest。运行时采用本 worktree 的源码，
可使用现有虚拟环境解释器。所有输出目录必须不存在。

```sh
python brian2-rust/experiments/flywire_learning/audit_connectome.py \
  --graph /path/to/flywire-v783-ei-sensory \
  --annotations /path/to/annotations-v2.1.0.tsv \
  --output brian2-rust/output/flywire-learning-audit

python brian2-rust/experiments/flywire_learning/mechanism.py \
  --output brian2-rust/output/flywire-learning-mechanism

python -m pytest -q \
  --confcutdir=brian2-rust/experiments/flywire_learning \
  brian2-rust/experiments/flywire_learning/test_mechanism.py
```

`audit_connectome.py` 复用现有二进制 CSR 格式验证器，以只读 memmap 读取数据。
验证官方 EI 图 SHA-256、原始 annotation TSV SHA-256，以及派生 CSV 的 root ID、
索引和细胞类型对应。导出的 `kc_mbon_edges.npz` 保留原图 edge ID、全局索引、
精确 uint64 root ID 和 signed_contacts；不重新编号冒充原始边。
`mbon_inventory.csv` 用于后续人工核对分区和输出解释。

`mechanism.py` 完全独立于连接组，是 128 个合成 KC、2 个抽象 MBON 通道的
速率参考模型。它用于确定实验协议和检查学习开关、时延、反转及保存行为。
它没有模拟 LIF、真实 DAN、嗅觉输入传播或生物记忆时长。

完整依据、实现过程和初期方案见 [研究记录](RESEARCH.md)。
最终小型产物保存在 [results/cpu-v1](results/cpu-v1/)，完整正式运行输出在
`brian2-rust/output/learning-study-compact`；大图、checkpoint 和完整脉冲数组不提交 Git。

## CPU 第一版运行入口

脉冲模拟需要本 worktree 的 Brian2（含 Cython 扩展）、Rust CPU runner 和
C++ 编译器。当前验证环境为 Python 3.14.4、NumPy 2.5.2、native ARM
Rust 1.98.1；C++/NumPy 仅作数值对照。先按仓库安装说明编译 Brian2 扩展，
并用所选 native Rust 工具链执行：

```sh
cargo build --release --locked --manifest-path brian2-rust/Cargo.toml
python brian2-rust/experiments/flywire_learning/record_environment.py \
  --rustc /path/to/rustc --output /path/to/cpu-environment.json
python brian2-rust/experiments/flywire_learning/validate_spiking.py --output /path/to/new-spiking-validation
python brian2-rust/experiments/flywire_learning/validate_continuation.py --output /path/to/new-continuation-validation
python brian2-rust/experiments/flywire_learning/prepare.py \
  --graph /path/to/flywire-v783-ei-sensory \
  --annotations /path/to/annotations-v2.1.0.tsv \
  --output /path/to/learning-prepared
python brian2-rust/experiments/flywire_learning/study.py \
  --prepared /path/to/learning-prepared --output /path/to/new-study --amplitude 40
python brian2-rust/experiments/flywire_learning/report.py \
  --prepared /path/to/learning-prepared --study /path/to/new-study --output /path/to/new-report
python brian2-rust/experiments/flywire_learning/plot_results.py --report /path/to/new-report
```

正式参数和两项输入幅度校准结果已冻结，见 [PROTOCOL.md](PROTOCOL.md)。
`study.py` 顺序运行全脑正确性门、25 个条件、全脑连续/分段/新进程恢复和原生重放。
它只跳过身份一致且已完整成功的任务；发现不完整目录时要求先核实原进程和日志，
避免盲目重启。`report.py` 在真实产物齐全时重新审计输入、权重和完整输出后生成报告。

本机输出软链接指向 `/atlas-storage/0002/codex-flywire-learning/output`，当前正式批次为
`learning-study-compact`。系统盘不足和停止的低效编译现场保留在旧目录，
不能把旧 `learning-study` 的部分文件当作正式矩阵已完成的证据。
