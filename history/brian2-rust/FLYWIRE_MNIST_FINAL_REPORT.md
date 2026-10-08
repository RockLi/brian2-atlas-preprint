# FlyWire 数字识别：最终实验与工程交付报告

日期：2026-09-10。开发分支：`codex/flywire-mnist`；集成目标：`codex/execution-plan-research`（ExecutionPlan / Metal / CUDA / WASM 的 GPU 研究分支）。

## 结论

基于真实 FlyWire 连接组构建的固定脉冲神经网络，其下游活动可以支持 MNIST 数字分类。完整 **60,000 张训练 / 10,000 张官方测试**的 CPU 实验准确率为 **87.14%**，全部样本、模型、输入与预测审计通过。这是一项可行性结果；相同测试图片上的直接投射输入、像素分类器和 CNN 均更准确，未证明真实拓扑带来额外性能收益。

学习发生在**外部读出器**。内部突触、结构和动力学固定，输入为人工 ALPN 刺激，不能表述为果蝇自然视觉、内部学会识字或已完成内部突触可塑性训练。

工程上已交付真实 CPU 单图推理、游戏与手写界面，以及无需推理后端的完整网络 WASM 静态分发。CPU、GPU、MPI 开发接口均保留，但 CPU 全量结果不能改标成 GPU / MPI / WASM 全量成绩。

## 1. 正式模型与训练过程

| 项目 | 配置 |
| --- | --- |
| 连接组 | FlyWire v783；139,255 个神经元、15,091,983 条有向加权连接 |
| 动力学 | 固定 conductance-LIF；每张图片从同一冻结初态重新仿真 |
| 输入 | 28×28 灰度图，规则脉冲 60 Hz，外部增益 64，fanout 4，人工投射到 302 个符合条件的兴奋性 ALPN |
| 时间 | dt 0.1 ms；100 ms 预热 + 200 ms 刺激 + 50 ms 尾段，共 3,500 步 |
| 监测与读出 | 5,177 个 KC + 96 个 MBON = 5,273 个读出神经元；四个 50 ms 计数窗口 |
| 分类器 | 神经活动 sqrt-sum 表示，Nyström RBF ridge；5,000 个基点仅从训练数据选择 |
| 学习范围 | 外部编码/刺激/读出校准及外部读出拟合，没有内部突触学习 |
| 种子 | 正式全量 seed 783；前期另有三个固定种子的 5,000/1,000 训练/验证确认 |

流程：图片 → 人工脉冲编码 → 固定 FlyWire 动力学 → KC/MBON 活动 → 外部读出 → 0–9 分数。测试标签不进入仿真或预测函数。

正式协议 SHA-256：`724ecf6176691d9862cec2002b1c1a5c9c37141f5990634ec77848ecb72bfd4c`。

训练特征覆盖全部 60,000 张，其中 6,500 张复用经过逐位核对的同条件校准活动，其余重新仿真；10,000 张测试全部新仿真。三个外部读出均拟合完整训练数据后锁定。神经读出拟合约 65.94 秒；此前的大量仿真是特征提取成本，两者不能混称为反向传播训练时间。

## 2. 同一官方测试集上的结果

| 方案 | 准确率 | 95% Wilson 区间 |
| --- | ---: | ---: |
| 固定 FlyWire 活动 + 外部读出 | **87.14%** | 86.47%–87.78% |
| 实际投射输入 + 匹配读出 | 95.40% | 94.97%–95.79% |
| 原始像素 + 匹配读出 | 97.62% | 97.30%–97.90% |
| CNN 对照 | 98.80% | 98.57%–99.00% |
| 原始弱刺激 FlyWire 流程（历史） | 23.64% | 22.82%–24.48% |

神经活动相对投射输入为 −8.26 个百分点，逐图配对 bootstrap 95% 区间 [−8.84, −7.69]；相对像素为 −10.48 个百分点，区间 [−11.11, −9.86]。未发现稳定优势。

从历史 23.64% 到 87.14% 的改进同时涉及编码、刺激增益、采集范围和读出，不能归因于单一因素。CNN 为 80,202 参数模型，训练内部验证选择 7 个 epoch 后以全部 60,000 张重训；已完成的 CPU 流程约 97.61 秒。其密集张量训练与每图 3,500 步的全图仿真工作量不同，这些耗时不是同算法的后端速度对比。

前期三种子 RBF 神经读出验证准确率为 84.2%、85.0%、85.6%。匹配线性读出下，神经活动与投射输入平均差 +0.27 个百分点，区间 [−1.07, +1.57] 跨零；不能用小验证集结果替换正式全量 RBF 比较。

**局限：**公共测试集已被历史实验观察，本次不是从未接触过的最终留出集。正式全量只有一个种子，且未完成统计匹配的随机重连拓扑比较。当前结果不证明真实拓扑优于随机网络，也不推断其他任务或编码一定没有优势。

依据：[完整结果](validation/FLYWIRE_MNIST_OPTIMIZED_RESULTS.md)、[终态报告](validation/evidence/optimized-full-complete-20260910/report.json)、[配对与 CNN 对齐](validation/evidence/optimized-full-complete-20260910/paired-check.json)、[训练报告](validation/evidence/optimized-full-complete-20260910/training-report.json)。

## 3. 因果消融与审计

规则输入消融使用同一 5,000 张训练 / 1,000 张验证图片，全部来自官方训练集。唯一干预是切断 ALPN 输出：6,000/6,000 张的投射输入保持逐位相同，ALPN 仍响应输入，而 KC/MBON 活动全部与空白一致。匹配线性分类准确率从 **85.7% 降到 10.0%**。

这支持当前任务信息通过 ALPN 输出影响下游活动；不证明真实拓扑特有收益，也不是正式 10,000 张测试上的 RBF 消融。见 [完整消融及只读复核](validation/FLYWIRE_MNIST_REGULAR_CUT.md)。

正式全量审计覆盖 1,875 个训练分片、313 个测试分片、全部 ID/标签/汇总行、来源与模型哈希、内部模型不变量，以及每张图片四个实际输入窗口的独立重建。三个模型的全部训练和测试预测均复算一致，全训练拟合最优性残差通过。独立统计复核修正了一个极小 p 值被浮点下溢显示为 0 的问题，未改变准确率或结论。

该审计验证制品和计算链一致性，不独立证明整个仿真器正确。原始阶段的动力学和独立 Brian2 诊断保留于 [早期深入审查](review-evidence/flywire-mnist-20260910/REVIEW.md)，不能把早期小规模诊断改标成当前全量对照。

## 4. 工程交付

| 交付 | 行为与边界 |
| --- | --- |
| CPU 训练与恢复 | 按图片多进程并发，独立冻结快照、原子分片、校验后续跑；不是 MPI |
| 原生单图推理 | 复用编译代码，通过校验的 spike-input sidecar 更换输入；每图重建仿真状态 |
| 远程模式 | Mac Studio CPU 服务，展示使用 8 线程，HTTP 接口不接受标签 |
| 浏览器模式 | 同一冻结 AOT Rust 源码经内存宿主适配编译为 WASM/f64；完整网络在 Worker 中运行 |
| 两种界面 | 十跑道游戏和自由手写；仅真实模型，没有预存预测或动画假答案 |
| 静态分发 | 页面、题库、模型、WASM、资产、来源与许可一起打包；成功加载后可缓存离线使用 |
| 录制 | 原创音效及横/竖版真实游戏录制；保存依赖配套服务，本机模式禁用录制按钮 |

游戏题库为**全部 10,000 张官方 MNIST 测试图片**，按样本 ID 洗牌、不放回抽取，不按命中或分数筛选。暂停、切换运行模式、录制不重置题库，刷新页面开启新遍历。失败保留原题等待重试。标签仅用于跑道和命中判定，识别只接收像素。手写示例也来自测试集；自由手写不属于 MNIST 准确率评估。

页面显示来源、原始 test ID 和会话成绩，少量游戏命中不是新的完整测试结果。身体来自 FlyBody 解剖资产，移动与激光是预测可视化，未模拟神经运动控制。

**Active monitored neurons** 的分母 5,273 是 KC/MBON 读出监测集合；分子是该集合中在四个 50 ms 窗口合计至少发放一次脉冲的神经元数。实际仿真仍包含 139,255 个神经元。页面及未来录制均明确这两个规模；旧视频不自动更新。

WASM 为本模型专用 AOT 配置，不宣称通用 BrowserExecutor 已支持全图 CSR。13 次真实 WASM 仿真覆盖 0–9 和 A→B→A，与单线程 CPU 的输入、活动哈希和预测一致；分数最大差约 7.41e−14，异常输入后的恢复通过。未重新运行 WASM 全量 10,000 张准确率评估。

依据：[游戏](validation/FLYWIRE_MNIST_GAME.md)、[浏览器实现](validation/FLYWIRE_MNIST_BROWSER.md)、[WASM 对齐](validation/evidence/browser-local-20260910/wasm-check.json)、[完整题库离线验收](validation/evidence/mnist-test-arcade-20260910/browser-check.json)、[活动计数](validation/evidence/activity-label-20260910/check.json)。

## 5. 性能与多后端状态

Mac Studio 同一组 13 个输入，CPU 1 线程到 8 线程的单请求中位耗时为 **1.922 → 0.863 秒（2.23 倍）**，活动和分数严格相同。计时包含编码、仿真、读出及每请求完整性检查，不含首次模型加载、网络和动画。网页另有 HTTP/SSH 往返，视觉降速不是推理提速。见 [性能验证](validation/FLYWIRE_MNIST_INFERENCE_PERFORMANCE.md)。

WASM 线性内存约 **534 MiB**，另有约 211 MB 的 JavaScript 读出数组、渲染和临时加载内存，534 MiB 不是浏览器总占用。压缩模型约 54 MiB，完整题库约 1.58 MiB。现场冷启动推理约 4–20 秒、部分后续请求约 1.4–1.9 秒，也观察到更慢请求；不承诺比服务器快或已适配手机。其主要价值是无需推理后端。

| 范围 | 已完成 | 未完成 |
| --- | --- | --- |
| CPU f64 | 完整科学评估、审计、单图推理、线程对齐与展示 | 全程峰值资源及跨硬件成本矩阵 |
| GPU Metal/CUDA | 保留 GPU 基础设施；FlyWire 实际 Metal 全图小样本及 CPU-f32 对照 | 当前冻结方案完整 GPU 分类评估、NVIDIA FlyWire 真机验收、系统 f32/f64 敏感性研究 |
| MPI CPU | 隔离适配入口、双 rank 参考对照 | FlyWire 多节点全量规模、成本及交互展示验收 |
| WASM f64 | 完整网络单图、对齐、Worker、双模式、静态离线演示 | 全量准确率重评、移动端内存与多浏览器长期资源验收 |
| 生物结构/学习 | 区分能力和输入通路消融 | 匹配随机重连主比较、内部可塑性学习、自然视觉输入 |

GPU 基础设施既有验收见 [GPU 交付报告](GPU_DELIVERY.md)，不能与 FlyWire 应用验收混为一谈。MPI 其他科学任务继续在独立分支，不纳入本次合并。

## 6. 本次合并前验证

| 检查 | 结果 |
| --- | --- |
| FlyWire Python 实验、审计、输入、推理、HTTP | 59 passed，2 skipped（可选 FlyWire Metal/MPI 门控） |
| ExecutionPlan / GPU 校验及外部脉冲发生器 | 42 passed，12 CUDA skipped；通过数包含 12 项真实 Metal |
| Node 运动轨迹 | 3 passed |
| 题库 | 两轮各完整 10,000 张无重复；来源哈希、类别计数、坏输入拒绝通过 |
| 本机输入与生命周期 | 26 个 Python 预处理对齐样例；快速模式切换/旧 Worker 取消通过 |
| 分发包 | 最新包 41 个文件逐一哈希验证；另已进行真实浏览器离线验收 |

首次 HTTP 检查因沙箱禁止端口绑定，首次 Metal 检查因沙箱不可见 GPU 而失败；获得对应本机权限后分别补跑 6 / 12 项，全部通过。汇总保留重试原因。没有在本次运行 NVIDIA GPU 或重新进行 70,000 张科学实验。

合计 **101 项 Python 检查通过、14 项跳过**。详见 [逐用例集成清单](validation/evidence/final-integration-20260910/tests.json)、[题库检查](validation/evidence/final-integration-20260910/dataset-tests.json)、[Worker/预处理检查](validation/evidence/final-integration-20260910/browser-client-tests.json)。

## 7. 制品与复现入口

| 制品 | 位置 |
| --- | --- |
| 正式模型与完整活动分片 | Mac Studio `rock@100.90.28.27`：`/atlas-home/0004/workspace/flywire-mnist-20260909/optimized-full-20260910` |
| 正式源、数据与环境 | 同服务器 `/atlas-home/0004/workspace/flywire-mnist-20260909/`，保留冻结 source 与版本化入口 |
| 审计、逐图预测、摘要 | 仓库 `validation/evidence/optimized-full-complete-20260910/` |
| 最新离线包 | 本机 `/private/tmp/brian2-flywire-mnist/brian2-rust/output/flylab-browser-mnist-test-activity.zip` |
| 双模式页面 / 静态预览 | 本机 `http://127.0.0.1:18767/` / `http://127.0.0.1:18769/` |

zip 为 70,258,177 字节（约 67 MiB），SHA-256：`19c369aadab3c2d161ba1bb7bea126fd575aa7af994994c078fc6d7c256b5ff2`。

完整图、读出模型和活动分片未全数提交 Git；大制品位于服务器及忽略的 `output/`。离线 zip 包含运行所需模型，仓库包含构建脚本、来源哈希与许可说明。仅克隆代码不会自动获得完整模型。静态包需 HTTPS 托管或 localhost 文件服务，不能以 `file://` 双击运行；浏览器可能回收缓存。

在仓库根目录、备齐环境与冻结模型后，CPU 服务入口为：

```sh
export PYTHONPATH="$PWD/brian2-rust/python:$PWD/brian2-rust/experiments:$PWD/brian2-rust/validation:$PWD"
python brian2-rust/validation/flywire_mnist_demo.py \
  --root /path/to/flywire-mnist-root \
  --artifact /path/to/flywire-mnist-root/optimized-full-20260910 \
  --threads 8 --port 18765
```

单图 `.npy` 入口为 `validation/flywire_mnist_predict.py --root ... --artifact ... --image digit.npy --result NEW_RESULT.json --threads 8`，结果须写在冻结目录之外的新文件。

浏览器重建依次使用 `validation/flywire_mnist_browser_export.py`、`tools/build_wasm_aot.py`、`tools/package_mnist_test.py`、`tools/package_flywire_browser.py`，参数与工具版本见 [浏览器复现说明](validation/FLYWIRE_MNIST_BROWSER.md)。正式全量与只读审计入口见 [优化全量流程](validation/FLYWIRE_MNIST_OPTIMIZED_FULL.md)。复现科学结果使用对应冻结源码/协议及制品，不覆盖原冻结输入。

## 8. Git 集成与后续

GPU 基线 `f014e616d` 是 FlyWire 分支的祖先，可使用 `git merge --ff-only codex/flywire-mnist` 集成。相对基线，核心原生生成器增加经过校验的外部 spike-input 与动态实例事件长度支持；其余主要为实验、读出、审计、测试、界面与交付工具。

GPU 工作区既有 `wasm/index.html`、`wasm/lab.css` 两处未提交修改与此次差异不重叠，保留原内容；`.playwright-cli/` 未跟踪目录也保留。主工作区 `gate0/minimal-rust-backend` 和独立 MPI 分支不改动。本次为本地提交和集成，不包含远端推送。保留检查见 [目标工作区记录](validation/evidence/final-integration-20260910/target-preservation.json)。

后续优先级：匹配随机重连与多种子比较；同冻结任务的 GPU/MPI 全量评估；浏览器内存和冷启动；另立内部突触可塑性实验。每项分别冻结输入、参数、比较口径与验收标准，保留本报告的固定网络结论作为基线。
