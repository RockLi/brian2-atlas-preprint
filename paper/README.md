# Brian2 Atlas 论文材料

当前稿件包含 **10 张主图、7 张主表（含 Appendix A / Table 7）、16 项参考文献**，补充材料为 **S1–S21**。数量与来源校验以[构建报告](validation/build_report.json)为准。

题目：**A Unified Intermediate Representation and Execution Architecture for Heterogeneous and Distributed Neural Simulation**（面向异构与分布式神经仿真的统一中间表示与执行架构）。作者：**Xinjun Li / Independent researcher**。

## 阅读与编辑入口

| 文件 | 用途 |
|---|---|
| [英文正文](MANUSCRIPT.md) / [HTML 阅读版](MANUSCRIPT.html) | 正文、参考文献与兼容性附录 |
| [补充方法与证据](SUPPLEMENTARY.md) | S1–S21、实验配置、数值门槛与复现材料 |
| [中文译稿](zh-CN/README.md) | 正文及合并进 PDF 的补充内容 |
| [PDF 构建说明](PDF.md) | 从保留数据生成阅读版本 |
| [当前验收记录](data/release_validation/acceptance.json) | 实现引用、CUDA/PD14 验收及当前 PDF 身份 |
| [PDF 检查](validation/current_commit_pdf_qa.json) | 页数、链接、文本边界与视觉复核 |
| [实验代码覆盖清单](../experiments/COVERAGE.md) | 正文及 S1–S21 的代码入口、已补齐内容与仍存在的构建限制 |
| [修订记录](CHANGELOG.md) | 各轮修订的实际范围 |

默认 PDF 包含正文、Appendix A、S13 / Table S5、S21 / Table S4 / Figure S2，以及三页 Supplementary Figure S1。其余补充章节见完整 SUPPLEMENTARY.md。生成的 PDF 位于仓库根目录 `output/pdf/`，不纳入 Git。

## 实现与验收范围

维护中的实现为 [brian2-atlas](https://github.com/RockLi/brian2-atlas)，论文引用固定 commit；准确版本见正文与[验收记录](data/release_validation/acceptance.json)。公开入口为 `brian2_atlas`、`AtlasDevice` 和 `set_device("atlas", ...)`；`brian2_rust` 与 `rust_standalone` 保留兼容。Rust 是实现语言，Atlas 是统一后端名称。

Atlas 保留 Brian2 建模前端，独立实现其执行路径。Brian2CUDA、Brian2GeNN 和 GeNN 是部分实验的外部比较对象，不是 Atlas 运行时依赖；CUDA 执行仍需要 NVIDIA 工具链与驱动。

Modal L4 后续验收覆盖先前跳过的 708 项及 1 项 ABI 回归，共 709 项通过，零失败、零跳过；包含两个 MPI rank 共用一张 GPU。此结果不证明跨主机或多 GPU 训练。PD14 固定版本入口已在 `bf1cf30af55a4a14ae42d0d75534728385b62d06` 上重新构建和运行，见[复现指南](../experiments/reproduction/pd14/README.md)。各性能测量保留实际源码、输入、工具链和硬件身份。

## 数据与重建

从仓库根目录执行：

```sh
python -m pip install -r paper/requirements-lock.txt
python paper/scripts/build_manuscript.py
```

该命令重建图表与 HTML，检查保留来源哈希、图表结构和数值一致性，不启动新的仿真或硬件测量。PDF 导出依赖及命令见[构建说明](PDF.md)。历史收集脚本的输入要求见[收集说明](COLLECTION.md)，重建稿件无需运行它们。

| 材料 | 范围 |
|---|---|
| [图表数据](data/figure_data.json) / [计划选择](data/plan_selection.json) | 报告样本、数值口径及来源身份 |
| [树突与训练证据](data/v3/evidence.json) | 默认树突实现、缓存消融和分阶段训练结果；原始记录不合并为共同硬件验收 |
| [已发表模型](data/published_models/evidence.json) | NMDA 与 Onasch 工作流及各自边界 |
| [弱扩展](data/capacity/weak-evidence.json) | 3/6/12/24/30 台主机，每档一次计时观测 |
| [条件资源分析](data/full_scale/analysis.json) | 86B 参考计数下的条件数组与通信模型，不是可行性或运行时间预测 |
| [归档目录](../archives/README.md) | 大型原始产物的校验和及保存位置；尚未提供公共归档下载 |

写作期的 [OUTLINE](OUTLINE.md)、[NOVELTY](NOVELTY.md)、[EVIDENCE_MATRIX](EVIDENCE_MATRIX.md)、[REFERENCES](REFERENCES.md) 以及带版本后缀的旧稿用于历史对照，不能替代当前正文、构建报告与验收记录。它们的阶段性数量、实验边界和未完成事项应按记录日期理解。
