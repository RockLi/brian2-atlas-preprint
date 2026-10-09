# brian2-atlas中文论文译稿

本译稿以`source_manifest.json`记录的44页英文PDF为依据，范围包括正文、附录A、原PDF内合并的S13树突CPU证据／表S5、S21资源分析／表S4／补充图S2，以及三页补充图S1。未并入原PDF的其他补充章节不在本译稿范围内。

- `MANUSCRIPT.md`：正文、参考文献与附录A。
- `DENDRITIC_SUPPLEMENT.md`：完整的S13／表S5中文译稿。
- `RESOURCE_ANALYSIS.md`：完整的S21中文译稿。
- `b2ir_panels.json`：补充图S1中文标题与图注；原始截图来源、哈希与裁剪范围保留。
- `MANUSCRIPT.html`：包含上述全部内容的自包含排版源文件。
- `../../output/pdf/brian2-atlas-preprint-zh-CN.pdf`：合并后的中文PDF。

正文图与补充图S2的说明文字已译为中文；数据、曲线与原始界面截图保留。参考文献作者、论文题目和DOI保留原文。中英文稿件同步修订；翻译与排版重建本身不新增仿真或资源测试。

`translation_validation.json`保存源文件身份、章节和段落对应、表格数值、引用与原始截图核验结果；`pdf_validation.json`保存PDF内容与排版检查结果。

2026-10-09同步更新：CUDA/PD14验收实现固定至`b769c21004a89e2a6f3a14521f23012db654aadd`，正文纳入709项Modal CUDA验收。PD14复测与历史来源保留说明见[当前版本验证记录](../data/release_validation/acceptance.json)。

Atlas公开入口同步为 `brian2_atlas`、`AtlasDevice` 和 `set_device("atlas", ...)`，当前实现引用为 `562ae5cd143ca8fd0bcc070cbef04cad00010246`。
