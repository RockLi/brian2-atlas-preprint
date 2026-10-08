# 公开仓库元数据核查

核查时间：2026-09-12。附件来自公开 OSF / Figshare API，仅保存仓库元数据与文件清单，未下载 EEG、人口学、药理或遗传原始数据。

- [OSF rsnu4](https://osf.io/rsnu4/)：`osf_raw.json` 是 Raw Data 目录；以文件夹 ID 命名的 JSON 是目录分页。`osf_epochs_*_complete.json` 合并了两个患者组 Epoch Data 的全部分页；每组 20 个个人 FIF 文件及 1 个汇总 H5。其他目录未全部遍历，不能以分页数推断总量。健康组已读取目录只出现汇总结果与人口学文件，未确认其逐试次 EEG。
- [GluN2D Figshare 30436471](https://doi.org/10.6084/m9.figshare.30436471)：`glun2d.json` 是完整条目元数据。40 个文件共 5,363,410 字节，主要是 Prism 绘图/统计文件，并非已核实的完整连续原始记录。

目录名称和文件名不能保证事件编码、有效试次、数据完整性或可分析性。上述项目尚未开展信号质量核查。
