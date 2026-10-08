# Brian2 Atlas 双仓库迁移方案

日期：2026-10-08。状态：已完成只读盘点；本文为迁移方案，尚未创建目标分支、推送源码或修改 GitHub 默认分支。

## 仓库职责

- `RockLi/brian2-atlas`：Atlas 产品源码、功能回归、安装构建、用户文档、发布流程。
- `RockLi/brian2-atlas-preprint`：论文、图表生成、实验脚本、复现环境、已筛选证据及数据索引。
- 当前 `RockLi/brian2` 和本地 checkout：保留迁移前的开发历史及尚在运行的研究工作，完成交接后停止双份维护核心源码。

Atlas 是本项目的正式发行仓库。README 应明确项目与上游 Brian2 的关系，并保留原有作者、许可及来源说明。

## 已核实的基线与迁移风险

| 项目 | 2026-10-08 盘点结果 |
|---|---|
| 当前分支 | `next-dev` |
| 当前 HEAD | `81eb571d78292a0e24a8e79980ea0d5c26e8a48c` |
| 当前 HEAD 与本地 `origin/master` 的共同祖先 | `cf0b25cce13b620aac740873e88ba91061a70275`，2026-09-01 |
| 本地 `origin/master` | `532d97267ead6d9e28acee4d2e4d37699dd001ad`，2026-09-14；不是本项目分叉起点 |
| 本轮 SSH 查询的上游 `master` | `27b5431168cf9959f0c27894cd4d92cdc6ad5c31`；这是新仓库的候选 base，尚未验证 Atlas 在此版本的兼容性 |
| 原基线之后的项目提交 | 558 个；包含产品、实验和证据变更，需按内容拆分 |
| 当前仓库 `.git` 占用 | 约 4.5 GiB；包含研究材料历史 |
| 已跟踪执行计划证据 | `brian2-rust/execution-plan-evidence/` 下 23,708 个文件 |
| 已跟踪验证文件 | `brian2-rust/validation/` 下 5,666 个文件 |
| 已跟踪 MPI 证据 | `brian2-rust/mpi-evidence/` 下 5,253 个文件；工作区目录约 3.1 GiB |
| 尚未跟踪的核心内容 | `src/` 30 个、`python/` 28 个、`tests/` 230 个文件 |

上述数字是盘点时观测值。工作区仍在开发，执行迁移前必须重新采集清单并校验内容；HEAD 无法单独代表当前实现。

前端也有 Atlas 修改。例如已提交的 `brian2/input/binomial.py`，未提交的函数、代码生成、单位检查修改，以及未跟踪的 `brian2/codegen/indices.py`。只迁移 Rust 目录，或只迁移 HEAD，会遗漏依赖。

新仓库可通过 SSH 读取；本轮查询未返回 `HEAD`、`dev` 或 `main` 引用。迁移推送前仍需读取全部远端引用，避免覆盖期间新增的内容。

## Atlas 从上游 base 逐步 port

根据用户最新选择，以固定的上游 Brian2 commit 建立 Atlas `dev`，按功能和依赖逐块移植我们的修改。每块都有可解释的提交、来源清单和相应验证，不一次性覆盖整个产品目录。

1. 保留当前 checkout，继续允许已有实验使用它；在独立目录构建迁移副本。
2. 从官方 Brian2 仓库取得固定 base 及其上游历史。优先预检本轮查询的最新 `master`，同时保留原共同祖先作为比较基准；若最新上游兼容性阻塞明显，报告具体差异后再决定基线。不要推送混合仓库的所有分支、tag 和完整 `.git`。
3. 将旧工作区的产品修改相对原共同祖先提取，再逐块应用到新 base。不能用旧版整个 `brian2/` 覆盖新上游目录，否则会抹掉上游的新修复。
4. 已提交的产品独立提交可以选择性 cherry-pick；混合提交提取产品部分重新提交并记录原 commit。未提交修改和新文件也要进入迁移清单，未归类内容进入复核清单。
5. 按下面的阶段推进；每个阶段构建、运行对应回归，通过后进入下一阶段。每次迁移一组能一起工作的代码，包含必要测试、fixture 和构建配置。
6. 记录旧仓库 URL、旧 HEAD、选定上游 base、源文件 SHA-256、原提交/工作区来源及新提交映射。最终核对所有产品修改都有已迁移、已被上游实现替代或有理由延期的去向。

这样保留 Brian2 的上游历史，同时让新的 Atlas 提交历史按产品功能展开。原有 Atlas 开发历史仍保存在旧仓库，拆分后重建的提交通过来源映射追溯；不要把重新组织的历史描述为所有旧项目 commit 都保持了原身份。

候选最新上游尚未拉入本地或运行测试。仅已缓存的 2026-09-14 上游与原基线相比，已包含 StateMonitor 和 Synapses 修复，因此基线升级确实需要差异评估。

## Port 阶段与验证门槛

| 阶段 | 内容 | 进入下一阶段前的验证 |
|---|---|---|
| P0：纯上游 base | 独立 checkout、固定上游 commit、`origin/upstream`、建立 `dev` | 干净环境安装和相关上游测试可运行；记录基线已有问题 |
| P1：Brian2 前端增量 | Binomial 参数契约、函数与代码生成、索引、单位检查等产品修改；按依赖拆成若干提交 | 对应前端回归；保留 NumPy/Cython/C++ 原有行为，检查上游已有同类实现 |
| P2：CPU 仿真闭环 | Device 注册、B2IR/协议、导出与计划、Rust reference/AOT、结果与 monitor；先形成可用闭环，再分提交补齐模型能力 | 构建、最小模型、reference/AOT 与 Brian2 对照、生命周期和兼容回归 |
| P3：GPU/WASM | Metal、CUDA、WASM/WebGPU 及其计划和交付；依赖 P2，按目标逐组移植 | 各目标构建和可用硬件回归；缺少硬件的项目显式保留待验收状态 |
| P4：MPI | 分布式计划、分区、事件队列、运行工具、checkpoint、GPU rank 卸载；依赖所用的 P2/P3 能力 | 1/2/4 ranks 对照、状态恢复和事件顺序；GPU MPI 单独验收 |
| P5：原生训练 | CPU trainer、Brian lowering、时间/事件/函数/状态契约，再逐步加入 GPU/MPI 训练 | 前向、VJP、更新与恢复专项回归；不能只以分类 loss 下降作为通过标准 |
| P6：发行与论文交接 | 产品示例/文档、完整打包、版本与 CI；preprint 路径适配及源码版本固定 | 干净 wheel/sdist 安装、代表性模型、论文图表生成和一条复现流程 |

P3、P4、P5 的内部顺序按实际依赖细化。上表是规划阶段，不代表每个阶段只有一个提交；公共接口与 kernel、训练或 MPI 中跨模块引用必须一起处理。

现有 `brian2_rust/__init__.py` 同时导入仿真、MPI、WASM 和训练接口，Rust `lib.rs` 也引用训练模块。阶段性迁移时需要同步收敛当期导出和模块声明，随后随对应实现恢复；否则仅迁移部分文件会导致基础导入或构建失败。

产品发行名与地址可在早期独立的 metadata 提交中设置；完整 runner 打包和发布 CI 在 P6 完成。GPU/MPI/训练等暂未 port 的能力列为阶段性待迁移，不从最终目标中默默移除。

## 文件去向

| 当前路径 | 去向 | 处理方式 |
|---|---|---|
| `brian2/` | Atlas | 保留完整 Python 前端、上游测试及我们的修改 |
| `brian2-rust/src/`、`python/` | Atlas | 包含全部新源码，排除编译及解释器缓存 |
| `brian2-rust/tests/` | Atlas | 保留功能与兼容回归；检查测试依赖的外部脚本和小型 fixture |
| `Cargo.toml`、`Cargo.lock`、`rust-toolchain.toml` | Atlas | 保留锁定工具链和构建输入 |
| `brian2-rust/wasm/` | Atlas 为主 | 保留应用源码与构建/验收工具；生成的 WASM/JS 二进制按交付需求另列，不默认提交全部产物 |
| 上游 `examples/` | Atlas | 保留可用示例及 Atlas 适配，移除 standalone 生成目录 |
| `brian2-rust/examples/` | 逐文件拆分 | 用户示例进 Atlas；论文比较、计时采集和专项诊断进 preprint |
| `brian2-rust/tools/` | 逐文件拆分 | 安装、构建、运行和产品验证工具进 Atlas；论文实验及图表工具进 preprint |
| `B2IR.md`、`COMPATIBILITY.md`、GPU/MPI/训练/WASM 文档 | Atlas 为主 | 产品契约与使用说明保留；历史结果和论文比较报告进 preprint |
| `docs_sphinx/`、`dev/` | Atlas | 保留必要上游文档与工具，重写发行入口；检查文档 submodule |
| `docs/preprint/` | preprint | 当前目录被根 `.gitignore` 的 `/docs` 忽略，必须显式纳入迁移清单 |
| `brian2-rust/experiments/` | preprint | 保留实验模型、流程和配置 |
| `mpi-evidence/`、`execution-plan-evidence/`、`validation/`、`review-evidence/` | preprint/外部归档 | 按论文和复现用途筛选脚本、小型结果、来源哈希；完整保留原始记录于旧仓库/归档 |
| `evaluations/` | 逐项归档 | 迁移有用的评测脚本与总结；源码快照和环境副本不整体导入 |
| 研究提案、无关领域调研 | 原仓库/研究归档 | 不因拆分而自动放进 Atlas 或当前论文仓库 |
| `output/pdf/` 与论文图表产物 | preprint | 将最终稿、必要图表及生成脚本列为明确例外；不整体复制 `output/` |
| `.venv*`、`target*`、缓存、standalone 目录、原始大型输出 | 留在本地/外部归档 | 不进入源码仓库；复现输入需要地址、大小、哈希和版本 |
| LICENSE、AUTHORS、CONTRIBUTORS 等来源材料 | 两仓库按所含内容保留 | 导入、复制或引用第三方材料时保留相应来源信息 |

迁移时保持 `brian2-rust/` 和 `brian2_rust` 名称。多个模块按 `Path(__file__).resolve().parents[2]` 寻找后端根目录，安装脚本也依赖当前布局。路径整理作为后续独立变更。

“我们的所有修改”应包含所有产品修改；论文脚本和历史诊断按职责迁入 preprint 或研究归档。不能用“未提交”或“实验目录”作为判定实现是否应保留的唯一依据。

## preprint 的版本与复现

建议最终组织为 `paper/`、`experiments/`、`scripts/`、`evidence/`、`data/`、`environments/`，另用 `atlas/` submodule 指向 Atlas 的确定 commit。

- 每个论文实验记录实际源码身份、工具链、依赖、随机种子、命令、硬件、输入/输出哈希及证据范围。
- 已有结果绑定其实际历史源码版本。不能把此前不同版本的实验全部改记为新的迁移 commit。
- 新的复现流程使用固定 Atlas commit；不依赖 `dev` 的浮动最新内容，不另维护一套可编辑引擎源码。
- 原脚本中的相对路径、runner 路径和源码根目录引用逐项适配到 submodule。完成路径测试前，不能声称拆分后已经可复现。
- 大型原始数据和完整执行归档使用独立存储及校验清单，Git 中保留可重建的脚本、小型图表数据和关键证据。
- 论文提交时，用 preprint tag 固定论文仓库；tag 内 submodule 固定引擎。更早实验的源码归档或旧仓库 commit 继续保持可定位。

## 分支与发布

- Atlas：`dev` 作为 GitHub 默认分支，接收开发和 PR；正式发布由独立 Atlas 版本 tag 标记。需要维护稳定发布分支时再建立 `main`。
- preprint：建议 `main` 为默认分支，以 `preprint-v1`、`preprint-v2` 等 tag 固定各次公开稿件。若希望两个仓库统一使用 `dev` 也可行，复现仍必须依赖 commit/tag。
- 默认分支、稳定分支和发布版本是三个不同概念；`dev` 作为默认分支不意味着其中每次提交都正式发布。
- Atlas 的 `origin` 指向 `RockLi/brian2-atlas`，`upstream` 指向 `brian-team/brian2`。新仓库使用独立配置，不修改当前 checkout 的 remotes 或分支。

## 包装与 CI 的必要调整

当前根 `pyproject.toml` 的 distribution 名为 `Brian2`，项目 URL 指向上游；包发现配置排除了 `brian2-rust*`。安装器通过额外 `.pth` 才加入 `brian2_rust`，这是开发安装方式，不是完整发行包。

迁移后的打包目标建议为 distribution `brian2-atlas`，继续保留 `import brian2` 和 `import brian2_rust`。需要显式打包这两个模块，并包含所需模板、原生构建输入及 runner 交付机制。在同一环境中安装上游 Brian2 与 Atlas 的覆盖关系必须写清楚。

检查并改写版本生成配置和 tag 匹配规则，避免把上游 Brian2 tag 自动当作 Atlas 版本。上游发布 CI 必须改为 Atlas 的构建/验证流程；发布凭据和 tag 触发配置在安装包验收后启用。

首轮迁移保留已实现的 CPU/GPU/MPI/WASM/训练代码。发行文档按目标列出实际验收范围，迁移不等于宣称所有目标都已达到发布标准。

## 执行顺序与验收

1. 生成文件级去向清单，包含 tracked、untracked 以及必需的 ignored 文件；记录内容哈希并校验快照是否稳定。
2. 在独立目录验证并固定上游 base，建立 Atlas `dev`。保留原 checkout 和远端历史。
3. 按 P1–P6 逐块 port、提交并验证；每块记录来源和测试结果。已验证的阶段可逐步推送到 `dev`，不要把半完成的阶段标成正式发行。
4. 最终核对产品修改去向，检查核心测试所需的 fixture、路径依赖和生成文件排除规则。
5. 验证 wheel/sdist 的安装内容、runner 获取方式和用户文档。安装器能运行不等于 wheel 已合格。
6. 在独立目录准备 preprint，从固定 Atlas commit 验证图表构建和至少一条端到端实验流程；不为目录迁移重跑所有大型实验。
7. 首次推送前读取目标仓库的全部远端 refs，核对仍为空；首次有效分支推送后设置默认分支，后续推送遵循正常增量流程。
8. 发布交接清单：新 Atlas commit、preprint commit、旧基线、文件/提交映射、测试结果、剩余待验收项。完成交接后，核心开发转至 Atlas。

验收完成前不删除或重写旧仓库，不使用强推覆盖目标内容。方案确认和实际执行是独立步骤；本文件没有执行上述迁移。
