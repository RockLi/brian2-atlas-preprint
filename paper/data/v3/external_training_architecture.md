# 运行时架构影响补充说明

本说明仅补充解释边界，不改写已冻结源、runtime、结果、种子、预算或既有报告。既有实测数值及其hash保留；凡旧报告中的引擎公平排名暗含ISA相同，应同时读取本限定。

| 对象 | 已核实的架构证据 | 解释 |
|---|---|---|
| 冻结 runtime/b2-train | file/lipo显示单一x86_64 Mach-O；SHA256为83cbd71f425f893578e42a4dfbf90f115aa4b1831201dd1946a116a2e7ea5101 | Atlas原生执行程序不是ARM64构建 |
| environment/cpu/bin/python | platform.machine=arm64，sysconfig平台为macosx-14.0-arm64；translated=0 | Torch及Atlas协调器由ARM64 Python运行 |
| environment/jax/bin/python | platform.machine=arm64，sysconfig平台为macosx-14.0-arm64；translated=0 | JAX主机执行环境为ARM64 |
| SSH的默认上下文 | /usr/bin/uname -m=x86_64；默认Python的translated=1 | 反映该SSH执行上下文，不能将其推广为所有Python子进程的ISA |
| 实际主机 | Apple M1 Ultra；原H0R依赖盘点已记录 | 需要区分Apple Silicon硬件、Rosetta上下文与每个可执行文件的目标架构 |

Atlas的公开API进程链是ARM64 Python协调器调用冻结x86_64原生runtime；Torch/JAX执行路径使用ARM64环境。这构成mixed-ISA主机配置观测，包含x86_64兼容执行路径的影响。没有同一实现、同一条件下ARM64与x86_64的成对消融，不能估计Rosetta带来的百分比，也不能把任何速度差全部归因于Rosetta。

E1-small与E1-large的原始墙时、中位数、全部种子、超时及RSS记录仍是真实观测；Atlas对ARM64 Torch/JAX的比值只能描述当时冻结配置。它们不足以给出与ISA无关的公平引擎排名，也不能证明Atlas算法或引擎本体固有地慢多少。旧validator中的strict_ranking_eligible只体现当时实现的数值/完整性/线程门，没有检验这次新发现的ISA对齐条件，不能把该字段当成架构公平性认证。JAX另外仍有线程预算未对齐问题；即使修正ISA，也不会自动获得strict1thread资格。

不同证据类型需要分别保留：

| 证据 | 仍成立的内容 | 不能推出的结论 |
|---|---|---|
| 已通过的数值Q0 | 冻结实现、实际runtime与指定fixture/tolerance下的检查通过 | 自动推广为新ARM64 runtime、其他shape或所有backend同样通过 |
| E1-small性能 | 原五种子完整测量与统计的算术结果 | Atlas与ARM64竞品的无条件公平排名、ISA独立倍数 |
| E1-large Atlas拒绝 | 五种子第二次完整JSON请求超过67,108,864-byte上限；逐次请求bytes/hash和软件准入原因仍有效 | OOM、原生ARM64速度或后续训练能力 |
| E2/R初始准入 | 冻结软件格式和预算下的初始请求获准/拒绝记录 | 数值资格、后续Adam请求获准或实际性能 |
| 整作业RSS/冷启动 | 原有采样范围和冷字段范围中的记录 | 排除架构影响的纯训练内存、完整冷启动或容量排名 |

当前A1 v4冻结文件的files映射也绑定上述x86_64 runtime，因此本阶段必须披露同一架构profile：Atlas为ARM64 Python→x86_64 native，Torch为ARM64；后续JAX独立阶段为ARM64 host-config diagnostics。完整epoch、checkpoint选择、预算内一次测试与准确率记录仍按原合同解释；架构影响会改变对墙时、达到质量所需时间和1800秒内训练量的解释，不能将预算截断差异单独归因于学习算法。不得就地改runtime、替换失败seed、把新版ARM64种子与旧版x86_64种子拼接，或借架构修正绕过原once-only test ledger。

Metal加载失败的架构解释是**强静态推断，尚不是已确认底层根因**。原Q0由ARM64 CPU Python执行，记录的clang目标为arm64；冻结training_metal.py构建命令没有显式-arch参数。Atlas原生runner则是单一x86_64。两案例均已生成临时dylib的hash，并在首次gradients调用发生“cannot load native Metal training library”；这种进程/库目标架构组合与加载失败高度一致。但临时dylib已被清理，不能直接对原件执行file/lipo确认；错误未保留底层dlerror，尚不能排除其他加载原因。不能将其称为Metal数值不一致，也不能把默认SSH python3的另一次dryrun架构冒充原ARM64 Q0的编译上下文。

若开展同ISA ARM64对照，应另立runtime和profile revision：保存独立构建产物及SHA、源快照、编译target/flags/SDK、file/lipo/otool和各环境进程架构证据；在新runtime上重做相关真实公开API数值Q0与工作负载资格，再按同一冻结shape、实际共享数组、线程政策、预算和五种子定义独立采集测量。新结果作为完整新revision报告，不能覆盖旧观测或混合新旧种子。涉及A1时还需预先定义新阶段的checkpoint/test政策，保留原ledger，不能自动重复已有测试。

本补充没有运行模型、oracle、训练或测试集，没有读取大型原始数组。原E1-large摘要SHA仍为972938fe6e97a62274ed5dfdd9ecc49ca737107d19daeb79cc199fc86a24302a；原qualification-followup摘要SHA仍为68f118c1cf5cef90d65fdb3c8b1226f4d414232d81d5fdfd206c0a8d189ada56。

本说明使用修正版诊断，原static-diagnosis.json中将默认SSH dryrun误称arm64的因果句已被r2明确纠正；原证据保留，不能混用两个上下文。确认架构与编译target不等于已经获得新的ARM64构建、Metal资格或性能结果。

主要证据及SHA256：

- evidence/h0r-native-load-diagnosis-r1/static-diagnosis-r2.json：6ce46531791e49e24d19688a9f32881e69188d8a615c69dcf7ddca901289f012
- evidence/h0r-native-load-diagnosis-r1/architecture-context-default.json：ff7606c4de0cb8adb8b9b5b2d0b913ebace7e439f72b2ebe27d8efae17d18a37
- evidence/h0r-native-load-diagnosis-r1/architecture-context-cpu.json：b34ae55d544a6e872f6b2fa29279c8c8d7d9fecd3ec6d083c6347b45aaf785a4
- evidence/h0r-native-load-diagnosis-r1/architecture-context-jax.json：5dadae73246d59f21b974ba79b871fd0c8b462989f8db1c75771ceb2fdb7013b

r2还通过raw_evidence映射绑定原file/otool、Rust工具链与runtime盘点。本次逐项复核该映射指向的小型JSON的SHA，并核对A1冻结runtime绑定。没有重跑编译、动态加载探针或任何模型。
