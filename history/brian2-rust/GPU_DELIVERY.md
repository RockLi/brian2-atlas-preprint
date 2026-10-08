# Metal / CUDA 阶段性交付验收

范围冻结日期：2026-09-09。生产实现基线：`7495df43a`。
本报告围绕已实现功能收尾，不把长期路线图当成无限扩张的验收条件。
当前状态：**本阶段交付完成。** 固定矩阵中的真机回归已收齐，已知失败已按原契约闭环；
[收尾审计](execution-plan-evidence/m1-current/README.md)通过 943 个证据引用、60 个幂函数用例
及 84 份 M1 基准结果，历史对照证据保持原样。

## 固定验收矩阵

下列条目使用已有原始输入、数值门槛和测量方案。已有通过证据直接引用，不重跑历史基准。
“通过”表示指定版本与场景的证据成立，不表示全部 Brian2 模型或所有特性组合兼容。

| ID | 固定范围 | 设备与证据 | 验收状态 |
|---|---|---|---|
| C1 | 基本表达式、类型、原生 Function、调度、拓扑及旧 M1 回归 | [Function 与 M1 修复](execution-plan-evidence/gpu-function-order/README.md)：M1 698 通过 / 276 跳过；L4、A100 各 97 / 33 | 已通过；历史原始失败保留 |
| C2 | 多时钟、延迟、塑性、链接、命名事件、store/restore、queued build 的混合用例 | [混合策略](execution-plan-evidence/composed-policies/README.md)：M3/L4/A100 各 20 / 12；精确恢复、完整坐标和 pending 队列检查 | 已通过；旧 L4 混合模型归档不完整仍见原报告，不伪造其结论 |
| C3 | C1 后变更的 33 个 GPU/Metal 测试文件及 5 个相关回归文件 | 本轮 M1 811 个测试身份：580 通过、6 失败、225 跳过；6 项均为幂函数逐位相等假设，原始结果完整保留 | 6 项已按原函数契约复验通过；580 项不重复运行 |
| C4 | 幂函数的类型转换、动态指数、符号零与不可覆盖的算术故障 | 本轮 M3/L4/A100 各 30 通过、10 平台跳过；M1 使用同一检查亦为 30 / 10 | 四设备已通过；后端源码不变 |
| C5 | 最后生产变更：浮点输入打包及隔离、验证、类型和初始化 | [打包验证](execution-plan-evidence/packing/README.md)：M3/L4/A100 各 100 / 26；[表达式边界](execution-plan-evidence/exprel-range/README.md)：各 66 / 28 | 已通过；对应版本范围明确 |
| P1 | ring 延迟 STDP，N=4096/16384，出度 8，256 ticks，drive=1/16，pre delay 0–7，post delay 3 | [规模对照](execution-plan-evidence/population-scale/README.md)：M3/L4/A100，408 结果 / 3264 字段；本轮补 M1 同输入 | 四设备均通过；M1 新增 84 结果 / 672 字段 |
| P2 | seed-42 随机出度 STDP，N=1024/4096，出度 8/32，256 ticks | [M1 随机图](execution-plan-evidence/gpu-function-order/README.md)、[多后端随机图](execution-plan-evidence/gpu-topology/README.md) | 已完成；小规模 CPU 更快保留 |
| P3 | 4096 神经元、出度 32 的较密 STDP；四种显式 GPU 策略及外部对照 | [位图对照](execution-plan-evidence/bitset-comparison/README.md)：M3/L4/A100，210 结果 / 1680 字段 | 已完成；CUDA 位图五轮配对均改善，默认策略不变 |
| P4 | 固定 seed-42 随机图，4096 神经元、出度 8、1024 ticks，quiet/11.1 Hz 两种活动率 | [低活动率](execution-plan-evidence/activity-stdp/README.md)：288 结果 / 2304 字段；[长时程](execution-plan-evidence/long-stdp/README.md) | 已完成；慢场景及 f64 坐标差异保留 |
| D1 | 显式数值模式、使用入口、限制、复现与原始证据 | 本报告及下面链接；大文件统一位于 T7 | 已完成；使用说明、证据和离线审计均可用 |

各行测试有重叠，不能把数量相加作为独立模型数量。历史通过并不替代较新变更的回归；
最后生产变更已有 C5 三设备验证，M1 的 C3 补齐其后续改动。幂函数收尾只改测试和运行器。

## 已有性能结论及计时边界

每个模型先构建一次，保存完整 bootstrap，再运行一次 warmup 和五轮随机顺序重放。
表中为同一主机上的完整重放中位数，单位 ms：从重置/初始化到完整主机结果数组，
包含各适配器实际需要的进程、读回和 GeNN load/unload 成本。编译、模型导出及证据压缩
不在重放计时内，准备时间在原报告另列。这不是纯 GPU kernel 排名。

| 已验证场景 | 自研 GPU | 同主机 Rust f64 | 解释 |
|---|---:|---:|---|
| M1 随机图，4096 神经元 / 131072 边 | 61.63 | 97.34 | 中型图已有收益；较小随机图 19.59 vs 15.80，GPU 更慢 |
| M1 ring，16384 神经元 | 44.40 | 76.91 | 五轮均更快，范围分离；本轮补齐 |
| M3 ring，16384 神经元 | 67.19 | 85.73 | 五轮均更快，范围分离 |
| L4 ring，16384 神经元 | 34.82 | 216.02 | f32 与该场景 f64 完整检查均通过 |
| A100 ring，16384 神经元 | 32.58 | 181.80 | 同上；不能用跨主机数值推断硬件优劣 |

Rust 的该 STDP 执行器是串行参考；CPU f32 是数值控制组，并非优化过的多核性能上限。
[本轮 M1 完整报告](execution-plan-evidence/m1-current/README.md)给出源码身份、逐轮区间及原始失败闭环。
这些结论足以证明已测中大型场景有可重复收益，不证明 GPU 对所有 CPU 模拟器普遍占优。

三类外部后端都已有同分配对照：Brian2CUDA、Brian2GeNN、直接 GeNN。
原版 Brian2GeNN 的延迟 STDP 语义失败独立保留，修正调度及 f32 factor 版本分别标注；
原版直接 GeNN 在 16384 ring 上未通过完整数值门槛，不能进入该场景的时间排名。
原版在其他合格场景的结果照常保留。完整输入、依赖版本、修改源码和逐轮区间见 P1/P3/P4。

## 支持范围、使用和数值限制

- [使用入口与 CUDA/Modal 环境](CUDA_MODAL.md)、[Metal 文档](METAL_RUNTIME.md)、[功能范围总表](GPU_IMPLEMENTATION_STATUS.md)。
- [数值模式](GPU_NUMERICS.md)：GPU 必须显式 `numeric_mode="float32"`，CPU f64 契约保留。
  不能保证 f32/f64 或不同 GPU 之间的逐脉冲一致；阈值 epsilon 不是兼容修复。
- [执行计划](EXECUTION_PLAN.md)：计划验证、EXPLAIN、后端与数值身份保持可检查。
- [缓存与恢复](GPU_BUFFER_REUSE.md)、[编译复用](GPU_COMPILATION_REUSE.md)、[显式调优](GPU_AUTOTUNE.md)。
  调优和跨激活复用按文档选择，不能假定默认启用。

支持已声明 B2IR 的群体更新、事件、显式连接、summed、延迟、受支持塑性、多时钟、
监控、typed state、Functions、随机输入及 Device 生命周期；具体 eligibility 和边界见范围总表。
GPU 不等于全面 Brian2 兼容。对 subnormal 算术、HH/循环网络长期轨迹已有明确精度边界；
这些模型的失败不计为合格性能结果。冻结 B2IR 的时钟和运行边界限制继续适用。

## 证据与后续工作边界

原始数据、编译产物、失败日志和临时目录位于 `/atlas-storage/0002/brian2-gpu-execution-plan`。
仓库保留小型索引、哈希、审计脚本与报告；各证据页面列出无需 GPU 的离线复核命令。
重放原始云端结果时使用已有 call ID；启动新的 benchmark 会产生新的付费任务。

全面兼容、自动 CPU 回退、跨模型自动调优、GPU 拓扑初始化、NUMA/MPI 和无明确收益的
微优化属于后续工作。当前固定矩阵通过后不自动追加新的规模或组合，不以小模型全面占优
作为本阶段交付条件。任何支持契约内新发现的实际正确性错误仍必须解决并保留修复证据。
