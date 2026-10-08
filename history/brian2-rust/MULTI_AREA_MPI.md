# Multi-Area Cortical Model：MPI 下一阶段适配评估

后续已完成 [完整 32 脑区受限运行](mpi-evidence/full32-population-storage/README.md)：
413 万神经元、241 亿条突触，在四台 Linux / 32 ranks 上跑完 100 ms。
下文保留初始适配评估；实测范围、资源峰值与局限以上述验收记录为准。

2026-09-07。结论：适合作为容量和分布式扩展目标；尚未宣称运行或复现该模型。
先用 FlyWire 验收本轮 MPI 分片/执行优化，再补齐模型适配与通信调度。

## 核对后的模型

官方完整版本包含 32 个脑区、约 4.13 百万神经元和 24.2 billion（242 亿）条突触。
因此“10–20 亿”低估了完整模型规模。缩放 N/K 或选择脑区子集都是官方支持的模式，
但应明确标注缩放参数，不能把缩减网络作为完整模型成绩。
[官方模型与运行模式](https://inm-6.github.io/multi-area-model/)。

模型使用电流型 LIF 与指数突触电流，区域内分层 E/I 微回路和跨区投射。
不能用当前 FlyWire 的电导型方程直接替代；需要导入原始细胞、连接和输入参数，
确定 Ground / Metastable 状态并验证网络活动。
[原始动态模型论文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006359)。

官方参数 dt=0.1 ms，局部 E/I 平均延迟分别为 1.5/0.75 ms，跨区平均延迟由距离及
3.5 mm/ms 的传导速度计算，相对标准差为 0.5。
[默认参数](https://github.com/INM-6/multi-area-model/blob/master/multiarea_model/default_params.py)。
实际连接代码从正态分布重采样延迟，下限为 dt。长的平均延迟并不构成 10–20 ms 的
最短延迟保证；在海量连接中尾部的短延迟不能忽略。
[官方连接实现](https://github.com/INM-6/multi-area-model/blob/master/multiarea_model/simulation.py)。

## 不应预设的结论

- **Brian2 必然 OOM**：容量取决于确切模型、建图方式和可用内存。23/24 每台约 1.1 TiB RAM，
  不能用普通工作站的内存推断它们一定失败。必须分别测前端、编译、运行峰值和 swap。
- **突触数自动变成数百万行 C++**：本仓库原版 `CPPStandaloneDevice.write_static_arrays`
  将数组写入二进制文件；连接数量本身不意味着逐条展开为源代码。很多投影/CodeObject
  可能增加生成和编译开销，但需要实际测量，不能预言编译器 ICE。
- **OpenMP 固定在 4 线程饱和、运行 1 秒需要数天**：这是模型、硬件及后端实现相关结果，
  不是 Brian2 的固定上限。应选择原版最佳已测配置，不能故意用退化的线程数做基线。
- **Isend/Irecv 自动隐藏通信且线性加速**：非阻塞 API 只提供重叠的可能性。
  还取决于最短跨 rank 依赖、计算窗口、MPI progress、网络、负载均衡及同步尾延迟。

本仓库原版证据：`brian2/devices/cpp_standalone/device.py` 的 `write_static_arrays`。
原版 OpenMP 的使用与模型相关性见
[官方计算文档](https://brian2.readthedocs.io/en/stable/user/computation.html#multi-threading-with-openmp)。
已有同类模型的 MPI/GPU 实现可作后续参照，不能宣称跨机运行本身是首创：
[2022 MPI-GPU 多脑区模型论文](https://doi.org/10.3389/fninf.2022.883333)。

## 初步容量估算（估算，不是实测）

以 242 亿条边，仅存 target u32、weight f64、delay u32，即 16 bytes/edge，
原始边数组就需约 387.2 GB（360.6 GiB）。不含 CSR、源/全局边身份、队列、监控、
分配器、临时构建缓冲区；若 delay 使用 u64，则基础数据增加到约 484 GB。

| 分布 | 理想均分边数/节点 | 仅 16-byte 边数组/节点 |
| --- | ---: | ---: |
| 2 节点 | 121 亿 | 193.6 GB |
| 8 节点 | 30.25 亿 | 48.4 GB |

所以“8 节点、每节点 1 亿条边、16 GB 足够”不适用于完整版本。
两台现有 EPYC 的内存值得继续评估，但内存总量充足不代表输入构建、磁盘和运行时间可接受。
分区应权衡入边数、预期投递事件数及跨区流量；各脑区的大小和活动不能视为相同。

## 资源准入

已按用户要求通过 Teleport 只读盘点 23/24/25/81–84。23 根盘当时已用 96%，
不适合直接准备完整模型；24、25、81、83 可作为两到四节点候选，但尚未预留资源。
完整模型启动必须先做受限规模的峰值试跑、重新盘点并验证内存/CPU/磁盘限制及
作业清理机制。详细快照、余量规则和未完成的保护措施见
[RESOURCE_ADMISSION.md](mpi-evidence/rank-local/RESOURCE_ADMISSION.md)。

## 实施与验收顺序

1. **通用运行时**：完成分片输入、可变状态和入边存储，保持全局 RNG/边顺序；
   在完整 FlyWire 上做同工具链、相同 CPU 资源的前后对照和逐字节回归。
2. **多脑区适配**：固定官方代码版本和参数，接入按 rank 生成的 fixed-total 拓扑、
   clipped-normal 权重/逐边延迟和外部 Poisson 输入。MPI 的 fixed-total 分布式生成已实现并进入受限规模验收，
   权重与逐边延迟保持 reference 随机身份；科学参数导入、区域映射仍需补齐。
   实现与实测范围见 [MPI_PROCEDURAL.md](MPI_PROCEDURAL.md)。还需审计 B2IR 的数组预算
   （当前全局显式值预算为 4,000,000）及 binary CSR 单投影 100,000,000 边上限，
   用有容量约束的分片实例支持扩容，不能直接去掉验证。
3. **区域分配与通信**：新增区域到 rank 的显式映射，而非把每个 population 平分到全部 rank。
   从实际跨 rank 边推导最短延迟；安全批处理和订阅路由先通过因果回归，再加入非阻塞重叠。
   必要时把短延迟连接和长延迟连接分开调度，但不能删除或增大原模型延迟来获得加速。
4. **递增规模**：先运行官方缩放版或有明确定义外部替代输入的脑区子集，核对统计量；
   然后扩展到完整 32 区。记录实际突触数、输入构建时间、CPU/RAM、峰值磁盘及完成时长。
5. **对照**：Brian2 C++ 的最佳合理线程配置、单节点 Rust、两节点 MPI。
   跨框架比较需要先对齐模型语义与输入；官方 NEST 实现还应作为科学行为的参照。
   小模型可精确比较，大随机模型需同时报告连接实现差异和预先确定的统计标准。

成功标准是相同网络和精度下的可复现容量/时间收益。完整模型能否完成、何时超过单机，
由实测决定；当前不承诺 8 节点、16 GB/节点或线性加速。

## 后续进展：官方参数与脑区归属

显式脑区/population 归属、Poisson 输入、空投影索引释放及共享内核已接入，
官方参数和 V1/V2 缩放适配的实际编译资源问题见
[MPI_MULTI_AREA_ADAPTER.md](MPI_MULTI_AREA_ADAPTER.md)。上述历史验收限制不代表最新功能上限。
