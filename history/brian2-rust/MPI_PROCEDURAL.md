# MPI fixed-total：分布式建图与受限双节点验收

2026-09-07，`codex/mpi-cpu`，接续分区存储提交 `4bbd0ae2`。
本阶段已打通 `connect_fixed_total` → MPI recipe → 分布式构建 → canonical f64 执行。
**不是完整 Multi-Area Cortical Model 的复现或性能成绩。**

## 实现

Python 只导出种子、边数、初始化规则及本地神经元数据；不会展开逐边 Python 对象。
每个 rank 负责全局抽样编号的一个互不重叠区间：

1. 首遍仅统计源神经元和目标 owner 的计数。整数 Allreduce/Exscan 给出总入边数与
   各源行的全局前缀，从而恢复 reference 的 source-major 边身份。
2. 第二遍仍只生成自己的抽样区间，每批最多 16,384 个候选连接。Alltoallv 将
   `(source,target,global_edge_id)` 交给目标 owner；接收缓冲区有显式上界。
3. 本地按唯一全局边身份排序，生成入边 CSR。clipped-normal/uniform 权重和逐边延迟
   使用原始全局边编号初始化，运行时的随机突触表达式同样保留该身份。

因此抽样工作为每 rank O(E/R)，没有让每个 rank 扫描全图；本地排序为
O(E_local log E_local)。源索引仍为 O(N_source)，构建临时记录为 16 bytes/local edge，
还有本地目标、原边身份、参数、延迟及通信缓冲区；不能把“分区存储”理解为零额外内存。
目前全局边身份数组在初始化后仍保留，后续可按运行时 RNG 需求缩短其生命周期。

支持 shared 常量、统一延迟、多投影、子组和空 owner；fixed-indegree、区域映射、
可塑性及跨 rank 动态突触前状态尚不支持。B2IR ABI 与独立 reference 执行器未修改。

规划阶段不构图以获得统计值：procedural 入边数为 `None`，最短跨 rank 延迟不宣称已知。
运行报告 `procedural_topology` 的每个投影包含 `rank_stats`，以
`[owned_edges,generated_draws]` 为一组，以及各 rank 的建图/初始化时间。
尚无 procedural 的精确路由矩阵或可用于 lookahead 的跨区延迟证明。

默认 `B2_MPI_MAX_LOCAL_EDGES=10000000`，限制每 rank **累计 procedural 入边数**。
先统计、检查预算，再分配本地边数组；超预算调用 MPI_Abort。该限额不涵盖已有静态图、
参数数量及监控内存，不能替代 cgroup；大模型不得在未完成容量准入时直接调高它。

## 实测

两台 `hk-prod-model-ae02-23` / `24`，同一 Rust 1.86.0、MPICH 4.2.3 ch3:sock。
合成 E/I 网络含 1,000 个神经元、4 个 recurrent fixed-total 投影，clipped-normal
权重及逐边延迟，dt=0.1 ms、运行 12.8 ms。增大连接数时按比例调整权重，以便做
可控的构建/存储试验；这两档不是同一个模型的强扩展对照，也不是科学 MAM 缩放版。

每档独立 reference 一次，MPI 1/2/4 ranks 各一次；共 8 个正式作业。
2 ranks 分布为 1+1，4 ranks 为 2+2，每 rank 一个计算线程。
六份 MPI 完整 results/events dump 均与对应 reference **逐字节一致**，
状态有限且有实际 spike。源抽样数按 rank 均分，入边数加总等于完整网络边数。
两节点的模型/源码/分片/可执行文件身份一致；本地重新生成的 44 个项目文件与
Linux 验收制品也逐字节一致。

| 全局连接数 | ranks | 最大单进程峰值 RSS | 最慢 rank 的建图及初始化 |
| --- | ---: | ---: | ---: |
| 262,144 | 1 | 18.0 MiB | 0.0327 s |
| 262,144 | 2 | 15.0 MiB | 0.0265 s |
| 262,144 | 4 | 12.0 MiB | 0.0158 s |
| 4,194,304 | 1 | 132.5 MiB | 0.5155 s |
| 4,194,304 | 2 | 77.6 MiB | 0.3274 s |
| 4,194,304 | 4 | 42.2 MiB | 0.1569 s |

大档四 rank 的入边数为 `[1048421,1046829,1049748,1049306]`。
输入分片在两档中完全同尺寸：1 rank 33,200 bytes；2 ranks 每份 16,700 bytes；
4 ranks 每份 8,450 bytes。每个可执行文件约 4.31 MB。六个项目的构建进程树
cgroup 峰值约 301.3 MiB，耗时约 17.61 秒。

本表是单次受限试跑，不是多次重复的性能结论。大档仿真/记录时间本次为
单 rank 0.0683 秒、四 rank 0.0519 秒；小档多 rank 明显更慢。
尚未实现通信订阅、异步重叠或 lookahead，不承诺扩大网络或节点数必然线性加速。
[完整报告](mpi-evidence/procedural/report.json) 保存全部数据及资源约束。

## 资源保护

所有文件位于两台独立数据盘 `/data/brick2/brian2-mpi-procedural-20260907`。
结束时实验目录仅约 33 MiB / 27 MiB，含保留的诊断记录；临时 systemd 作业全部退出。

每个构建/reference/控制器/代理作业均由临时 systemd service 承载，实际进程 UID=1000
（rock），设置 MemoryMax=2 GiB、MemorySwapMax=0、CPUQuota=200%、AllowedCPUs=0–7、
TasksMax=64、运行时限及整个 cgroup 的退出清理。控制器和代理是不同 cgroup，
不能把“每作业 2 GiB / 两核配额”说成整台机器总上限；在 23 上二者同时运行。

包装器在执行前核对实际 cgroup 配置和数据盘挂载，检查可用磁盘至少 100 GiB；
正式仿真还核对 MemAvailable，至少保留 64 GiB 主机内存余量。
子进程继承单文件 64 MiB 的 RLIMIT_FSIZE。该文件限额不是目录总量配额，
完整模型仍需要根据实际输出/队列/建图峰值另做准入。

隔离机制不是仅检查配置：先在独立的 64 MiB cgroup 中让测试子进程申请 128 MiB，
它在上限处被终止，记录 `oom_kill=1`；主机和其他任务未受影响。
正式构建/运行共 19 份资源报告均正常退出且 `oom_kill=0`。
加该保护探针共 20 份 cgroup 记录通过审计。

## 验收与下一步

本地原 32 项 MPI 回归加 8 项新测试全部通过（完整 39 项及随后补充的 1 项）；
另有 20 项 CPU 回归和 12 项 Teleport 启动解析测试通过。
新测试覆盖随机权重/延迟、运行时 RNG、共享常量、统一延迟、空 rank、子组、
跨批次构建，以及单投影与累计预算中止。

第一批试跑发现 systemd 会提前展开 `${PMI_RANK}`，导致同节点 time 指标覆盖。
这些数值结果本身仍与 reference 一致，但未用于最终表。加入
`--expand-environment=no` 后重新运行全部 8 项，最终每个 rank 都有独立指标。

完整多脑区模型下一步仍需：固定官方版本及科学参数；区域到 rank 的映射；
外部输入/模型行为的科学验证；神经元实例的紧凑初始化及容量预算扩展。
当前仍受显式值预算与默认 procedural 入边限额保护，未运行 413 万神经元 / 242 亿突触。
更大试验必须继续在数据盘、有资源硬限制的作业中递增推进。

[复现与证据索引](mpi-evidence/procedural/README.md)，
[多脑区科学模型评估](MULTI_AREA_MPI.md)。

## 后续进展：官方参数与脑区归属

显式脑区/population 归属、Poisson 输入、空投影索引释放及共享内核已接入，
官方参数和 V1/V2 缩放适配的实际编译资源问题见
[MPI_MULTI_AREA_ADAPTER.md](MPI_MULTI_AREA_ADAPTER.md)。上述历史验收限制不代表最新功能上限。
