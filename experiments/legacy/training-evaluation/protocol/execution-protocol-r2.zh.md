# 执行协议 r2：正式测试迁至 100.90.28.27

日期：2026-10-04（Asia/Singapore）。本修订落实用户的新指示：MacBook Air 仅运行轻量资格；复杂模型及正式测试转到 `100.90.28.27`。r1 原件、完整覆盖矩阵与全部失败证据保留；本文件对执行位置的规定优先于 [r1](execution-protocol.zh.md)。配套机器计划为 [execution-plan-r2.json](execution-plan-r2.json)。

本机可以继续 Q0、最小 Brian 前向、L1、小型独立 oracle 与准入预检；本机不执行正式 E1–E7、L2 扩展、MNIST/SHD 训练和复杂生物物理性能测试。已有本机诊断数据保留，并注明其资格/探索身份。

远端实测为 `rock-mac-studio-1.local`，macOS14.5/Darwin23.5 arm64，Apple M1 Ultra、20CPU、128GiB RAM、48核GPU/Metal3，可用磁盘约682GiB，无NVIDIA。观测负载1.70/1.45/1.40且有其他常驻进程，不声称机器完全独占。详细原始记录见主评估目录 `environment/remote-inventory.txt`。独立评估根目录为 `/atlas-home/0004/workspace/atlas-training-evaluation/20261004-r1`。

该设备新增为 H0R（远端Apple CPU/Metal），不冒充原 H1 Linux32核或 H2 CUDA；Linux/NVIDIA/真多卡/双机槽位继续pending。正式CPU首先使用1线程并记录资源控制；额外线程/进程梯度只能取实际可获的核心数，M1 Ultra的性能核/能效核放置需另记录。Metal运行时/驱动访问与每个适配器仍需真实资格，物理Metal3存在不自动等于所有框架MPS路径可用。

源码副本、运行时、环境、缓存及输出在远端使用独立评估根目录，禁止覆盖活动项目或共享环境。保存源文件/二进制/adapter/env 的双端 SHA256；本地 macOS 二进制不得直接当作 Linux 运行时。若远端需要编译，使用相同冻结源码和独立构建目录，记录编译器、flags 与生成物哈希，重新 Q0 后才进入正式测试。

CPU 总预留仍为72小时、50GiB可用磁盘，转至H0R；运行前复查可用磁盘/负载并冻结每个适配器环境。实际线程预算不能超过远端可获的物理核。原 H1 的32核仍是待资源匹配的候选梯度，不强行在20核主机超订阅。CUDA及双机配置仍为待获得资源；此次迁移指示不授权付费云资源。

所有数学、输入、五种子、计时、失败分类、19个固定E形状、12个Brian模型分母、质量目标和完成条件继续继承r1。可运行切片须单独 `execution_frozen`；一次 SSH 成功、一次小Q0通过或资源搬迁不等于完整比较完成。
