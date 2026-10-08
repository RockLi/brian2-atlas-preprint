# 标准数学函数与原生导数

已加入 tan、cosh、sinh、log10、expm1、log1p、exprel、arccos、arcsin、arctan、
ceil、abs、sign，并补齐 floor 的标量路径。表达式编译为原生 SSA math 节点，
CPU 与 Metal/CUDA 在各自执行器内完成值计算和 VJP，不调用 Python 数学回调。

expm1/log1p 在 CPU 使用对应稳定基本函数。Metal 没有这些 intrinsic，因此设备
代码对小输入的 expm1 使用 Taylor 多项式，对 log1p 使用补偿对数，避免直接
相减丢失有效位。exprel(0)=1，其导数为 1/2；小输入导数使用级数，大负值及大
正值使用避免消差/中间溢出的等价式。CUDA 翻译沿用相同设备算法。

ceil/floor/sign 使用零路径导数；abs 在零点使用零导数。整型 abs/sign 保持
int32 控制值，包含明确的 int32 包回语义；整数控制不产生浮点状态梯度。
惰性分支只执行所选路径，未选路径内的 log1p 定义域错误不会污染结果；真正
执行到的非有限值仍按既有事务规则拒绝请求。

设备布局仍为 v4r7/v5r9，新增 opcode 52，并要求独立的 `b2_train_math_v1`
能力符号返回 1。新主机在执行前验证能力标识，拒绝旧 GPU 库，避免旧解释器
把未知操作码当作零。默认 target/release 未替换；当前执行器位于
`mpi-evidence/training-standard-math-final-r2-20261004/b2-train`。

从仓库根目录使用本次已验收的执行器：

```sh
export B2_TRAIN_RUNNER="$PWD/brian2-rust/mpi-evidence/training-standard-math-final-r2-20261004/b2-train"
```

## 当前证据与未完成范围

正式回归 **649 passed／301 NVIDIA-only skipped／0 failed**，十个完整模块、950 个身份、
585 个冻结源码/依赖文件。完整 collection/JUnit 身份、源码和独立执行器均已核对。CPU、实际 Metal、本机
MPI 已运行。初轮在旧 MPI dispatch 断言处停止；修复仅涉及该测试文件，完整重跑
该模块和尚未执行的两个模块，另七个无依赖模块按源码/证据哈希保留原结果。

标准数学模块覆盖标量、状态向量、动态动作三条路径的 90 位参考值和解析 loss
VJP、可去奇点、定义域、惰性分支、整型控制及旧库拒绝。新增联合模块将 14 种
函数置于真实 Brian 随机 drift/diffusion、threshold/reset、连续突触和延迟 pre/post
可塑性中，分别验证 Heun/Milstein、冷暖状态、独立固定噪声 full/TBPTT 全权重与
初态差分、2/8 ranks、双样本和 carry/checkpoint。另运行既有方程、多状态、动态、
随机、整数控制回归，保持旧执行路径的完整模块验收。

19 项 Rust 测试及 CUDA 六内核主机 C++ 语法检查继承自同日数学实现阶段；原生源
和执行器哈希一致，不冒充新运行。NVIDIA 实机未验证，跨机器继续暂缓，无云调用。

后续已实现[Poisson 及其率梯度](NATIVE_TRAINING_POISSON.md)、int32 位运算以及
[有界纯函数 AD](NATIVE_TRAINING_PURE_FUNCTIONS.md)。新版完整回归仍在进行。
完整 int64/timestep、任意 Python／目标代码回调及其他未列入支持范围的 Brian
语义仍待补齐；标准数学组合通过不等于任意 Brian 语义全部实现。

完整证据和续接入口：[联合验收](mpi-evidence/training-standard-math-final-r2-20261004/README.md)。
数学实现过程和初检：[实现记录](mpi-evidence/training-standard-math-20261004/README.md)。
