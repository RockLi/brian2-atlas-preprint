# 外部输入字段物理所有权

新增接口的审查发现：输入组 alias 可能指向选中训练图的物理变量；原转换器会接受
外部表，却由已选内部状态绑定抢先读取，静默忽略该表。14439 actual exit1 chunk73d134
保留 `DID NOT RAISE ValueError` 的反例。现要求外部字段由输入组拥有，避免混淆图内
状态和显式外部输入。修复65704 actual exit0 chunk5ff208，1p。

源码、测试相对前一外部状态阶段仅两处变化，native/kernel/runtime 未变。
冻结98663 actual exit0 chunkf5a1ba：747源码、802资产；外部模块44身份，五个
完整模块249身份。前一248 CPU77476继续等待自己的终态；不会代替修复后的验证。
新CPU回归95379 初始chunk68dc33 已启动，Metal/CUDA尚待验证。原完整目标active，跨主机暂缓，无云作业。

实际 CPU 终态95379 exit0 chunkec54e4：148p101明确硬件skip/249，5完整模块。
外部模块24p20skip/44，包含新所有权反例。audit.py cpu 实际 exit0 chunk76f3a5；
249精确身份包含前248的全部身份与原顺序，747源/802资产、runtime/controller一致。
前248实际77476 exit0 chunka64620，147p101skip；该旧版本不含所有权修复。
当前阶段仅 CPU/MPI 验收，Metal/CUDA 尚待运行。
