# 动态突触端点参数索引

实现读取由 mutable int32 selector 选择的 canonical 参数bank，涵盖声明constant/普通
Synapses参数和冻结/可训练字段；银行在编译路径前分配，语句内selector改写不会
改变本语句已读取的系数。整数bank保留int32元数据，布尔与整数控制停止梯度；
shared无bank参数保留Python整数/布尔字面量类型，避免GPU将2**24+1折为f32。

开发实际CPU/Cython检查：浮点四组合4p、布尔动态控制1p、整数取模1p、
shared零索引两类型2p。原BitAnd/shared可变索引等非法Brian fixture失败保留。
最终69身份新端点模块、14完整受影响模块729身份准备；通过前不得验收。
冻结的前一版705 CPU回归exec20333及更早full53 exec68504仍各自运行；
新数据寻址不由这些旧源码结果代表。当前无云/NVIDIA作业，原完整目标active，
跨主机暂缓；源/kernel/runtime变更与测试范围都须通过实际terminal和哈希审计。

最终729/14 CPU已启动actual exec82989初始chunk04e5b5；self collection729
精确身份含旧705顺序及新增24，planned已保存。源745/资产800，未终态验收。
当前checkout与该最新快照的两个实现/测试输入应保持同hash。

CPU 完整回归已实际 exit0，729 身份：{'passed':441,'skipped':288}。audit.py cpu 通过，冻结源码、资产及 runtime 校验一致。此项仅验收 CPU/MPI；Metal 尚待实测。
