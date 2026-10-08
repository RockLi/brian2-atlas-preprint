# 显式外部状态输入

动态转换可用 `external_state_inputs` 将输入组的物理字段绑定到 Brian `TimedArray`：

```python
bundle = lower_brian_training(
    network, input_group=inputs, layers=[hidden, output], dynamic=True,
    external_state_inputs={"drive": drive_trace, "bias": bias_trace},
)
```

字段必须是输入组的可变 float、int32 或 Boolean 物理数组。时间表使用相同单位；逐元素字段要求二维
`time × canonical physical element`，标量/shared 或只有一个元素的字段也可用一维表。
转换冻结表值，不保留对可变 Python 数组的执行依赖。表中全部值须有限，大小、
时间分辨率、参数银行和动作内存仍受现有 native 预算检查。

物理变量必须由输入组拥有；指向选中训练图状态的输入 alias 明确拒绝，避免由内部
状态绑定抢先读取而静默忽略显式表。该拒绝有修复前反例和修复后实际测试。

pre/post 路径、连续/事件驱动突触方程、summed 表达式和选中对象的定时代码通过物理
变量身份读取该表。固定索引及由已选训练图 int32 状态控制的运行时索引链均可选择列。
一个代码块先按物理索引快照读取字段；同块随后改写 selector 不改变已读取的值，
下一动作读取更新后的 selector，保持 Brian Cython 的语句语义。整数列路由使用冻结
银行和边界校验，控制与选择停止梯度；字段表值仍累加输入 VJP。
读取发生在消费者的 Synapses/Group
clock 上，遵循原生 TimedArray 采样、边界保持和精度规则。这不是自动运行输入组的
信号生成器：异步消费者的 clock 可能领先输入组。调用者须提供该采样契约所需的
表值；输入脉冲仍单独通过 `inputs` 传入。

输入组的标准定时代码只有在全部物理写入均属于显式供给字段时才可作为外部信号
生成器排除出训练图。训练图中对此字段的 pathway、summed 或定时代码写入会被
拒绝。无显式契约的外部可变字段继续拒绝，不能降为其初始常量。

时间表是冻结 optimizer 银行，但 native 返回每个表元素的输入 VJP；离散时间和列
选择停止梯度。现有 `trainer.update_timed_input(bank, values)` 在序列边界替换表值，
不推进时钟、随机序列、optimizer step 或运行时物理状态。checkpoint 保存更新后的
表；新训练器恢复后 `initial='carry'` 沿原绝对时钟继续。只读与失败操作不推进游标。
`bundle.provenance['timed_inputs']` 提供银行和形状，`external_state_inputs` 提供字段、
物理列、消费者 clock 与采样方式。

普通 TimedArray 在 batch 中共享；按样本输入见文末 BatchTimedArray 扩展。
非 int32 整数和任意可变 Python 对象仍有显式边界。

选中神经元的浮点 linked 字段现可读取同一外部表，包括积分、动态threshold、reset
和refractory表达式。`external_neuron_aliases` 标记其矩形ABI占位槽：这些槽不承载
外部输入的运行时物理状态，初态伴随为零；数据梯度由时间表银行返回。训练图对
此类字段的reset/regular/pathway写入均明确拒绝。固定与运行时索引仍使用原先的
代码块前快照，不能因同块改写selector而提前重新读取外部字段。

`tests/test_training_external_state_inputs.py` 使用独立物理递推、所有表元素/参数与初态
有限差分、原始 Brian Cython 信号生成器、Heun 乘性噪声、异质延迟、full/TBPTT、
MPI2/8、替换/carry/新训练器恢复和无效契约检查。最新接口模块24项 CPU 实际通过；
五完整模块249身份冻结回归148通过、101明确GPU硬件跳过，源码与实际终态审计一致。
见 `mpi-evidence/training-external-state-inputs-owner-20261006/`；开发与修复前失败仍保留。
Metal/CUDA 尚未验收此新接口。

运行时索引扩展详见 `mpi-evidence/training-indexed-external-inputs-20261006/`。独立
连续/事件驱动递推、全梯度、原始 Cython、MPI、失败原子性 CPU 开发23项通过；
六完整模块312身份冻结回归171项通过、141明确硬件跳过，源码与实际终态审计一致。
这是当前运行时索引版的 CPU/MPI 验收；Metal/CUDA仍待实际运行。

神经元外部linked扩展记录于 `mpi-evidence/training-external-neuron-links-20261006/`：
开发45项CPU物理/梯度/实际Cython与MPI通过，七完整模块417身份已另行冻结运行；
其实际终态与精确身份/源码审计已通过：216项通过、201明确GPU硬件跳过。新增模块
45项CPU通过、60硬件跳过。新linked前端Metal/CUDA仍待实际运行，旧GPU阶段不能替代。

int32/Boolean 外部字段已接入。int32 原表在同一个冻结银行编码为相邻 high16/low16
列，通过现有原生 timed_parameter、integer_cast、left_shift 和 bit_or 重组，精确
保持 16777217、2147483647、-2147483648；Boolean 原表只允许0/1。输入离散字段
停止梯度，浮点物理状态和训练参数的 VJP 保持原有契约。神经元 linked、连续随机
Heun、threshold/reset/regular 和突触 pre 路径使用同一消费者clock/索引快照机制。

原物理 shape/dtype/银行/编码见 bundle.provenance['external_state_inputs']['sources']。
边界更新使用 trainer.update_external_state_input(sources['code'], original_values)，
它检查精确 int32/Boolean 和原形状，单银行原子替换；不能把未经编码的整数原表
直接写进内部 high16/low16 银行。checkpoint 保存实际编码值，carry/恢复不重置时钟。
外部 int32 已可作为另一个 linked 地址的根 selector；与已选图的 int32 状态一样，
可控制外部列和 selected canonical 状态的间接读写。

完整新增模块开发22通过、24明确硬件跳过，包含乘性随机Heun的全梯度、实际Cython
固定normal重放、所有prefix整数精确状态、shared一维表、MPI2/8、替换/恢复/失败
回滚。最新八模块463身份、750源码已完成CPU/Metal实际终态与严格审计：CPU238通过、
225硬件跳过；Metal353通过、110 CUDA-only跳过。新增模块CPU22通过、24硬件跳过，
Metal34通过、12 CUDA跳过。不能替代CUDA实机。见 training-discrete-external-inputs-20261006。

外部整数根索引通过只读 int32 物理缓存进入原生间接读写。缓存读取发生在每个使用
它的动作 clock，采样使用拥有对象的消费者 clock pending time；两者在异步 pathway
中可能不同。缓存与外部 linked 矩形占位槽分开，整数字段仍零VJP/只读，间接写回
不读取占位槽的0值。source初始整数即使越界也不会被静默冻结为控制值，实际采样
后的索引越界才按原生规则拒绝，失败不会提交状态或clock，修复表后carry可继续。
新sampling动作前缀平移全部delay路径/事件编号；原待到达队列和delay更新/恢复已
通过原始Cython与固定normal验证。源生成器仍须遵循显式消费者采样契约。

training-external-selector-roots-20261006保存九完整模块527身份CPU262通过265硬件
跳过的实际终态及审计。Metal完整运行393通过130CUDA跳过4错误文字断言失败；仅
修正测试matcher后4个原失败身份实际复验通过，合并审计397通过130CUDA跳过。
全部生产代码未因测试matcher修正而变化，AST证明其它物理/失败回滚断言未改变。
原失败和逐身份来源保留，不能伪装一次完整GPU exit0；最新CUDA仍未实测。

按样本扩展：`BatchTimedArray(values, dt=...)` 供给 `(sample,time,physical element)`
或 shared 标量的 `(sample,time)` 表，可与普通共享 TimedArray 混用。所有按样本表
须有同一固定batch数；carry和custom initial保留原行编号，不能隐式重排样本。
新native只读、detached整数sample_index初始化为行号，参与现有精确列选择；直接/
间接写和迁移均拒绝。float银行梯度通过 `external_state_input_vjp(source, gradient)`
还原原表shape；int32/Boolean返回零。边界更新仍用update_external_state_input并保持
原shape/单位/编码，恢复后沿原绝对clock继续。该描述符不自动执行Brian信号生成器。

`training-batch-external-inputs-20261006`记录新隔离native和Rust50实际通过、78项
新增身份及十完整模块605身份CPU/Metal冻结回归；完整实际终态与严格审计均已通过：CPU296通过309硬件跳过；Metal453通过152
CUDA-only跳过。新模块CPU34通过，Metal56通过。最新版CUDA尚未实测。
