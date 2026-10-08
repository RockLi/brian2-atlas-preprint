# NumPy 回调的调用点写回

2026-10-08 事件调用者返回constant别名：在实际优化语句中追踪捕获引用、派生副本与逐语句重新加载，只有真实写入才提升父字段运行状态；后续独立选择读取观察当前canonical捕获值。原始反例实际exit1；首轮两项通过后captured系数读取拒绝失败，现已接通，四项CPU/Metal/MPI前向恢复实际exit0。VJP首轮CPU alias通过，Metal rebind参考漏掉Brian的tmp+=.1优化，代码实测确认后修正；13项全bank/连续初态、Metal/MPI TBPTT、副本及旧scalar/编译边界实际exit0。31完整模块/21同版抽查准备冻结，尚未整体验收。一般多元素whole-selected混合、可变delay位置重建、神经元跨列、guard/replay及最新CUDA仍待完成，整体目标active。详见 mpi-evidence/training-event-returned-constants-20261008/README.md。

整体返回阶段验证：22项冻结抽查全部实际wait0、无skip，XML/request/实际collection同序，851/907哈希一致。30完整模块已启动CPU→Metal串行回归，预计每端3503身份，未取得完整终态。详见 mpi-evidence/training-selected-capture-returns-20261008/README.md。完整目标active。

2026-10-08 整体捕获返回：每个到达保留独立 presence，按队列/源顺序求位置与长度，返回 alias 先保留整列修改再映射选择写回，单元素返回与选择数组混合计算已接通，长度错误原子失败。前向/恢复/形状错误三项实际 exit0；VJP首轮实际 exit1 为参考遗漏初始队列反事实门，补齐后 CPU/MPI 全 bank/连续初态与 Metal/MPI TBPTT、singleton 恢复三项 exit0。广播/旧步长VJP/编译边界12、银行实时 carry/restore 与 Metal 域2、mutable-delay 明确边界1各 exit0。851 源码/907 资产30完整模块冻结，22同版抽查执行中，完整回归尚未启动，未整体验收。一般多元素 whole-selected 混合、可变delay到达位置重建、神经元跨列、guard/replay和最新 CUDA 等仍待完成，整体目标 active。详见 mpi-evidence/training-selected-capture-returns-20261008/README.md。

重复索引验证终态：18 项同版 Metal/MPI 全部实际 wait0，无 skip，XML/request/实际 collection 同序，850/906 哈希一致。29 完整模块 CPU/Metal 回归已启动，尚未整体验收。详见 mpi-evidence/training-repeated-capture-views-20261008/README.md。完整目标 active。

步长／反向视图抽查终态：28 项冻结 Metal/MPI 全部实际 wait0，无 skip，849/905 前后哈希一致。旧快照完整回归由新增重复索引版本覆盖，不再重复启动；它仅作为阶段证据保留。

2026-10-08 可写重复索引事件捕获：zero-stride 逻辑列共用一个物理状态单元，整列结果缓冲后按列顺序写回；新增逐列不同增量、int32 溢出、Boolean 停止梯度及恢复测试。开发三项原始/恢复和两项全参数/连续初态 VJP 实际 exit0，包含 Metal/MPI 与 TBPTT。新快照 850 源码/906 资产、29 完整模块已冻结，18 项同版抽查正在执行，完整回归尚未启动。神经元重复捕获在整列调度完成前显式拒绝；whole-selected 返回、神经元跨列、guard/replay、最新 CUDA 等仍待完成，整体目标 active。详见 mpi-evidence/training-repeated-capture-views-20261008/README.md。

部分视图v2：首轮Metal/MPI实际wait1，19项通过后 scalar readonly-first 两到达应在第二次立即报错，旧首步成功断言错误。续跑首步改单到达，增加双到达同批次错误独立测试；CPU/Metal/MPI四项exit0通过。旧快照未启动正式作业。v2已冻结846源码/902资产25模块，29项同版Metal/MPI全部实际wait0通过，无skip，XML身份集合完整且哈希一致；collection实际wait0且XML同序；完整双端各2712身份同序回归已启动，未整体验收。详见 mpi-evidence/training-partial-capture-views-v2-20261008/README.md。完整目标持续进行。

2026-10-08 部分事件捕获视图：连续同dtype子视图映射实际runtime单元，constant部分修改提升所属字段；只读银行切片保留父银行索引和carry/restore实时值。外部重叠视图按地址合并，整列符号解释保留跨列先写后读。原始/恢复、偏移域/银行续跑、partial及overlap全bank/连续初态VJP/full/TBPTT、非零共享初态和编译/typed回归，开发3/3/4/3/3/12各实际exit0，范围有交集。846源码/902资产25模块已冻结，28项同版Metal/MPI正在执行，完整回归未启动，未整体验收。详见 mpi-evidence/training-partial-capture-views-20261008/README.md。whole-selected混合广播/返回、更多view布局、神经元跨列、guard/replay、最新CUDA及全目标组合仍待完成；跨机器暂缓，目标持续进行。

2026-10-08 不同长度事件捕获：按真实列分配，独立数组和一元素广播已接通；认证效果解释器追踪捕获/派生长度，拒绝不兼容运算和原地扩形。原始 array/vectorised 开发两项、修正实际 FIFO 后五项、CPU全bank/连续初态VJP四项、CPU/MPI末列域/空批次原子性六项、编译边界十一项分别真实通过，部分范围重叠。delayed TBPTT 非零断言误用已修，full/TBPTT 两项CPU/MPI复验exit0；旧冻结候选未启动。v2已冻结841源码/897资产、二十模块；27项同版Metal/MPI实际wait0全部通过，哈希一致、XML身份集合完整；额外collection实际wait0，XML顺序与真实collection同序，请求顺序被pytest重排。完整双端各2040身份同序，已启动回归，未整体验收。详见 mpi-evidence/training-mixed-capture-lengths-v2-20261008/README.md。whole-selected混合广播、partial views、guard/replay、最新CUDA与完整目标组合仍待完成；跨机器暂缓，目标持续进行。

2026-10-08 神经元共享捕获：积分/阈值/reset 使用权限独立的逻辑名、相同存储的 canonical 状态，reset 整列捕获别名已接通。独立 NumPy/梯度首轮 Euler、RK2、RK4、阈值四项通过；reset 缓存比较与域测试接口两处 fixture 失败保留，修正后 reset 全参数/连续初态 VJP 与八项 CPU/MPI 域/空 reset 原子性共9项真实 exit0。前向恢复完整新模块40项CPU/Metal通过20项CUDA跳过，实际exit0；838源码/894资产六模块已冻结，十六项同版 Metal/MPI 全部实际 wait0，通过全参数/连续初态VJP、TBPTT、域/空 reset 原子性及恢复，XML身份同序且前后哈希一致。完整双端回归已启动，尚无完整验收。事件共享视图旧冻结20项同版 Metal/MPI 抽查已实际 wait0 且835/891前后哈希一致，仅证明旧快照。更广 view/shape/guard/replay 与最新 CUDA 待完成；跨机器暂缓，整体目标持续进行。详见 mpi-evidence/training-neuron-shared-captures-20261008/。

2026-10-08 共享capture视图v3：exact readonly/mutable共享canonical状态，symbolic readonly wrapper观察最新可写source且禁止readonly写，注册顺序独立；物理h与array/vectorised/scalar FIFO写回接通。Scalar winner无write丢失mutable输出真实失败已修。CPU前向/延迟/恢复20、物理写回4、全VJP2、Scalar2、编译/注册11、typed3，各自真实exit0。835源/891资产十二完整模块冻结，二十项同版抽查与完整CPU/Metal执行中，未整体验收。详见 mpi-evidence/training-shared-capture-views-v3-20261008/README.md。更广view layouts/mixed shapes/guard/replay/neuron binding与最新CUDA仍待完成，goal active，跨机器暂缓。

只读批次v3结果：二十四项同版CPU/Metal/MPI真实wait0全部通过，无skip，832源/888资产前后哈希一致；九完整模块双端各1100身份同序，正式执行中未整体验收。当前额外共享mutable/readonly视图guard新完整模块2项live exit0通过，防止注册顺序导致storage分裂；该单生产guard与新模块不属于v3冻结范围，完整共享视图仍待实现。详见 mpi-evidence/training-batch-readonly-captures-v3-20261008/README.md。整体goal active。

2026-10-08 只读批次捕获v3：current typed银行整列eager检查、empty跳过、carry后银行替换及restore原子性接通；外部readonly flags允许读取并拒绝真实写，非银行只读用途字段无写输出时保留whole-eager每列检查。开发定点2+3+3项CPU/Metal/MPI真实exit0通过（各自范围）；前两版delay optimizer fixture失败已修为3-tick前缀，旧v2双端child wait=-15、controller/supervisor241明确停止，无整体验收。832源/888资产九完整模块冻结，二十四项同版抽查及完整CPU/Metal回归执行中。详见 mpi-evidence/training-batch-readonly-captures-v3-20261008/README.md。readonly物理状态视图alias、mixed shapes、guard/replay等与最新CUDA仍待完成；整体goal active，跨机器暂缓。

批次v4验证结果：十六项同版CPU/Metal/MPI真实wait0全部通过，无skip，829源/885资产前后哈希一致，当前829源码匹配冻结；包含域敏感中间写回及单/双调用延迟/TBPTT全bank/连续初态VJP。八完整模块双端各2129身份同序、正式执行中，尚未整体验收；整体goal active。详见 mpi-evidence/training-batch-event-captures-v4-20261008/README.md。

2026-10-08 批次捕获v4：array/vectorised presence+整列更新+selected写回接通；修复向量化两次调用间raw物理存储读取，persistent statements拆stage。域敏感真实反例修复后CPU/Metal MPI2两项通过；两次调用/延迟/TBPTT全bank/连续初态两项通过，真实exit0。v3十二项同版抽查wait0仅限其旧范围。v4冻结829源/885资产八完整模块，已启动CPU/Metal正式回归及十六项同版抽查，未整体验收。详见 mpi-evidence/training-batch-event-captures-v4-20261008/README.md。readonly/混合shape/guard/replay等与最新CUDA仍待完成，整体goal active，跨机器暂缓。

批次捕获v3：已补齐独立参考的真实第二stage队列初态门控VJP，失败槽定点CPU全bank/连续初态真实exit0通过；v1/v2两次wait1失败证据保留，未启动其正式作业。v3冻结829源/885资产、八完整模块，十二项CPU/Metal/MPI抽查执行中，未整体验收。详见 mpi-evidence/training-batch-event-captures-v3-20261008/README.md。

批次捕获验证更新：初版CPU16/Metal2项前向通过后初态VJP失败，已定位fixture默认detach_reset与独立参考不一致；v2明确False并重新冻结829/885，十二项同版抽查执行中，尚未完成验收。失败日志和wait1 receipt完整保留。详见 mpi-evidence/training-batch-event-captures-v2-20261008/README.md。

2026-10-08 批次事件捕获：新增 detached presence 阶段，非空批次整列更新一次，随后 selected own-state 写回；array/vectorised、partial/empty、延迟、恢复、多次调用、域检查及全 VJP 测试已加入，CPU 开发与冻结 Metal/MPI 抽查运行中，未整体验收。829源/885资产八完整模块冻结。上一版 reset 九模块双端已正式 wait0/audit0：CPU294通过540合法跳过、Metal564通过270CUDA跳过，两端834身份及824/880哈希完整，仅限其旧冻结范围。详见 mpi-evidence/training-batch-event-captures-20261008/README.md。readonly/guard/replay/混合shape等及最新CUDA仍待完成；目标 active、跨机器暂缓。

2026-10-08 pathway.delay捕获：路径拥有的delay绑定canonical queue/runtime storage，避免错误constant owner与独立external快照。标量事件支持共享单单元和逐边数组；增加原生captured_shared_delay标记，保持旧直接scalar写入检查，标记随rebuild与checkpoint保留。原始单次/分段运行、已有排队到达、全bank/连续初态VJP/full/TBPTT及恢复，66+16+4开发测试通过、真实exit0。新原生运行器离线真实wait0编译且哈希固定。828源/884资产六完整模块已冻结，CPU/Metal360身份同序，完整回归运行中；14项同版Metal/MPI抽查真实wait0全部通过且哈希一致。详见 `mpi-evidence/training-captured-path-delays-20261008/README.md`。批次array/vectorised捕获、guard/replay/混合shape与reset readonly、更多dtype/Brian语义和最新CUDA等仍待完成，完整目标active，跨机器暂缓。

2026-10-08 标量事件捕获：FIFO每项调用一次回调，原生单事件动作更新全部capture列；空批次跳过，真实scalar显式写回覆盖capture别名，但所有列的域检查保留。跨物理电压alias、constant提升、typed离散状态、readonly实时优化器bank、固定延迟、全部bank/连续初态VJP/full/TBPTT及恢复，开发8+32+4+12+2项通过，各真实exit0。827源/883资产四完整模块冻结双端回归306身份同序，13项Metal/MPI新旧模式抽查真实wait0全部通过且哈希一致；当前四模块双端正式wait0/audit0：CPU110通过196合法硬件跳过、Metal208通过98CUDA跳过，306身份完整同序且哈希一致，仅限其冻结范围。捕获pathway.delay诊断仍在canonical owner检查拒绝，待接通路径存储。上一版reset九模块CPU已真实wait0/audit0：294通过540合法硬件跳过，834身份完整且哈希一致，仅证明其冻结范围，Metal仍执行。详见 `mpi-evidence/training-scalar-event-captures-20261008/README.md`。array/vectorised批次捕获、guard/replay、更多shape/reset readonly、动态delay捕获验收及最新CUDA等仍待完成，完整目标active，跨机器暂缓。

2026-10-08 reset 整列捕获：独立 closure 数组写入在空/部分/完整事件选择均执行，选中变量保持 NumPy 副本及写回顺序；加入捕获电压别名、跨组常量、全部 bank/初态 VJP、full/TBPTT 和恢复。开发 CPU/MPI 36 通过，电压别名/形状边界 13 CPU/Metal/MPI 通过，真实 exit=0。返回别名 CPU 前向/恢复32、VJP64通过；只读捕获完整模块193 CPU/Metal/MPI通过96CUDA跳过。824源/880资产九完整模块已冻结并启动本地双端回归，尚未整体验收；同版11项Metal/MPI真实wait0全部通过且哈希一致。电压别名单独VJP追加测试24 CPU/Metal/MPI通过12CUDA跳过，exit0（另一个live开发模块，未加入九模块冻结范围）。typed reset 24 CPU/Metal/MPI通过、返回别名 Metal96通过，均真实exit0。只读捕获相关常量完整回归124通过62CUDA跳过，exit0。已确认失败的旧 mutable-capture v1/v2作业已停止并保留诊断；不存在其整体验收成功。详见 `mpi-evidence/training-reset-captures-20261008/README.md`。事件捕获选择、reset只读/混合广播等、最新CUDA及完整目标仍未完成，目标active，跨机器暂缓。

2026-10-08只读物理捕获：绑定实时canonical optimizer bank，避免carry继续读
旧快照；浮点全bank/初态VJP/full/TBPTT、优化器更新与恢复，初版完整模块
84CPU/MPI通过168硬件跳过；typed bank替换/精确奇偶/拒绝修改25CPU/Metal/MPI
通过，真实wait=0。当前完整模块和822源/878资产33模块冻结回归运行中，
双端8332身份同序；25项同版Metal/MPI抽查真实wait=0全部通过，哈希一致。
旧mutable-capture v2回归发现integer/Boolean/
private常量失败已确认局部bindings遮蔽；当前定点3通过，完整相关模块
重测中。旧v2两端实际wait=-15停止，无整体验收成功。详见
`mpi-evidence/training-readonly-captures-20261008/README.md`。返回别名修改发现、
reset/event选择、广播/标量等仍需完成，整体目标active。

2026-10-08捕获声明constant：认证符号解释器发现真实捕获写入并提升canonical
持久状态，浮点初始bank VJP/离散精确初态保持一致；完整物理身份解决动态
Synapses返回不同NumPy视图。浮点84CPU/MPI通过168硬件跳过；typed/probe25、
只读外部捕获但修改借用参数24CPU/Metal/MPI通过，真实wait=0。该冻结版完整新增模块217CPU/Metal/MPI通过108CUDA跳过，真实wait=0；
821源/877资产32模块冻结回归仍运行中；双端8043身份同序，28项
同版Metal/MPI抽查真实wait=0全部通过，哈希一致。
详见 `mpi-evidence/training-captured-constants-20261008/README.md`。reset/event
选择、只读captured constant bank绑定、广播/标量等仍待完成；整体目标active。

2026-10-08连续积分器/阈值捕获：神经元隐藏上下文与连续Synapses已接通，
共享ndarray身份跨神经元层/突触/定时回调合并，实际Brian v物理别名和重复
阈值写目标已修复。原始NumPy生成码、全bank/连续初态VJP/full/TBPTT及恢复，
早期CPU/MPI神经元42和突触36通过；共享8、物理别名8CPU/Metal/MPI实测通过，
实际wait=0。该冻结版三完整模块重测224CPU/Metal/MPI通过108CUDA跳过，真实wait=0；
共同820源/876资产31模块冻结回归仍运行中，
双端7718身份同序；21项同版Metal/MPI抽查真实wait=0全部通过，哈希一致。详见
`mpi-evidence/training-capture-equations-20261008/README.md`。reset/event选择
语义、captured constant提升、广播/标量及更多语义仍待完成；整体目标active。

2026-10-08持久捕获v2：Boolean初态ABI序列化已修复，int32溢出/Boolean
停止梯度实际覆盖。当前完整捕获模块及认证边界61CPU/Metal/MPI通过22CUDA
跳过，真实wait=0，源码哈希匹配。共同818源/874资产29模块已冻结，11项
同版Metal/MPI抽查真实wait=0全部通过且哈希一致；v2整模块已启动，
v1整模块仍执行。详见
`mpi-evidence/training-mutable-captures-v2-20261008/README.md`。integrator/
threshold/reset/event捕获自动绑定、广播/标量等仍需完成，整体目标active。

2026-10-08持久Python数组捕获：认证描述符保留ndarray身份，绑定native持久
状态；neuron/Synapses whole-array定时回调已自动分配并共享canonical cells。
原始Python/Brian、全部连续初态VJP/full/TBPTT及恢复，完整新增44CPU/Metal/MPI
通过18CUDA跳过，真实wait=0。冻结818源/874资产、29模块双端7448身份同序，
9项冻结Metal/MPI2真实wait=0全部通过，完整回归仍运行。追加typed实測发现
Boolean初态JSON类型已修复，typed8CPU/Metal/MPI通过4CUDA跳过；当前
修正版完整模块与认证边界运行中，源码与冻结版本不同。详见
`mpi-evidence/training-mutable-captures-20261008/README.md`。积分器/阈值/事件
捕获自动绑定、广播/标量与更广语义仍待完成；整体目标active。

2026-10-08 时间输入与持久常量修改组合：TimedInputRegistry 已接入隐藏修改发现，
覆盖 neuron/Synapses、borrowed/private、1D/2D非对齐采样及加性噪声。新模块
以真实 NumPy 生成码、TimedArray NumPy 实现和独立噪声重放对照全部bank
（含冻结时间表）及连续初态VJP、full/TBPTT与恢复；完整新增模块80CPU/MPI通过，160硬件跳过。
共同817源/873资产、27完整模块双端7377身份同序收集。16项冻结
Metal/MPI2抽查真实wait=0全部通过且哈希一致；完整CPU/Metal尚在执行。详见 `mpi-evidence/training-timed-constant-effects-20261008/README.md`。
可变Python捕获、更广Brian语义和最新CUDA仍待完成；目标active，跨机器暂缓。

2026-10-08积分器/阈值constant隐藏修改：从实际生成块识别borrowed
canonical常量数组写回，保留scalar prefix优化器临时量。五种神经元与
突触连续积分器、阈值、full/TBPTT全部bank/连续初态VJP和恢复，完整
新增模块90CPU/MPI通过180硬件跳过；原始NumPy生成码独立对照。
生成临时变量解析与_lio缺失的实际诊断失败已修复；三个旧完整模块
先前380CPU/MPI通过752硬件跳过，当前更广改动的相关回归执行中。
共同816源/872资产二十四完整模块双端6986身份同序收集，21项同版Metal/MPI2抽查全部真实exit0通过，哈希一致；完整回归执行中。详见
`mpi-evidence/training-constant-equation-effects-20261008/README.md`。
更广constant与时间输入/噪声组合、可变Python捕获和更多数组操作及
最新CUDA仍需完成，完整目标active；跨机器暂缓。

2026-10-08持久可变constant数组：认证效果编译器识别定时回调真正修改的
canonical声明，浮点初始optimizer绑定/状态VJP、精确离散初态及停止梯度
已接通；私有副本不提升。积分器、阈值与动态突触事件消费者及恢复已有
54CPU/MPI完整开发通过108硬件跳过；两轮整数参数初态准入真实失败保留。
命名SDE组合/精确离散比较最终完整开发62CPU/MPI通过124硬件跳过；旧回归执行中。共同815源/871资产
十九完整模块双端5153身份同序收集，完整CPU/Metal及23项Metal/MPI2同版
抽查真实exit0全部23通过，哈希一致；十九完整模块仍在执行。详见 `mpi-evidence/training-mutable-constant-effects-20261008/README.md`。
其他constant效果调用点、可变Python捕获、更多数组操作与最新CUDA尚待
实现或实测；完整目标active，跨机器暂缓。

2026-10-08 typed事件系数重放：int32/Boolean数组与共享标量类型已接通，
私有数组别名、溢出、标量重绑定、16777217奇偶控制、延迟与全VJP/恢复
已有独立原始Brian验证。完整新增模块48CPU/MPI通过，静态索引副本修复
后的扩展98CPU/Metal/MPI通过49CUDA跳过。二/三级索引保留后端精度边界
案例后168CPU/Metal/MPI通过84CUDA跳过；静态/动态混合映射新模块执行中。
旧nested v2完整回归发现阈值精度分岔及int64静态索引误捕获，真实失败保留，
两端实际pytest wait=-15停止，无整体验收成功。修正版共同814源/870资产，
十六完整模块双端4916身份同序收集，完整回归执行中；23项Metal/MPI2同版抽查全部真实exit0通过，哈希一致。详见
`mpi-evidence/training-typed-event-replay-20261008/README.md`。
共享可变常量/捕获与更广数组语义及最新CUDA仍需完成，目标保持active；
跨机器按用户要求暂缓。

2026-10-08多级定时回调索引：二级/三级读链的最终局部选择器快照写回已接入，
标量fallback使用当行局部索引，ufunc.at逐语句重新读取。根与中间索引修改、
全部bank/连续初态VJP、full/TBPTT、续跑恢复和各层越界原子性，完整新增
模块82CPU/MPI通过164硬件跳过；相关三模块220通过392跳过。v1实际失败
保留，参考变量顺序按原始create_runner_codeobj排序修正；CPU/Metal MPI两项
定点实测通过，完整新增模块再次82通过。v1完整作业实际pytest wait=-15停止，
无验收成功。共同v2冻结813源/869资产十三模块3777身份双端同序收集，
17项同版Metal/MPI2抽查真实exit0全部通过，哈希一致；完整CPU/Metal仍执行中。详见
`mpi-evidence/training-nested-regular-effects-v2-20261008/README.md`。
共享可变常量/捕获、typed事件重放剩余限制及最新CUDA实测仍待完成，
完整目标保持active，跨机器暂缓。

2026-10-08定时 ufunc.at 回调更新：按原始 NumPy 逐语句读取、RHS 求值、
有序重复索引写回执行；临时数组的借用、私有别名和重绑定保持真实语义。
float/int32/Boolean 与混合 dtype 逐元素运算后转换、整数溢出、命名噪声、
refractory 副本、空掩码标量域错误原子性已接入。开发完整新增模块145
CPU/MPI通过288硬件跳过，最终同版Metal/MPI2抽查21通过，真实exit0，
哈希一致；抽查名称错误的exit4记录保留。812源/868资产十二完整模块
3531身份双端同序收集成功；CPU真实exit0/audit0，1187通过2344硬件跳过，Metal完整回归真实exit0/audit0，2359通过1172CUDA跳过。后续多级写索引修正
正在开发验证，尚未纳入该冻结版；共享可变常量/捕获及最新CUDA仍待完成。
详见 `mpi-evidence/training-regular-ufunc-effects-20261007/README.md`。
完整目标保持active，跨机器按用户要求暂缓。

`lower_brian_dynamic_training` 现将选中完整 NeuronGroup 的 `run_regularly`
canonical 浮点数组回调写回编译为原生动态动作。转换读取并核对实际加载的 Python 函数体，
不会调用用户回调；运行及反向传播使用原生 CPU/Metal/CUDA IR。

2026-10-07首次完整回归发现事件无guard局部Store误拒绝，已修正，
原先135失败身份逐项重跑通过；相关147CPU/MPI检查通过。最终v4
十一完整模块3098身份两端exit0/audit0：CPU1042/Metal2070通过；
十二项同版Metal实机（多数MPI2）
抽查全部真实exit0通过，源/资产/运行器哈希一致。详见
`mpi-evidence/training-indexed-regular-effects-v4-20261007/README.md`。

2026-10-07动态linked索引定时回调已按原始NumPy数组/标量模式接入，
重复索引与同块selector修改保留实际写回。40CPU全部VJP/前向、12
carry/原子性与4原始NumPy回归通过；最终十一模块和实机抽查执行中。
错误文字断言的真实Metal失败保留，4项CPU/Metal原子性已核对；
共同v3十一模块3096身份完整回归正在执行；八项同版Metal/MPI2
实机抽查真实exit0通过。详见
`mpi-evidence/training-indexed-regular-effects-v3-20261007/README.md`。

2026-10-07连续Subgroup run_regularly数组回调支持高级索引副本与
显式写回，48CPU/MPI全部VJP/原始Brian前向及12carry/恢复通过。
最终八完整模块1521身份两端exit0/audit0：CPU515/Metal1018通过；
同版八项Metal/MPI2抽查也通过。详见
`mpi-evidence/training-subgroup-callback-effects-20261007/README.md`。

2026-10-07零连接Synapses定时回调也保留scalar eager与共享标量更新，
空Poissonλ校验不采样。四完整开发模块180CPU/MPI通过；最终九模块
共同v2完整回归1224身份两端exit0/audit0：CPU422/Metal823通过；六项
Metal/MPI2抽查也通过。v1完整回归的三个旧错误
文本断言已用原始NumPy核对并修正。详见
`mpi-evidence/training-empty-synaptic-regular-effects-v2-20261007/README.md`。

2026-10-07非空Synapses定时回调已支持整数组隐藏写回、临时别名及
typed状态；真实NumPy优化和int32别名类型保留。96CPU全部VJP与原始
前向、16carry/恢复检查通过，最终八完整模块1998身份两端真实exit0/audit0：CPU674、Metal1336通过；同版六项实际Metal/MPI2抽查也通过。
详见 `mpi-evidence/training-synaptic-regular-effects-20261007/README.md`。

2026-10-07连续typed突触：整数／布尔borrowed隐藏状态与系数副本已接入
五种积分方法、summed输入、SDE及全部VJP。完整开发梯度模块160通过，
8续跑/恢复通过；最终共同版6项实际Metal/MPI2抽查通过，18完整模块
3310身份CPU/Metal仍在执行。神经元typed积分
此前15模块已完整双端exit0/audit0，CPU788/Metal1500通过。
详见 `mpi-evidence/training-typed-synaptic-integrators-20261007/README.md`。

2026-10-07空reset更新：标量eager工作在空事件选择仍执行，数组discarded
域只在硬执行标记下检查，代理F反向不误报未执行的域。Poisson空数组率
验证不抽样；原始失败与修正均保留。修正后三模块50CPU/MPI通过，最终
冻结版实际Metal7项通过，十完整模块1607身份回归仍在运行。
详见 `mpi-evidence/training-empty-reset-effects-v2-20261007/README.md`。

2026-10-07阈值更新：canonical借用数组的隐式写回、比较求值顺序、
refractory期间执行、NumPy eager复合布尔及int32/Boolean控制已支持。
三个开发完整模块72/40/16 CPU通过；最终7完整模块659身份共同版本
CPU/实际Metal仍在执行；最终同一冻结版6项实际Metal验证真实exit0通过。
详见
`mpi-evidence/training-threshold-callback-effects-20261007/README.md`。

2026-10-07 reset新增：完整canonical神经元的事件选择副本、仅显式目标
写回及可修改私有系数副本已支持，含float/int32/Boolean。开发完整模块
48 CPU/MPI通过，最终冻结版实际Metal4项通过，完整CPU/Metal各146身份
验证仍在执行。空事件标量
实参会无条件执行，当前明确拒绝，原始Brian反例2项已核对。
详见 `mpi-evidence/training-reset-callback-effects-20261007/README.md`。

2026-10-07 typed更新：常规完整神经元和固定地址事件回调现在保留int32/
Boolean数组类型、借用/副本、合法原地运算、最终或逐语句存储转换，离散
状态停止梯度。整数事件缓存以integer_state节点复制，避免经GPU浮点丢低位。
常规完整模块88通过，6非法转换的原始NumPy错误已核对；修正二值事件队列
代理VJP验收后的最终Metal6项通过。48完整模块共同版本各8392身份同序
收集成功，测试尚在执行。其他dtype/索引/调用点仍有边界。
详见 `mpi-evidence/training-typed-callback-effects-v2-20261007/README.md`。

2026-10-07 条件事件路径现已接通：固定地址浮点数组的选择副本、scalar-if
fallback、实际无mask的向量普通端点赋值、带条件的分阶段临时重放及随机发射
身份均保留。隐式条件缓存为detached Boolean，延迟更新与恢复已有原始对照。
268个CPU身份有分次开发证据；最终冻结Metal4项实测通过，41完整模块仍在运行。
完整范围与失败历史见 `mpi-evidence/training-guarded-event-effects-20261007/README.md`。

2026-10-07 新增 unless refractory 条件写入及带 refractory 的数组效果积分器。
条件 RHS 数组读产生独立选择副本，LHS 修改已有局部数组，因此早先借用别名
仍能观察到选择写入。最终显式赋值应用掩码；隐藏回调写入按实际语句顺序执行。
NumPy eager where 的两个值分支均执行，包括 Heun/Milstein 的重复回调。
完整新积分器模块80 CPU通过，常规模块164通过；最终冻结版实际Metal抽查4项
通过。完整40模块各6478身份已收集并启动，尚未取得整体终态。
源码、失败历史与验收范围见 `mpi-evidence/training-guarded-callback-effects-20261007/README.md`。

当前实现保留借用参数和临时数组的差异、函数内部别名、标量参数重新绑定、返回
别名在后续赋值中的变化、左操作数先求值但在右侧调用后才读取数组的规则。
标量增量运算若提升为新数组，保存的别名会继续共享新数组。未显式赋值的输入
状态若被回调修改，也会加入动作写回。随机函数在调用点先绑定
独立的原生随机流；反向传播重放已绑定的值，支持 full/TBPTT 和本机 MPI。

Brian 的语句优化决定 `1*v`、`v+0`、`v/1` 是否仍是新数组。因此含有此类回调的
动作使用 Brian 自身的语句优化结果，遵循当前 loop_invariant_optimisations 设置。
测试同时覆盖两种优化设置、两种单位设置和 stateless=False/True。

```python
import brian2 as b

def curve(x):
    saved = x
    x *= .8
    return saved

group.namespace['curve'] = b.Function(
    curve, arg_units=[1], arg_names=['x'], return_unit=1, stateless=False)
# v 数组引用的求值不同于 v+.125 这个新数组。
group.run_regularly('v=curve(v)+v', when='start')
```

当前只接受普通可读取 Python 函数、固定位置参数、有限不可变 builtin 标量捕获、
直线算术与明确的局部增量写入。递归、局部名字遮蔽后的函数调用、未知操作、只读
存储写入明确拒绝。源码、语句、表达式和 eager 运算均有预算；最终程序仍遵循
128 节点的原生限制。即使中间算术结果未被返回，其域错误也不会被删去；训练失败
不得推进 optimizer、carry、时钟或随机序列。

canonical linked 名字指向同一物理数组时，转换器现在保留 NumPy 的完整写回规则：
显式写入的局部数组初始化为独立副本，未写入的同址名字共享借用引用，实际写回
按 Brian 生成器使用的顺序更新物理数组及所有存活的借用引用。最终原生输出对应
写回后的物理状态，而不是独立求值的局部名字。因此先写 u 再写 a、先写 a 再写 u
这两种原始 NumPy 结果均能正确保留。原反例与早期拒绝阶段继续保留为历史证据。

共享标量保持独立局部值，物理写回遵循实际 NumPy 顺序。向量块开始时，共享
ArrayVariable 会从已写回的物理数组重新读取；预先计算的标量 AuxiliaryVariable
（例如 _lio_1）保留原值。两种语句优化设置均有原始 Brian 对照。写回顺序记录在
provenance，并已组成持久原生程序；重放不用再次依赖 Python 的集合迭代顺序。

固定或动态数组索引产生的 NumPy 副本当前仍明确拒绝，整数/布尔数组输入也明确
拒绝，保留原地浮点运算的 dtype 错误规则。

尚未完成的范围包括带 refractory 条件/子群/
动态索引的写回、Synapses 路径与 SDE 回调中的借用写入、更多整数/布尔/容器及动态
Python 控制。纯回调已有的更广能力仍使用原来的纯函数转换器。这一扩展不表示
任意 Python 或全部 Brian 语义已经支持。

## 可核对的本地证据

- 隔离原型：`mpi-evidence/training-callback-effects-prototype-20261006`；该阶段本身
  不含公共入口集成。
- 公共入口初版：`mpi-evidence/training-public-callback-effects-20261006`；5 个完整
  模块、368 身份，冻结 770 输入/826 资产。CPU 真实退出成功、134 通过/234 硬件
  跳过，严格审计通过。实际 Metal 251 通过/117 CUDA 硬件跳过，真实退出和严格审计也成功。
- 中间 linked 检查版：`mpi-evidence/training-public-callback-effects-linked-20261006`；
  新的独立冻结快照，3 个完整模块/351 身份，CPU125/Metal238 通过，均真实退出与严格审计成功；这一版早于整数数组和标量提升修正。
- 前一完整源码版本：`mpi-evidence/training-public-callback-effects-canonical-fixed-20261006`；
  3 个完整模块/373 身份，CPU135 通过/238 硬件跳过、实际 Metal254 通过/119 CUDA
  硬件跳过，均真实退出 0 与严格审计成功。770 输入/826 资产全部一致，主工作区
  输入与冻结源码一致。此前 canonical 测试构造失败保留，修正版仅改测试。

当前重叠写回版本单独冻结在
`mpi-evidence/training-public-ordered-writeback-20261006`，5 个完整模块、642 身份。
开发阶段 52 项原始 Brian 前向/顺序检查及 12 项固定噪声梯度检查通过；六个独立
Python hash seed 的 18 个原始 Brian/native 探针验证两种写回顺序。完整 CPU228/Metal435 通过，
分别414 GPU/207 CUDA硬件跳过，真实退出及严格审计均成功；642身份、770输入/
826资产一致，主工作区源码一致。不同 Python hash seed 保存/恢复的固定噪声 carry
尾段状态/梯度/时钟/序列完全相同。共享向量重读的原始两项
失败及修正结果保留。

初版和修正版的源码与测试不同，不能将初版的整套结果称为修正版已通过。原生
二进制、Rust/GPU 源码和 ABI 在本轮未改动。CUDA 实机尚未验证；跨主机按用户要求
暂缓。本轮未启动云端作业。原始“完整实现随机方程与动态突触”目标保持 active。

## 生成积分器回调（2026-10-06，整模块回归运行中）

Euler/RK2/RK4/Heun/Milstein 已接入生成块的数组复制、借用、阶段求值与隐藏写回。
开发检查90通过162 GPU硬件跳过，包括原始 NumPy 前向、两种单位模式、全部参数/
初态有限差分、full/TBPTT、本地 MPI2，以及随机数组实参和失败原子性。
普通回调64次操作预算保持；生成积分块允许最多128次有界操作，最终程序仍限
128节点。完全 eager 的原生程序保留全部顺序节点；懒程序仍显式连接执行根。
这使合法耦合 Milstein 不再因冗余序列节点被拒绝，非对角共享噪声的原始拒绝保持。

8完整模块/994身份、776输入/832资产已冻结，CPU及实际Metal回归运行中；不将
开发检查或收集成功当作整体验收。详见
`mpi-evidence/training-integrator-callback-effects-20261006/README.md`。
无条件 canonical 浮点神经元组之外的积分器回调、阈值/重置及突触路径仍需开发。
原始完整目标保持 active，最新 CUDA/跨主机结果仍不能由本地 CPU检查替代。

## 连续动态突触回调（2026-10-07，最新整模块回归运行中）

连续 own canonical 浮点数组的5种生成积分器已接入，包括普通 a:1 的隐藏写回。
直接借用回调参数的有界符号所有权检查决定是否分配运行时状态；私有临时数组
不会改变调用者持久存储。120项全部梯度/原始前向检查和6项carry、恢复、失败
原子性检查通过；3项纯突触TimedArray兼容检查通过。原始反例与各真实退出保留。
777输入/833资产、12完整模块CPU593/Metal1041通过均真实exit0且严格审计。
前一个776输入/
832资产的994身份CPU已严格验收392通过602硬件跳过，它不含本轮动态突触变化。
具体证据、边界及实际终态见
`mpi-evidence/training-synaptic-callback-effects-20261006/README.md`。
事件路径、端点借用、shared/linked/indexed、更广随机函数实参及其他回调语义仍
需继续开发，完整目标保持active。

## 事件数组回调（2026-10-07，最新整模块回归运行中）

NumPy事件生成器的整块数组、逐语句向量化及标量fallback已有本轮实际对照。
独立索引副本、系数重读、完整事件批次读取值的复制及scatter当前左目标已接入。
256项全部参数/初态VJP和原始前向、10项延迟更新/carry/恢复、2项失败原子性检查
通过。新增快照在延迟重建范围之外，重建后保持真实语义。另保留1个scatter后
跨边重读反例：转换器仍拒绝，尚未实现，完整目标保持active。
最新782输入/838资产、17完整模块CPU/Metal已启动；不能以开发检查宣布整体验收。
详见 `mpi-evidence/training-event-callback-effects-20261007/README.md`。
更早8模块994身份的CPU392/Metal693通过已严格审计；12模块1485身份CPU593通过
已严格审计，Metal1041通过也已严格审计。两者均不含本轮事件实现。

## 分阶段事件与动态延迟（2026-10-07）

scatter后跨边重读已按原始NumPy阶段顺序实现；临时数组/别名在先前快照上
重放，随机重放保留原始发射/edge/stream。路径API更新同步所有阶段并支持恢复。
新增模块真实205通过408硬件跳过，另1项单位断言失败已保留并修正，历史18
模块冻结回归不得冒称成功。主工作区进一步允许严格验证的阶段共享物理delay，
解除事件内delay写入拒绝；独立运行器编译成功，原始前向/队列/全部VJP验收进行中。
梯度为操作顺序/阶段行门控的surrogate延拓；路由量化和空历史退休是离散边界。
最新证据见 `mpi-evidence/training-event-staged-delay-20261007/README.md`。

共享延迟阶段现有101项CPU分轮组合证据及3项实际Metal冻结版预检通过；
最新20完整模块3440身份CPU/Metal仍在运行。原始失败日志保留，完整目标active。

## 组合调用与零维单位包装（2026-10-07）

经过认证的PureFunction/PoissonNoise/TimedInput已在效果解释器组合执行。借用、
独立分配、标量和零维返回保留真实NumPy/单位包装语义。动态突触中的纯函数
别名隐藏写回和显式Poisson+命名SDE已接通；两个新模块各72CPU检查通过。
最终787输入/843资产29完整模块已启动，同一冻结版4项实际Metal检查通过。
共享可变捕获需要原生状态存储，仍明确拒绝；其实际原始反例与复制/标量化
路径均已验证。详见 `mpi-evidence/training-callback-callables-20261007/README.md`。
