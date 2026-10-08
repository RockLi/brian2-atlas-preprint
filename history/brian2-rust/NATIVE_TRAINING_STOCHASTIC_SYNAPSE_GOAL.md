# 随机方程与动态突触：持续实现及验收

2026-10-08 整数组依赖事件选择：从实际到达lane压缩完整回调参数与可训练系数，支持单事件广播，普通重复目标累加保持live值；追加单元素capture广播、运行时inplace形状与discarded整列eager检查。CPU/MPI前向/恢复和全部bank/初态各1项通过，两次旧开发作业Metal重复目标实际exit1，定位snapshot覆盖accum并修正；后续开发mixed imports两次KeyError exit1保留。当前编译边界9项actual exit0。v1冻结856/912、33模块8抽查运行中；v2追加singleton/完整batch修复与形状原子性测试，33模块10抽查准备冻结，未整体验收。多shape/selected-accum参数、mutable-delay lane、神经元跨列、guard/replay、最新CUDA仍待完成，目标active。详见 mpi-evidence/training-whole-selected-captures-v2-20261008/README.md。旧mixed捕获冻结pilot因macOS字体清单KeyError actual exit4，未执行任何测试，855/911哈希一致；保留凭证复用既有字体cache后仍在运行。

2026-10-08 多元素事件／capture等长混合：按实际到达位置保护逐列eager工作，避免未选中配对的域错误；空批次保留连续queue-gate VJP并跳过其真实eager工作。CPU前向1、CPU/MPI与Metal/MPI原始/恢复2、形状原子性及singleton回归2、全bank/连续初态CPU/MPI与Metal/MPI TBPTT及编译边界11各actual exit0，共16项。首轮VJP因空批次shape guard actual exit1，修复后通过，失败证据保留。32模块/29同版抽查正在冻结，未整体验收。整列写入依赖selected、单selected行对较大capture广播、mutable-delay lane重建、神经元跨列、guard/replay及最新CUDA仍待完成，整体goal active。详见 mpi-evidence/training-selected-capture-mixing-20261008/README.md。上一版returned-constants 21同版Metal/MPI实际wait0及identity/order/hash审计0，31模块CPU→Metal串行回归已启动。

2026-10-08 事件调用者返回constant别名：在实际优化语句中追踪捕获引用、派生副本与逐语句重新加载，只有真实写入才提升父字段运行状态；后续独立选择读取观察当前canonical捕获值。原始反例实际exit1；首轮两项通过后captured系数读取拒绝失败，现已接通，四项CPU/Metal/MPI前向恢复实际exit0。VJP首轮CPU alias通过，Metal rebind参考漏掉Brian的tmp+=.1优化，代码实测确认后修正；13项全bank/连续初态、Metal/MPI TBPTT、副本及旧scalar/编译边界实际exit0。31完整模块/21同版抽查准备冻结，尚未整体验收。一般多元素whole-selected混合、可变delay位置重建、神经元跨列、guard/replay及最新CUDA仍待完成，整体目标active。详见 mpi-evidence/training-event-returned-constants-20261008/README.md。

整体返回阶段验证：22项冻结抽查全部实际wait0、无skip，XML/request/实际collection同序，851/907哈希一致。30完整模块已启动CPU→Metal串行回归，预计每端3503身份，未取得完整终态。详见 mpi-evidence/training-selected-capture-returns-20261008/README.md。完整目标active。

2026-10-08 整体捕获返回：每个到达保留独立 presence，按队列/源顺序求位置与长度，返回 alias 先保留整列修改再映射选择写回，单元素返回与选择数组混合计算已接通，长度错误原子失败。前向/恢复/形状错误三项实际 exit0；VJP首轮实际 exit1 为参考遗漏初始队列反事实门，补齐后 CPU/MPI 全 bank/连续初态与 Metal/MPI TBPTT、singleton 恢复三项 exit0。广播/旧步长VJP/编译边界12、银行实时 carry/restore 与 Metal 域2、mutable-delay 明确边界1各 exit0。851 源码/907 资产30完整模块冻结，22同版抽查执行中，完整回归尚未启动，未整体验收。一般多元素 whole-selected 混合、可变delay到达位置重建、神经元跨列、guard/replay和最新 CUDA 等仍待完成，整体目标 active。详见 mpi-evidence/training-selected-capture-returns-20261008/README.md。

重复索引验证终态：18 项同版 Metal/MPI 全部实际 wait0，无 skip，XML/request/实际 collection 同序，850/906 哈希一致。29 完整模块 CPU/Metal 回归已启动，尚未整体验收。详见 mpi-evidence/training-repeated-capture-views-20261008/README.md。完整目标 active。

步长／反向视图抽查终态：28 项冻结 Metal/MPI 全部实际 wait0，无 skip，849/905 前后哈希一致。旧快照完整回归由新增重复索引版本覆盖，不再重复启动；它仅作为阶段证据保留。

2026-10-08 可写重复索引事件捕获：zero-stride 逻辑列共用一个物理状态单元，整列结果缓冲后按列顺序写回；新增逐列不同增量、int32 溢出、Boolean 停止梯度及恢复测试。开发三项原始/恢复和两项全参数/连续初态 VJP 实际 exit0，包含 Metal/MPI 与 TBPTT。新快照 850 源码/906 资产、29 完整模块已冻结，18 项同版抽查正在执行，完整回归尚未启动。神经元重复捕获在整列调度完成前显式拒绝；whole-selected 返回、神经元跨列、guard/replay、最新 CUDA 等仍待完成，整体目标 active。详见 mpi-evidence/training-repeated-capture-views-20261008/README.md。

2026-10-08 步长／反向捕获视图：按真实 stride 映射物理存储，外部共享缓冲区按地址合并；只读银行反向／重复索引保持实时银行值。开发 3、4、12 项分别实际 exit0。849 源码／905 资产、28 完整模块已冻结，28 项同版 Metal/MPI 抽查正在执行，完整回归尚未启动。详见 mpi-evidence/training-strided-capture-views-20261008/README.md。可写重复单元、whole-selected 返回／广播、神经元跨列、guard/replay 和最新 CUDA 仍待完成；整体目标 active。

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

用户目标：完整实现此前缺失的随机方程与动态突触。该目标保持 active；某个子集通过
测试不代表整体完成。本文件记录验收要求与各阶段实际结果；较早阶段的剩余项
以文末后续实现记录为准。

## 最新可核对状态（2026-10-07）

2026-10-07动态索引完整回归修正：首次完整CPU真实136失败，1项
旧静态索引拒绝断言与新支持冲突，135项事件无guard局部临时量契约
误拒绝。现识别真正AST Store局部，并将旧拒绝断言改为原始NumPy
私有副本及父状态不污染对照。相关147CPU/MPI检查真实通过；原先
135失败身份精确比对均已实际通过。最终v4冻结811源/867资产十一
完整模块CPU46637/Metal46638已启动；十二项同版Metal/MPI2抽查
覆盖动态索引和新修正的事件临时量，全部真实exit0通过；最终十一
模块3098身份两端exit0/audit0：CPU1042通过2056硬件跳过，Metal2070
通过1028CUDA跳过，源/资产/运行器哈希一致。详见
`mpi-evidence/training-indexed-regular-effects-v4-20261007/README.md`。
ufunc.at定时多阶段回调、多级写索引、共享可变常量/捕获与最新CUDA
仍需实现验证，完整目标active，跨机器暂缓。

2026-10-07动态索引回调更新：按真实NumPy生成模式执行数组快照/
私有计算/有序scatter或重复索引逐行标量循环。runtime linked选择器、
索引副本、同块修改索引、全部bank/连续初态VJP及原始Brian前向
40CPU/MPI通过，12恢复/错误原子性通过。原始NumPy stateupdate
标量循环的未定义_idx与ITERATE_ALL读写错误已修正，四项原始
NumPy向量化回归通过。首轮真实失败保留。最终811源/867资产十一
完整模块首次冻结CPU75537/Metal75538仍在执行；Metal抽查7通过1
错误文字断言失败，后续CPU与单进程/MPI2 Metal四项错误原子性均
真实通过。共同v3只修改测试文字匹配，811源/867资产十一模块
CPU15890/Metal15891完整3096身份回归仍在运行；八项同版Metal/MPI2
实机抽查真实exit0通过，哈希一致。详见
`mpi-evidence/training-indexed-regular-effects-v3-20261007/README.md`。
ufunc.at多阶段回调、一般多级写索引、共享可变常量/捕获与最新CUDA
仍需完成，完整目标active，跨机器暂缓。

2026-10-07子群定时回调更新：连续选中神经元Subgroup的高级索引
副本与显式NumPy写回已接入，隐藏修改不污染父群体，float/int32/
Boolean、系数副本、别名及随机流保留真实语义。完整开发48CPU/MPI
全部bank/连续初态VJP及原始Brian前向通过，12续跑恢复通过；首两轮
测试误用的实际失败保留。最终810源/866资产八完整模块CPU14857/
八完整模块1521身份两端真实exit0/audit0：CPU515通过1006硬件跳过，
Metal1018通过503CUDA跳过；同版八项Metal/MPI2实机抽查也通过。详见
`mpi-evidence/training-subgroup-callback-effects-20261007/README.md`。
一般动态/多级索引、端点重复索引、共享可变常量/捕获及最新CUDA
仍需完成，整体目标active，跨机器暂缓。

2026-10-07零连接Synapses定时回调更新：共享标量更新与向量回调中的
scalar eager/domain动作已实现，数组域不执行；空Poisson标量λ校验而
不采样、不生成似然站点。基础16CPU/MPI及扩展四完整模块180CPU/MPI
通过，分别32/344硬件跳过；两个实际失败记录保留。首个冻结版六项实际Metal/MPI2通过；九模块CPU/Metal真实exit1的三个
旧casting错误文本断言已由原始NumPy核对并修正，三项实测通过。
共同v2冻结809源/865资产，CPU79526/Metal79527已启动，九模块与
同版六项Metal/MPI2实机抽查通过；九完整模块1224身份两端真实exit0/
audit0：CPU422通过802硬件跳过，Metal823通过401CUDA跳过。详见
`mpi-evidence/training-empty-synaptic-regular-effects-v2-20261007/README.md`。
当前新源码完整验收尚未结束，更广索引/共享借用、可变常量/捕获和
最新CUDA仍需完成；目标保持active，跨机器暂缓。

2026-10-07 Synapses定时效果更新：非空canonical整数组回调、隐藏状态
分配、跨语句别名、float/int32/Boolean、真实NumPy优化写回和命名随机
调用已接入。透明int32数组别名不按C临时推断误提升为int64。三个实际
失败保留；完整开发梯度模块96CPU/MPI通过192硬件跳过，16续跑恢复
通过32跳过，均真实exit0。最终807输入/863资产八完整模块1998身份两端真实exit0/audit0：CPU674
通过1324硬件跳过，Metal1336通过662CUDA跳过；同版六项实机抽查通过。详见
`mpi-evidence/training-synaptic-regular-effects-20261007/README.md`。
零连接scalar eager、indexed/shared、常量借用/可变捕获及最新CUDA仍需
实现验证，完整目标保持active，跨机器暂缓。

2026-10-07连续typed突触更新：own canonical int32/Boolean隐藏写回
现在分配真实运行时存储，五种生成积分器和系数副本保留dtype/停止
梯度；包含summed输入、命名SDE、full/TBPTT全部bank/初态VJP。
完整开发梯度模块160CPU/MPI通过320硬件跳过，8续跑/恢复通过16跳过，
均真实exit0；首轮只读误判真实失败保留。最终806输入/862资产18完整
模块各3310身份同序真实收集；CPU真实exit0/audit0，1154通过2156硬件跳过，
实际Metal真实exit0/audit0，2232通过1078CUDA跳过。
最终同一冻结版6项实际Metal/MPI2全部VJP和恢复抽查真实exit0通过。
冻结源/资产/运行器哈希一致，当前实现相同；另外一个无关测试文件的
工作区变动单独记录，不能冒称整个工作区806输入仍完全相同。
详见 `mpi-evidence/training-typed-synaptic-integrators-20261007/README.md`。
更广shared/indexed、Synapses regular、可变捕获/常量借用与最新CUDA
实机仍待完成，完整目标active，跨机器暂缓。

2026-10-07神经元typed积分完整验收：int32/Boolean借用状态及系数副本
已接通Euler/RK2/RK4/Heun/Milstein、refractory、命名SDE和全部VJP。
最终805输入/861资产15完整模块2212身份两端真实exit0/audit0：CPU788
通过1424硬件跳过，实际Metal1500通过712CUDA跳过。新增typed完整模块
CPU200/Metal400通过；最终6项Metal抽查也通过。详见
`mpi-evidence/training-typed-integrator-effects-20261007/README.md`。
后续连续typed突触、更广索引/共享可变捕获/常量借用及最新CUDA仍需实现
验证，完整目标保持active。

此前空reset修正版本十完整模块1607身份也已两端真实exit0/audit0：CPU568
通过1039硬件跳过，实际Metal1095通过512CUDA跳过。阈值七完整模块659
身份CPU252/Metal463通过，407/196硬件跳过，均真实exit0/audit0；仅证明
各自冻结版本，不替代后续源码的验收。

2026-10-07空reset更新：无事件标量eager工作、组空选择判断和原子失败
已接入，Poisson空数组检查负值/非有限/NumPy int64率上界且不抽样。
独立硬执行标记只检查实际选中数组的discarded域，代理反向F候选不误报
未执行操作；v4形状prefix以identity表示私有v5 selector。原始两失败、
真实旧作业SIGTERM/child.wait=-15与Metal测试warm wrapper差异均保留。
修正后三完整模块真实exit0：50CPU/MPI通过100硬件跳过，含原始Brian、
恢复与全部VJP。最终803输入/859资产十完整模块CPU51269/Metal51277各
1607身份同序收集通过，回归仍在执行；同一冻结版实际Metal7项真实
exit0通过，源/资产/主工作区/运行器哈希一致。完整目标active；详见
`mpi-evidence/training-empty-reset-effects-v2-20261007/README.md`。
更广shared/linked/动态索引、可修改常量借用、共享可变捕获与最新CUDA
实机仍待完成；跨机器按用户要求暂缓。

2026-10-07阈值更新：canonical整组借用数组的左右求值顺序、隐藏状态
写回、refractory期间回调、NumPy eager复合谓词以及int32/Boolean控制
已接入现有原生动作。比较margin/谓词与状态同时输出，连续代理VJP及
离散控制停止梯度保留。三个开发完整模块分别72/40/16 CPU通过，均真实
exit0；最终798输入/854资产7完整模块CPU17875/Metal17876各659身份
同序收集通过，完整回归仍在执行。最终同一冻结版实际Metal6项真实exit0
通过，包括MPI2恢复、操作数顺序、refractory、eager别名和typed TBPTT
全部VJP；798输入/854资产、主工作区与原生运行器哈希一致。详见
`mpi-evidence/training-threshold-callback-effects-20261007/README.md`。
无事件标量reset、更广索引/共享可变捕获/常量借用及最新CUDA仍待完成。
完整目标保持active，跨机器按用户要求暂缓。

2026-10-07 reset调用点更新：canonical神经元选择副本、显式目标写回、
可修改私有系数副本及float/int32/Boolean效果已接入。积分器按真实调用点
分流。开发版完整新增模块真实exit0：48通过96硬件跳过，含原始Brian、
恢复、MPI2及full/TBPTT全部参数/初态VJP。空事件标量实参初次2项失败已
保留，准入修正后2项原始Brian反例真实exit0通过；该路径仍明确拒绝，
尚未实现无条件原生执行。最终795输入/851资产2完整模块CPU91080/
Metal91082各146身份同序收集通过，尚无完整终态。同一最终冻结版实际
Metal4项真实exit0通过，含MPI2恢复及系数/整数/布尔TBPTT全VJP；源码、
资产与主工作区哈希一致。详见
`mpi-evidence/training-reset-callback-effects-20261007/README.md`。
后续同一reset冻结版完整146身份已两端真实exit0/audit0：CPU50通过96
硬件跳过，实际Metal98通过48CUDA跳过；795输入/851资产严格审计通过。
阈值调用、更广索引、共享可变捕获及最新CUDA仍待完成，完整目标active。

2026-10-07 typed数组效果更新：常规神经元和固定地址事件回调接通int32/Boolean
输入、借用/副本、存储转换与停止梯度。整数快照采用integer_state搬运而不是
float；vectorised事件逐语句转换，普通向量块保留浮点局部到最终写回。本机
NumPy的有限正溢出转换已实测并快照，保留原生checked域和显式dtype边界。
完整常规模块88通过176硬件跳过，6非法原始NumPy运算检查及12事件VJP预检
通过。v1实际Metal4通过2失败，发现测试错误地停止非detached二值事件队列
的代理梯度；生产实现不变，独立延迟队列12项全部VJP复验通过。v1正式作业
主动终止并保存真实wait终态；直接模块保存SIGINT/exit2部分失败证据。最新
修正验收793输入/849资产48完整模块CPU68501/Metal68502各8392身份同序
收集成功，整体验证仍在运行；最终冻结版同一6项实际Metal已真实exit0通过，
包括停止梯度控制与非零事件队列代理VJP。完整目标保持active；详见
`mpi-evidence/training-typed-callback-effects-v2-20261007/README.md`。
旧20模块共享延迟CPU92369已真实exit0/audit0：1286通过2154硬件跳过，仅
证明旧785输入/841资产冻结版；不替代最新typed、条件或组合回调验收。

2026-10-07 条件事件更新：固定浮点地址的向量选择副本、scalar-if fallback、
实际NumPy无掩码普通端点赋值、分阶段条件快照和临时数组重放已接通。随机事件
保留发射身份，Boolean缓存保持detached。分次开发证据覆盖268个唯一CPU身份，
其中包括真实exit1作业中48个成功项，不能冒称最终共同版本的一次完整exit0。
最新790输入/846资产冻结版实际Metal4项已真实exit0通过，含MPI2/TBPTT/恢复/
随机延迟重建。41完整模块CPU79488/Metal79495各7278身份同序收集成功，测试
仍在执行。原始基线16模型可运行而旧前端全拒绝。详见
`mpi-evidence/training-guarded-event-effects-20261007/README.md`。
更广可变条件/索引、离散数组、阈值/重置及共享可变捕获仍需开发，最新CUDA仍
无实机证明；完整目标保持active，跨机器按用户要求暂缓。

2026-10-07 条件回调更新：完整 canonical 浮点神经元的 run_regularly 条件写入
已保留 NumPy 选中副本、局部别名、系数私有修改、空选择和标量错误顺序。
Euler/RK2/RK4/Heun/Milstein 的 refractory 效果积分器已接通；Heun/Milstein
实际原始 Brian 对照揭示 eager where 分支重复执行回调，修复后完整模块真实
exit0，80通过160硬件跳过，覆盖全部参数/初态 VJP、两单位模式、MPI2及TBPTT。
常规完整模块164通过324硬件跳过，真实exit0。最终冻结Metal抽查4项也真实exit0
通过并断言实际GPU分发。789输入/845资产40完整模块的最新冻结CPU25456/
Metal25457各6478身份同序收集成功，完整测试尚在运行。事件条件/索引/离散数组及共享可变捕获仍
需实现，目标保持active。详见 `mpi-evidence/training-guarded-callback-effects-20261007/README.md`。

2026-10-07 组合调用更新：经过认证的纯函数、Poisson、时间输入已接入数组效果
解释器，保留借用/私有/标量返回和自动单位包装的零维转换。普通动态突触a:1
经纯函数返回别名的隐藏写回及显式Poisson+命名SDE已接通。两个新模块各72项
全部参数/初态VJP和原始前向通过，旧所有权/边界/VJP三个模块136项通过。
共享可变捕获仍缺原生状态存储，实际反例及明确拒绝已验证。最终787输入/
843资产29完整模块CPU69996/Metal69997已启动；同一最终冻结版4项实际Metal
抽查已真实exit0通过，仍不能宣称完整回归及目标验收完成。详见
`mpi-evidence/training-callback-callables-20261007/README.md`。完整目标仍active。
另旧17事件模块2485身份已两边真实exit0严格审计：CPU951通过1534硬件跳过，
实际Metal1720通过765 CUDA跳过，仅证明该旧冻结版本。

2026-10-07 共享事件延迟已实现：严格验证的 NumPy 阶段共享物理delay，在运行
入口共同冻结路由，事件内写入影响下次运行。独立原始 Brian 前向、48项全部
参数/初态VJP、40项非随机carry/恢复、5项非法别名原子性通过；8项随机carry
参考已修正并真实复验通过。合并证据为101CPU通过192硬件跳过，不冒称单次
完整exit0。最新785输入/841资产20完整模块同一冻结版CPU92369/Metal92370
已启动（两边各3440身份）；同一冻结版实际Metal3项预检也已真实exit0通过。
最新CUDA尚无实测，更广条件写/索引/离散数组和混合回调语义仍待实现。
详见 `mpi-evidence/training-event-staged-delay-20261007/README.md`；完整目标保持active。

2026-10-07 最新分阶段事件开发：scatter 后跨边重读、临时数组/别名重放及原始
随机流身份已实现。新增模块真实205通过408硬件跳过，1项原始Brian的带单位
测试断言失败已保留并修正。18模块784输入/840资产冻结回归仍保留该历史失败，
不能宣布整体验收。主工作区正在独立版本实现分阶段事件内delay写入；原生只允许
顺序、源事件、时钟及物理延迟一致的阶段共享延迟。新运行器已独立编译，正向/
续跑/恢复/全部VJP和失败原子性预检进行中。详见
`mpi-evidence/training-event-staged-delay-20261007/README.md`，完整目标保持active。
此前12模块连续突触CPU593/Metal1041通过均已真实exit0严格审计；仅证明各自
冻结版本。更广条件写/动态索引/离散数组及最新CUDA实测仍有剩余工作。

事件回调更新（2026-10-07）：保留 NumPy 整块数组、向量化逐语句重读、标量
fallback及事件批次快照。256项全部梯度/原始前向、10项延迟更新与恢复、2项
失败原子性检查通过；初始队列VJP采用独立连续代理延拓，真实前向二值约束保持。
最新782输入/838资产17完整模块CPU/Metal验证已启动。该历史版本另有真实scatter后跨边
重读反例明确拒绝（后续已实现，见上述最新记录），更广别名、条件写/索引/离散存储等必须继续开发。详见
`mpi-evidence/training-event-callback-effects-20261007/README.md`。完整目标仍active。

此前8模块994身份实际CPU/Metal均exit0并严格审计：392/693通过；连续突触
12模块1485身份CPU实际exit0严格审计593通过，Metal实际exit0严格审计1041通过。它们不含最新事件
代码，不能替代17模块的实际终态。

2026-10-07 更新：连续动态突触 Euler/RK2/RK4/Heun/Milstein 的数组回调已接入，
包括普通a:1的隐藏运行时写回及临时数组的区分。120项全部参数/初态梯度和原始
Brian前向、6项carry/恢复/失败原子性、3项时间输入兼容检查通过。最新777输入/
833资产的12完整模块CPU/Metal回归已启动；尚无最新整体验收终态。详见
`mpi-evidence/training-synaptic-callback-effects-20261006/README.md`。事件及端点/
索引等更广回调语义仍需实现，目标保持active。

此前积分器8模块994身份CPU已严格验收：392通过602硬件跳过，真实exit0。
该冻结版本不含最新突触回调，不能直接当作最新12模块整体验收。

最新开发已接通 Euler/RK2/RK4/Heun/Milstein 生成块的数组回调与隐藏状态写回。
90项开发检查通过，162项对应GPU硬件跳过；覆盖合法耦合共享/独立噪声、全部
参数和初态梯度、full/TBPTT、本地MPI2及直接随机数组实参。8完整模块994身份的
CPU/Metal回归正在同一776输入/832资产冻结版本上运行，尚无完整终态；详见
`mpi-evidence/training-integrator-callback-effects-20261006/README.md`。
普通回调默认64次操作预算保持，生成块最多128次操作且最终仍限128原生节点。
无 refractory canonical 浮点数组之外的积分器及突触回调语义仍需继续开发。


本轮已实现重叠 canonical linked 数组的 NumPy 复制、借用及精确写回，也补上共享
标量写回顺序和向量块重读。52 项原始 Brian 对照及 12 项随机梯度检查通过；
六个 Python hash seed 的独立进程验证两种实际写回顺序。当前 5 模块 642 身份
已真实退出并严格审计：CPU228通过414硬件跳过，实际Metal435通过207 CUDA跳过；
770输入/826资产及主工作区输入全部一致。跨 Python hash seed 的固定噪声 carry
恢复结果完全一致。详见 `training-public-ordered-writeback-20261006`。
原始目标仍 active，积分器和突触路径的借用写回仍需开发。

公共 `run_regularly` 浮点回调借用写回已接入本轮开发，详见
`NATIVE_TRAINING_CALLBACK_EFFECTS.md`。初版 5 模块 368 身份 CPU 已严格验收；
新 linked 检查版另有独立快照与实测，不将旧 121 模块或初版结果冒称当前整体验收。
前一版重叠 linked 的实际反例和拒绝阶段已保留；本轮已补齐 canonical 重叠写回，
其他回调调用位置仍需继续实现。目标保持 active。
最终 canonical 公共写回版 3 完整模块/373 身份已真实退出并严格审计：CPU135通过238
硬件跳过，实际Metal254通过119 CUDA跳过；770输入/826资产一致，主工作区源码一致。
数组标量提升反例及固定索引测试构造失败均保留，详见最终 canonical-fixed 证据目录。

本轮补齐不可变数值keyword-only默认参数，并修复Brian NumPy丢弃单位时复制
回调丢失__kwdefaults__的问题。旧冻结版本明确discard_units=True的独立反例
真实exit1，原日志保留；修复后的完整新增124身份CPU48通过76硬件跳过、
实际Metal86通过38CUDA跳过，真实wait退出和严格审计9dc4aa/00af71均成功。
两种warm单位模式、原始Brian前向、Heun共享/Milstein噪声、全部bank和初态
有限差分、本地MPI2及full/TBPTT已有新增模块证据。当前13模块2029身份完整CPU49282已真实exit0，严格审计041e2e证明
755通过1274硬件跳过；Metal51492也真实exit0，严格审计402c25证明1393通过636 CUDA硬件跳过；它们与较早119模块冻结版本不同，不能
互相冒称最新整体验收。

随机 Euler/Heun/Milstein、多状态/命名噪声、时间依赖、refractory、动态连续/
事件突触、pre/post、延迟、summed、BPTT、迁移和恢复均已有实现及分阶段实际验证。
完整目标仍 active；分阶段结果不能直接证明当前全部语义或最新 CUDA 完成。

此前私有浮点数组与回调阶段的完整验收：10 模块 1,168 身份，CPU458通过710 GPU
硬件跳过，实际 Metal814通过354 CUDA跳过。两套真实 exit0 与严格审计证明761源/
817资产及逐项XML一致；归档912成员全部回读哈希匹配。显式私有浮点数组的别名
原地更新、独立复制、重新绑定及标量零维行为已验证；动态循环、可变捕获、借用
调用者存储和更广离散存储/任意Python语义仍未全部覆盖。

旧回收 Metal49818 已确定句柄/进程/临时冻结目录消失，且没有真实最终终态；
不再视为运行中或完整成功。旧CPU2107身份为963通过1144硬件跳过的严格组合
验收，保留原单项启动路径失败及复验，不冒称一次完整exit0。

数值包装函数修正前版本已对118原生训练模块创建同一持久T7快照：761核心源码加2必要的
离线验证器依赖，819资产；原生运行器及Rust/GPU源码不变。CPU43189与Metal65111
各自完整收集到相同9417身份。数值包装修正后两套旧作业被主动停止，均有真实
exit143；保留部分完整模块结果，不能宣称118模块完整验收。详情见
`mpi-evidence/training-full-current-corpus-20261006`；原中断证据不覆盖或删除。

最新CUDA仍缺实机证据，跨机器按用户要求暂缓。本轮本机MPI/CPU/Metal验证不
上传云端、不分配CUDA、不执行SSH/Teleport作业；跨主机证据模块只验证离线数据。
下方阶段记录保留对应冻结版本和真实证据，不可互换作为最新完整目标的终态。

## 随机方程

- Brian 原生随机方程转换：加性 Euler、随机 Heun、Milstein；保持 Brian 内置方法
  的公式和噪声结构约束，不把一种随机积分语义替换为另一种。
- 多状态、命名噪声与状态间共享噪声、不同神经元/样本独立噪声；噪声幅度及其状态/
  参数依赖的路径导数，兼容时间依赖、refractory、full/TBPTT。
- 原生随机数生成；seed、序列和时间位置可重放，反向传播不重新抽样；分区数不会
  改变样本。新序列与 carry 的行为明确，失败/只读操作不推进状态。
- checkpoint 及新进程恢复、CPU/Metal/CUDA、单机 MPI；独立 Brian 前向、固定
  噪声有限差分、随机流独立性/分布检查和跨后端测试。

## 动态突触

- 突触逐边状态与常量参数、连续积分以及事件驱动状态；表达式可读取相关神经元状态。
- pre/post 路径和 Brian 顺序赋值语义；支持动态传递、短时可塑性及 STDP 更新，
  连续/事件状态的初态与训练参数不能混淆。
- 前向状态、事件路径 VJP 和全部状态的跨时间梯度；明确硬事件、surrogate、
  饱和/分段函数与 detach/TBPTT 的导数规则。
- 完整状态 carry/store/restore、冻结/掩码语义、MPI 所有权与归约、GPU 执行；
  独立 Brian、手工动态模型与数值差分的证据。
- 审核零延迟之外的事件队列、summed 状态、随机突触及组合情况，不能在转换时
  静默丢弃。缺失能力必须继续实现或明确记录为未达成，不能用静态权重替代动态状态。

跨机器按用户先前要求暂缓。本机开发不依赖新云授权；CUDA 实机验收仍须使用明确
授权的资源。现有边界（有界 SSA、内存预算、默认 Brian 调度等）保持显式检查。
最终完成前逐项核对代码、测试与实际运行证据，不以收集成功或旧版本云报告代替验收。

## 当前进度与下一实现接口（2026-10-03）

神经元 SDE 已接入 native v4r6，详见 `NATIVE_TRAINING_STOCHASTIC.md`；验收结果以
对应目录为准。这不是整体完成。动态突触 v5 的 CPU/MPI 原生动作核心和 Brian 转换
入口现已接入，见 `NATIVE_TRAINING_DYNAMIC.md`。连续/事件驱动 STDP、随机边状态、
pre/post 顺序路径、完整状态 BPTT 和恢复已有本地测试。动态 refractory、summed、
异步事件时钟和延迟队列已在后续阶段接通；结构 mask 状态迁移也已实现，本轮完整
回归记录在文末。动态 Metal 单机与本地 MPI 执行已接通，CUDA v5 实机验收仍待完成。
已选神经元之间的 shared/linked 物理状态映射、合法的 shared Synapses 跨对象链接、
常量参数 bank 链接和 TimedArray 输入均已接入后续阶段。外部任意可变输入仍不支持。
以下约束继续适用于剩余工作：

1. 动态突触状态独立于 optimizer 参数。特别是被 STDP 修改的 `w` 是每个 batch
   的运行时边状态；可训练初值/常量参数是 optimizer 槽。carry 后不能重新覆盖运行时 w。
2. 不能只在当前静态投影中替换一个权重表达式。需要保存边状态、事件上下文和对应
   反向 tape；顺序语句必须看到前一语句的新值，pre/post 路径须保留 Brian 的排序。
3. Brian `SynapticPathway.update_abstract_code`（`synapses.py:360` 附近）先用
   `linear(event_driven, variables)` 生成精确推进，将 dt 换为 `t-lastupdate`，
   再执行 pathway code，最后写 lastupdate=t。lastupdate 是离散事件时间，不作
   连续可训练量；状态衰减系数仍须对 tau 等参数求导。
4. `SummedVariableUpdater` 在 groups 槽、目标 order-1 执行，并重置目标累加量。
   连续突触更新与神经元更新同处 groups 槽，须取 Network 排序，不能统一移到其后。
5. 建议扩展为独立动态执行计划/版本：统一动作序列（连续更新、summed、阈值、
   pre/post 路径、reset），完整神经元+边状态布局，逐动作上下文快照和逆序 VJP。
   `clip` 等可塑性常用函数需显式导数规则。CPU 与 GPU 必须共享同一 IR 契约。
6. 事件门可用硬事件前向、声明的 surrogate 反向；零事件处的反事实表达式和
   timestamp 门要有明确规则，不能将离散时刻伪装为连续路径导数。
7. 读取/写入 pre/post 两侧状态会产生跨分区依赖。MPI 必须保持动作顺序与状态一致，
   初期可用逐动作 owner 计算并归约状态/伴随增量保证正确，再优化可交换操作分组；
   不得把所有 rank 重复串行计算说成 target-owned 并行。GPU 不得静默改成 CPU 执行。
8. 延迟需要脉冲历史/事件队列及其跨块状态；反向门贡献须落到原始触发 tick。
   不能仅用当前 tick 的脉冲替代到达事件。随机突触使用独立边/群组地址的原生流，
   避免与已有神经元噪声地址碰撞。

动态事件 IR 还需注意：一个 edge/path 的顺序语句应先组合成完整状态变换 F，再以
该路径的一个事件门作用（反事实贡献 F(state)-state）。把同一门逐语句独立平滑会
改变无事件处的导数，例如 `x+=1; x*=2`，因此不能作为等价实现。refractory 对
神经元写入的屏蔽则必须在顺序组合内部逐目标应用；它不应跳过同一路径的突触状态写入。
self-synapse 的 pre/post 引用和 shared 边状态可能指向相同存储，VJP 要正确累加别名。
外部输入目前只有脉冲数组；动态路径如需读取输入组随时间变化的物理状态，须增加
明确的外部状态输入契约，不能把这些状态静默折叠成初始常数。

### v5 本地阶段验收与下一步

`training-dynamic-core-20261003/verification.json` 汇总 704 passed / 114 skipped；
最后一次受影响模块重跑 111 passed，149 个最终源码哈希保持一致。
动态核心 53 项、Brian 动态转换 28 项；随机方程旧验收仍保留。原始全量日志中的
4 个测试接口笔误失败未删除，最终重跑替换相关模块后无未解决失败。

下阶段应先补本地语义：

- refractory 已接入：旧 tick 活动 latch、递减 counter、threshold 后关闭 latch、
  reset 安装新 counter；神经元写入屏蔽嵌入路径顺序组合，塑性更新继续。
  新 select SSA 采用惰性分支并停止 gate 梯度，支持被屏蔽表达式的域错误隔离。
  103 项专项测试通过（包含 Cython 前向、独立差分、SDE 和 MPI）。
- summed 已把非共享的可写神经元 parameter 纳入物理状态；按实际 updater.order
  清零目标，再按原始 edge order 累加，符合 Cython 模板。事件/reset 也能写入这些
  状态，积分默认 identity，初态梯度与 optimizer 常量参数分开。
- 延迟需要可恢复的离散事件历史以及回传到原始触发 tick 的梯度，不能沿用当前零延迟门。
- 固定 mask 已作用于 w 对应的全部 edge actions；训练器 update_mask 对 v5 改动
  先拒绝，防止旧 runtime w/traces 在 regrowth 后复活。需要设计显式状态迁移策略。
- dynamic GPU 当前原生/前端均明确拒绝，尚无 v5 GPU kernel，不能用 v4 测试替代。

入口 `lower_brian_training(..., dynamic=True)` 或 `lower_brian_dynamic_training`；
默认入口保持原静态 v3/v4 转换边界。新前端目前在训练神经元状态和突触状态之间
建立显式 action context，按 Brian sorted_objects 和每个 pathway 的 stable edge order
构建程序。时间、噪声、clip、局部临时变量、子表达式和神经元参数别名已接通。

### 下一阶段实现注意点（当前源码审查）

- summed 接收量在 Brian 是 parameter，不在 neuronal diff_eq_names。需要把这些
  可写的非共享参数加入物理状态和 identity 更新，同时保留 reset/threshold 参数契约。
- `SummedVariableUpdater` 排在目标 groups updater 之前。Brian Cython 模板先清零目标
  再逐边求值/累加；NumPy 模板先求全部贡献再 bincount 写回。表达式若读取同一接收量
  也有后端顺序差异，需明确 v5 顺序和独立参考，不可无条件声称后端等价。
- 本轮可塑性用例确认 Brian NumPy 的跨边 add.at 与逐边完整路径可不同；Cython
  对照已通过。autapse 审查随后确实发现 pre/post 别名错误：Brian Cython 缓存每个名字
  的独立局部值，再按名字排序写回。已改为前端独立 context slots、最终同址写入选
  最后一个逻辑名字，反向读别名归并。低层 helper 的显式同槽即时别名契约不变。
  8 项 Cython 与独立差分用例通过，覆盖单双别名写入、refractory 和 TBPTT。

### refractory / autapse 阶段最终验收

`training-dynamic-refractory-20261003/verification.json`：21 个模块最终合并
815 passed / 114 skipped / 0 failed；150 个最终源码哈希保持一致。
全量原始回归 807 passed，autapse 修复后 3 个受影响动态模块重跑 192 passed，
按模块替换去重，新增 8 个 alias 用例。原始 alias 不一致证据单独保留并标为已修复。
本轮源码只在 frontend 和新测试中发生全量运行后的修改；Rust 和其它回归源码未变。

该阶段之后继续 summed / 可写神经元 parameter 状态，再接延迟历史 BPTT 和动态 GPU。
不能把当前 CPU/MPI 完成度当作总体目标已实现；CUDA 硬件验收和跨机器仍未执行。

### summed 阶段

summed / 可写神经元状态已实现；53 项专项测试通过（16 组独立差分、随机神经元与
突触的共同噪声 Brian 对照、MPI 训练恢复、空连接、mask、改变调度等）。当前全量
验证见 `training-summed-20261003/verification.json`：最终 22 个模块合并
871 passed / 114 skipped / 0 failed；151 个最终源码哈希保持一致，Rust/runtime 未变。
原始全量为 868 passed，时钟修正后受影响 3 个模块重跑 195 passed，按模块替换去重。

时钟审查另外发现并修正了突触表达式 dt 错取神经元 timestep 的问题，新增 3 项测试。
不含连续积分的 Synapses 可以有不同 dt，表达式的 dt 现取真实突触时钟。
异步 synaptic t 仍待实现，目前显式拒绝（含 event-driven 自动插入的 t）。下一步
需按 Brian Clock._calc_timestep 的近整数容差/ceil 规则计算每个突触时钟的采样时间，
而不是以当前神经元时间代替。该差异由 clock 内部 t/dt 引用及实际 Brian 对照确认。

### 延迟实现候选方案（尚未实现）

源码 `brian2/synapses/spikequeue.h` 规定 delay tick = int(delay/dt + 0.5)，
队列 append 顺序保留较早发出的事件，再按 source 与原始 edge 顺序插入。
`SynapticPathway.queue._full_state()` 返回 `(offset, bins)`；path 执行结束后 advance。
因此已运行网络的转换需要读出尚未投递的 edge events，不能初始化成全零队列。

可扩展 Trigger 支持二进制 state-cell 门，并让门 VJP 写入对应状态伴随。用普通
线性 shift actions 保存每条延迟 edge/path 的事件 FIFO、clear+record action 记录
当前硬脉冲，便可让 BPTT 经 FIFO 返回实际的原始 threshold tick，复用现有 TBPTT、
carry、初态梯度和 MPI 机制，而无需错误地把 delayed event 接到当前 spike。
离散 FIFO 的初值/状态必须校验为二进制；延迟 tick 和事件排序停止梯度；事件幅度
的路径 VJP 才经 FIFO 回传。跨调用的 carry 仍遵循调用边界截断，返回其初态伴随。

需要专项核对异质 delay 的到达顺序、pending queue snapshot、pre/post delay、
event-driven lastupdate 与延迟组合、warmup/store/restore，以及更改 dt/delay 后尚未
投递的历史事件。旧队列和新延迟不一致时不能静默重新解释；C++ prepare 的 dt 转换
依赖 queue 内部旧 dt，而当前 _full_state 未导出该字段，需进一步设计/验证。

时钟实现可考虑 Action 的可选 synaptic clock 描述：同一个 Time SSA 在该动作上
读取经 Clock._calc_timestep 规则量化的时间，避免把 ceil 插入可微参数运算。
若同时读取 t_pre/t_post，还需提供主神经元时钟的独立只读绑定。该设计尚未实现，
必须用不同 synapse dt、非整周期 warmup、事件驱动 lastupdate 和分段执行验证。

### 异步突触时间阶段（本地已验收）

`dynamic.clocks` 与 `clock_time` SSA 已接入原生 CPU/MPI。此处取代上一阶段
“异步 t 尚未实现”的状态。前端快照 Network.t、所有标准 clock 的 dt/epsilon；
原生先按 `_calc_timestep` 初始化，再按 `_nextclocks` 的最小时钟与相对容差规则推进。
第三个时钟可提前合并突触事件，故不能仅对 main time 逐个 ceil。已有 Brian 对照证明
这一边界，以及 t/t_pre/t_post、不同 dt、非整周期 warmup 和 event-driven lastupdate。

时钟表只读并停止梯度，依赖它的物理状态与参数仍可微。16 组独立有限差分通过，
包括 full/TBPTT、reset detach 与初态。时钟最多 256 种、单请求最多 1000 万 clock
visits（包括 start_tick 之前的重放）；表内存计入 native budget。CPU/Metal/MPI 全量
回归见 `training-clocks-20261003/verification.json`：23 个模块 939 passed / 114 skipped /
0 failed，631.34 秒；68 项时钟测试全部通过，152 个源码哈希和新原生运行程序已验证。
此阶段是一次冻结源码完整运行，没有合并此前版本的测试数字。
连续突触积分仍要求公共 timestep，任意异步 ODE 调度不在本次采样接口中。

下一阶段延迟实现审查：

- 仓库实际使用 Cython CSpikeQueue；`spikequeue.py` 只导入扩展，没有 Python fallback。
  C++ `dt` 是 public 字段，但目前 Cython 未声明/导出。应新增只读 snapshot metadata
  接口并选择性重编译扩展，保留 `_full_state()` 旧协议，避免用不安全内存猜测读 dt。
- 当前延迟对应的新事件可按 delay 降序、source index、原始 edge 顺序执行，并用
  per-edge FIFO state 将门 VJP 送回原始触发 tick。
- 已排队旧事件必须保留每个 bin 的原始 edge 顺序，并在同一到达 tick 的新事件之前
  执行；不能按新的 delay 重排。候选实现是独立 snapshot pending-prefix actions，
  按原始 bin/entry 保序，再执行固定新延迟动作。各事件幅度/历史的梯度与离散调度
  元数据须分离，后者 detached。相同 edge 的旧/新事件可同 tick 到达，需执行两次。
- 若队列旧 dt 不同，忠实模拟 CSpikeQueue.prepare：先按旧 offset 展平，用
  int(i*old_dt/new_dt+0.5) 映射，目标 bin 是赋值而非 append，碰撞保留后写入者。
  这包括更改 delay/dt 后再转换的已有网络，不能假定 pending queue 为空。
- 原生 Trigger 可扩展 state-cell 门，验证 binary 并从 action context 读取历史值；
  反向写入该 cell 的伴随。FIFO shift 与 clear/record 可复用现有动作和 TBPTT。
  全部队列状态需要统一 carry/store/restore、预算、mask 策略与独立差分测试。

以上仍是下一阶段设计，不是延迟队列已完成的声明。整体目标保持 active。

### 延迟队列阶段（本地已验收）

延迟新事件与 pending snapshot 前缀已经实现，不再是上述候选方案。入口支持 fixed
scalar/per-edge pre/post delay，按 CSpikeQueue 的半向上取整构建 binary history；
Trigger.state 读取 taped context 并把 gate VJP 回写历史 cell，shift/record 动作接回
原始 threshold tick。32 组 full/TBPTT 独立参数/初态差分已经通过，含非零历史伴随。

队列旧 dt 通过 Brian 现有具名 C++ capsule + 实际头文件的只读 Cython bridge 获取，
无需修改/重建 Brian 扩展，也没有猜测内存偏移。桥接程序在独立临时目录编译一次，
还原 cache preference 后清理文件；需要本地 C++/Cython。历史事件按原 queue bin/entry
顺序保留，先于新事件；dt 重采样忠实采用覆盖而非 append，包括空 bin 的覆盖。
对应测试覆盖 warmup、变更 dt/delay、旧新同边同时到达、Brian store/restore 和大于
63 的历史长度。随机神经元/突触 + summed + refractory + delay 的共同噪声对照通过。

原生 binary layout/初值/写入校验、前端 history/action/program 预分配预算已接入。
训练状态完整 carry/store/restore，MPI owner 归约协议保持不变。最终受影响六模块
CPU/MPI 回归位于 `training-delays-20261003/verification.json`：420 passed / 0 skipped /
0 failed，166.54 秒；100 项延迟测试通过，另有 4 组 pending mask invariant audit。
157 个源码/头文件/已加载 Brian 扩展哈希与新 Rust 程序身份一致。没有将前轮 GPU
结果合并成新版验收数字。
本阶段没有 GPU 实机运行。动态 GPU、结构 mask 状态迁移、共享/linked 可写状态等
仍未完成，目标保持 active，不能把这个子阶段称为任意 Brian 语义已实现。

### 结构 mask 状态迁移设计（下节记录实现与验证）

当前 `NativeLIFTrainer.update_mask(masks, growth_weight=0)` 在 v5 仍显式拒绝变更。
仅改变 optimizer 权重/momentum 会留下 runtime w、塑性 trace 和延迟队列。需要在 plan
中加入经过原生验证的 edge-mask→state ownership 元数据，迁移时只影响突触私有状态
与其历史；不能从 action.writes 直接推断所有权，因为其中包括别的神经元电位。
共享突触状态可能属于多个 edge；仍有存活 owner 时不可因单边 pruning 被清零。

候选明确策略（尚未实现）：prune 清理禁用边的运行态与全部 pending/new history；
regrow 重建声明初态，runtime w 从 growth_weight/optimizer 初值绑定，history 一律零，
不复活 plan 初始 pending prefix。lastupdate 要使用当前边时钟，而非原 snapshot 时间。
如采用 plan initial 作为其它物理状态基线，需在 API 文档明确；它不是上一段训练的残留。
同一批 mask 变更须验证全部形状、有限值、所有权与状态后原子提交，失败保留 weights、
moments、state、mask、clock/noise。当前静态 update_mask 对 growth_weight 的有限性
也可一并修正。检查 topology identity/checkpoint，给出完整 prune→carry→regrow→restore
的独立参考与 MPI 测试，不能只是取消现有拒绝分支。

动态 GPU 随后需执行 v5 动作、binary history gates、lazy select、clock_time、synaptic
noise 和 full/TBPTT VJP 的同一契约；不能回落 CPU。已有 Meta/ABI v4 与动态 action
核心分开，Metal 可本机实测，CUDA 代码/实测仍须区分。本轮没有新的云端授权。

### 结构 mask 状态迁移实现

已增加原生 `migration.rs` 与前端 ownership 元数据，`update_mask` 现在通过独立原生
边界操作暂存并一次提交 weights/moments、live state 和 masks。仅支持已声明的 edge
mask；原始 v5 IR 缺少迁移布局时仍拒绝变更。neuron/activity 存储受保护，元数据必须
精确覆盖全部 masked synaptic/history writers，不能遗漏 owner 或混入 unmasked writer。

prune 清零私有状态和所有 pending/new history；regrow 从声明/当前 learned initial
恢复其它状态、从 growth_weight 恢复边权，history 始终空。共享 cell 有旧 owner 存活
则保持，否则清零或按新 generation 重建。共享 owner 的原生测试不等于任意 Brian
共享标量路径写入已开放。optimizer step/RNG、神经元状态、clock/noise 游标保持不变。

lastupdate 使用真实分段结束边界时钟。专项 Brian 对照发现第三个近容差时钟可导致
“下一神经元 tick 的采样时钟”多推进一次；新增独立 boundary 重放，按最早时钟的
rounded interval end 停止。普通 SSA clock sample 保留原行为。已通过专项对照及
clock 回归，并加入原生晚期 work-budget 错误的完整回滚/重试测试。

本轮 CPU/Metal/MPI 回归见 `training-migration-20261003/verification.json`：25 模块
最终 1092 passed / 114 skipped / 0 unresolved failures，49 项迁移测试全部通过。
首次冻结源码全量 1091 passed / 114 skipped / 1 failed，681.75 秒；唯一失败为旧
binary_update 测试构造的无 mask writer 被新增 ownership 校验提前拒绝。补上原 owner
mask 后完整延迟模块 100 passed，49.93 秒，按完全相同 test ID 替换该模块结果。
验证器检查这一行测试差异及 159 个最终源码/依赖哈希，原生程序和所有实现源码
在两次运行之间未变，原始失败日志保留。另有 8 组迁移后参数/携带状态独立差分，
覆盖连续/事件驱动、剪枝/再生、full/TBPTT；最大绝对误差约 2e-7 / 2.2e-10。
动态 GPU、共享/linked
可写外部输入、运行中改变延迟和任意异步连续积分仍未完成，总目标保持 active。

### 动态 GPU 设计约束（实现进度见下一节）

源码核对表明静态 GPU ABI `v4r6` 只覆盖固定 neuron-state 程序，不能复用其 header
解释 v5。需要独立版本符号与 metadata：扁平 action/context/program offsets、写入与
触发索引、mask/detached/binary 标志、threshold 引用、物理状态与初值参数绑定。
GPU forward/VJP 必须实际执行 action，不能通过 CPU 求值后仅做设备复制。

每个 sample 可按调度顺序在一个 GPU lane 中执行动作，先保证 alias/顺序准确；优化
并行度以后再做。所有写入来自同一 pre-action context，同时提交；阈值门与 history
门的伴随规则复用 CPU 契约，包括 gate=0 时的 counterfactual F(old)-old。lazy Select
必须用有界栈/visited 图遍历，只执行选中分支，不能用 eager v4 evaluator。
min/max tie、detached 时间、binary validation、TBPTT 与初态绑定都要独立测试。

Metal 不支持递归 evaluator，GPU SSA 可使用显式最多 128 项 DFS 栈，CPU 已验证每个
operand 引用更早节点。共享 Metal/CUDA 数学源码需保持有限值检查及 float32 profile。
时间表和 counter-based noise 可由 Rust 生成并上传，跟随 action noise_domain/entity/
stream 和绝对 tick 索引，确保反向重放与 forward 使用相同样本。staging、device tape、
per-batch gradients、context scratch、clock/noise 表都必须纳入预分配预算。

本地 MPI 阶段需要每个动作后的 owner delta 归约与 apply 阶段，不能把整条动态轨迹
当作静态两阶段 tick 归约。MPI 最終参数梯度每 rank 只归约一次；初态绑定只由一个
rank 加入。single-rank Metal 可先独立验收，但仍需明确 MPI/CUDA 的未完成范围。
结构迁移是独立 host control 操作，不计作 GPU forward/VJP dispatch。

### v5r1 单机动态 GPU

独立 `dynamic_gpu.rs`、`training_dynamic.metal` 与 Metal/CUDA `v5r1` host entry
已接入。GPU 每 sample lane 顺序执行全部动作、loss 和逆序 VJP；Rust 仅准备有界
metadata、f64 调度生成的 f32 time/noise 表、归并梯度并提交 optimizer，没有 CPU
动作求值回落。64 context / 128 SSA 节点边界保留；lazy select 用有界显式栈，设备端
保留二进制 history/activity 检查及 gate=0 的反事实事件梯度。详细接口见
`NATIVE_TRAINING_DYNAMIC_GPU.md`。

首轮 41 项真实 Metal 测试通过，另补 4 组 delayed/history 初态独立差分并通过。
覆盖 full/TBPTT、默认/显式初态、batch、连续/事件 STDP、随机系数独立差分、pending
delay、异步 clock、summed/refractory、惰性域错误、carry/migration/恢复与 optimizer。
完整回归及后续资源审查修正的最终证据记录在 `training-dynamic-gpu-20261003/`：
26 模块 1137 passed / 159 skipped / 0 failed，752.41 秒。随后只修改
`dynamic_gpu.rs` 的 Vec 预留容量，避免大型 noise/input 暂存几何扩容超出元素数预算；
重建原生程序后整个动态 GPU 模块 45 passed / 45 CUDA skips，33.73 秒。最终验证器
按相同 test ID 替换该模块，检查 162 个最终源码/依赖哈希并保留两版 runtime hash、
初始源码和精确 diff；device math 与其他实现源码未变。最终总数没有混入旧版本结果。
CUDA 翻译成功生成五个 kernel 和动态 evaluator，但没有 nvcc 或 NVIDIA 实机证据。

动态 GPU MPI 仍由前端/native 明确拒绝，是下一步实际实现任务。现有单机 kernel
一个 dispatch 跑完整轨迹，不能直接放开 MPI 参数：必须加入每动作 owner delta 的
设备 compute、host MPI 归约、设备 apply 阶段，确保跨分区读写发生在正确位置。

候选 v5r2 ABI：独立 control block 保存 rank/ranks/phase/tick/action，action header
写入已有 owner，增加每 batch 至少 129 项 delta scratch。前向 transform 输出完整
new-value 向量（gate=0 不应用），threshold 输出 spike。反向与 CPU 对齐：c 项 read
adjoint、w 项保留的旧 target adjoint、1 项 gate adjoint，owner 归约后由所有 rank
执行相同 apply。threshold 反向归约 voltage adjoint；threshold 参数梯度留在 owner。
参数梯度最终只在 Rust 归约一次，默认初态绑定梯度只在 rank 0 加入。分段 TBPTT 清理
每个 rank 的物理 adjoint，但保留当前参数梯度。独立 scratch 和 staging 均计入预算。

host GPU 失败可返回 Rust；`mpi::Context::Drop` 在未 finish 时调用 MPI_Abort，不能
吞掉任一 rank 的错误或让其他 rank 无限等待。新版需覆盖 2/8 rank、闲置 rank、同机
共享物理 GPU、带 delayed history/随机流/alias 的 VJP、checkpoint、失败回滚与硬件
计数。可以保留 single-rank 单 kernel 路径，但不可将所有 rank 重复算完整轨迹当作
owner 分布式执行。跨机器继续按用户要求暂缓。

### v5r2 动态 GPU MPI

上述 owner compute/collective/apply 已实现，新增 `training_dynamic_mpi.metal`，
Metal/CUDA `v5r2` native symbols 和 metadata control block；Rust 写入 action.owner、
rank/ranks，按阶段计入真实调度的 GPU dispatch 数。single-rank 保留原有一个 kernel
执行完整轨迹。每动作 MPI 前向归约新状态值/threshold spike，反向归约 c+w+1 项
delta。设备 apply 负责 alias 累加、history gate、TBPTT 清理；初态绑定只 rank 0，
参数梯度由 Rust 最终只归约一次。设备域错误通过非有限 collective delta 拒绝，
GPU/host 错误由未完成的 Rust MPI context abort，无 Python 状态提交。

本机 2-rank smoke 前向/全部梯度与单机 Metal 完全一致。首批 21 项 Metal MPI
测试通过；追加自连接 pre/post 别名、双写与 refractory 组合后 29 项专项通过。
冻结源码受影响七模块回归位于 `training-dynamic-gpu-mpi-20261003/verification.json`：
215 passed / 134 skipped / 0 failed，244.05 秒；29 项动态 Metal MPI 和 45 项单机
动态 Metal 全部通过，164 个源码/依赖哈希与新原生程序身份一致。没有将此前 26 模块
全量数字混入本次计数。测试中的 2/8 ranks
共享本机一张物理 Metal GPU，不是多物理 GPU 或跨主机验收。CUDA 六 kernel 翻译通过，
仍没有 nvcc/NVIDIA 硬件证据。本轮没有上传或启动云计算任务。

### shared/linked 状态接口审查（实现前记录）

本地剩余主要实现任务是 shared/linked 可写/外部状态，以及运行时 delay 变更、任意
异步连续积分等。应先用仓库 Brian 实际运行验证合法语义，避免把 Brian 本身禁止的
scalar pathway 写入列为兼容目标。当时的前端会明确拒绝 mutable shared/linked neuron
storage 及 linked Synapses，而不会把它们悄悄当常量。

已选群组内部的 linked storage 应通过真实 variable/index 映射到 canonical physical
cell，让重复读取的 VJP 累加到同一伴随，而不是为每个别名复制可演化状态。必须覆盖
重排/标量索引、reset/event 写入和 self-alias、初态参数绑定、refractory、summed、
MPI owner 与 migration ownership。外部可变状态不能只快照初值：需显式时间输入契约
或导入其更新动作；任意 Python callback 不能隐式当可微 SSA。外部输入若可微还需
明确返回其 B/T/state 伴随与调用边界截断规则。这是实现前的接口审查；本轮进展见下节。

### 已选神经元 shared/linked 物理映射（2026-10-03）

本轮通过 Brian Variable 身份和真实 `_idx`/`0`/固定整型数组索引建立 canonical
physical storage。shared 浮点神经元参数只保留一个实际 cell；NeuronGroup 和 Synapses
的 linked 参数可读写已选神经元状态，支持重排、重复索引、自引用、标量广播和链式引用。
原矩形 ABI 前缀继续分配，但别名占位槽默认清零、detached；真正的值和初态伴随通过
`provenance.neuron_state_layout` 读取，突触映射仍用 `dynamic_state_layout`。

神经元动作现在仅写实际赋值目标，避免未赋值 alias 的 identity 输出覆盖已写源状态。
同一物理 cell 的多个逻辑赋值保持 Brian Cython 的独立局部变量、排序写回最后胜出
语义，VJP 则累加所有别名读取。神经元所有的状态不会作为突触私有状态被剪枝重置。
Brian 本身不允许 scalar/shared reset/pathway 写入，本轮显式拒绝这类模型。

新增独立 Euler 递推/固定 surrogate anchor 有限差分验证 canonical 初态和权重梯度；
真实 Brian Cython 验证 Euler/RK2/RK4 下的读写、自引用、重复写回、refractory 和
summed 联动，并验证 CPU、Metal、2/8 本机 MPI、carry/checkpoint/mask 边界。
本轮最终统一回归证据写入 `mpi-evidence/training-linked-20261003/`。

额外 Brian 实测发现 `(constant)` 整型索引仍可被 pathway 赋值，不能只检查声明标志。
前端现显式拒绝事件对 constant 参数/固定索引 storage 的写入；这不是声称 Brian
禁止该写法，而是该模式需要尚未实现的运行时索引/参数更新语义。修复前的广泛回归
在 494 passed / 89 skipped 时主动停止并单独归档，不计入最终冻结版本验收。

这里没有实现外部可变状态、跨突触 storage 链接、constant optimizer bank 别名、
运行时变化的索引或延迟，以及任意异步连续积分。CUDA 最新实机和跨主机仍无新增
证据；未使用云端资源，整体目标保持 active。

最终验收：8 个受影响模块的去重结果 **447 passed / 37 skipped / 0 未解决失败**；
新增 linked 套件为 63 passed / 15 CUDA skipped。完整运行中有一个旧错误提示断言
不匹配，产品仍正确拒绝非法 shared reset；只更新该断言后，多状态模块 7 项重跑
全部通过。验证脚本证明产品源码未变，仅测试中这一预期字符串不同，并用重跑的
7 项替换原模块结果而不重复计数。172 个最终源码/依赖哈希与原生程序身份核对通过。

### 合法的跨对象 Synapses 链接与引用生命周期（2026-10-03）

Brian `LinkedVariable` 构造器明确拒绝 `DynamicArrayVariable`，所以直接链接逐边
突触数组不是其支持的 Brian 语义。本轮真实构造测试证实这一限制；没有通过 monkey
patch 绕过它，也不再把非法模型当作仍欠缺的兼容能力。另一个真实限制是通过 Synapses
转发已经间接索引的引用：动态索引数组不能用于预计算这类组合索引，测试使用合法的
直接源索引，并由 Brian Cython 独立执行。

本轮实现可链接的 shared Synapses 源，以及经 Synapses 转发的神经元存储。前端先
分配全部真实存储、再解析引用，名称排序不影响接受能力；非 constant 的 shared 源
成为运行时状态，其优化器绑定训练初值，所有消费者的 VJP 汇入同一物理 cell。

源边与引用边共同保持 shared 状态的生命周期。仍有旧引用存活时保留状态，全部引用
被剪枝后清零，新一代恢复声明/学习到的初值。神经元这种未受掩码控制的引用则长期
保持源状态。编译器用单 cell identity 动作表达这些引用，其数值变换与 VJP 都是
恒等的；原生迁移从 unmasked 动作推导永久存储，拒绝对这些 cell 声明可剪枝所有权。
检查点、carry 与 mask 更新仍然保留原先的时间/RNG/优化器边界。

验收记录在 `mpi-evidence/training-cross-linked-20261003/`，最终结果以其验证输出为准。
仍需继续实现 constant 参数 bank 链接、外部可变输入、运行中索引/延迟变化和任意
异步连续积分。CUDA 最新版本仍缺 NVIDIA 实机证据；跨机器按用户要求暂缓，整体
目标继续 active。

本阶段最终冻结验收：5 个模块 **211 passed / 100 CUDA hardware skipped / 0 failed**，
416.58 秒；新增跨对象套件 25 passed / 11 skipped。173 个源码/依赖哈希和重建的
原生程序身份一致。没有把前面的非法模型试验、修正前 fixture 或中断记录混入结果。


### 常量参数链接、索引阈值与唯一优化器存储（2026-10-03）

本轮支持 selected NeuronGroup 常量数组/标量与 Synapses shared 常量作为链接源，
包括源局部方程未使用但消费者使用的参数。源变量仅分配一个优化器 bank；
消费者的固定重排、重复索引与广播通过原生索引表读取，梯度回收到相同物理槽。
阈值别名使用逐神经元物理参数引用，正值检查覆盖每一个引用。

新增 SSA opcode 24 与有界 parameter_maps；CPU、Metal、局部 MPI 使用相同映射，
CUDA 共用 shader 翻译但未运行 NVIDIA 硬件测试。常量别名不会成为携带状态副本，
优化器更新与 checkpoint 恢复后直接读取当前参数。只有源变量可选择为训练参数，
冻结源保持冻结。4100 个不同索引的测试检查程序数量不随神经元数线性膨胀。

独立 Euler/固定替代梯度差分、真实 Brian Cython Euler/RK2、Metal 与 2/8 本机
MPI、优化器 Adam 单次更新及 checkpoint/carry 都纳入本轮验证。最终冻结结果记录在
`mpi-evidence/training-constant-links-20261003/`；先前 fixture 使用 Brian 单位保留名
和沙箱内无法访问 Metal/MPI 的探索运行不计为通过证据。

仍需外部可变输入、运行中索引/延迟变化及任意异步连续积分；最新版 CUDA 尚待实机。
跨机器按用户要求暂缓，整体目标继续 active。

本阶段最终冻结验收：8 个模块 **278 passed / 113 CUDA hardware skipped / 0 failed**，
565.01 秒；常量链接新增套件 34 passed / 9 skipped。174 个源码/依赖哈希、
原生程序哈希和测试唯一性检查全部通过。没有将探索运行重复计入最终结果。


### TimedArray 外部输入、随机幅度路径梯度与原生输入更新（2026-10-03）

本轮新增原生 SSA opcode 25，支持一维/二维 TimedArray、SI 单位、负时间与末端保持、
RK/Heun 阶段采样及神经元/突触 ODE/SDE、reset 和事件读取。同一表跨对象共享一份
冻结参数 bank，表值梯度正常返回；离散的时间 bin 和列索引不求导。前端向 Brian
积分器提供已解析的函数，保留 Euler 对局部常量噪声系数及 Heun/Milstein 的限制。

真实 Brian 对照确认取整精度由 CodeObject.owner 的组时钟决定，事件 runner 的
执行时钟不能替代这个时钟。专门覆盖不同突触时钟和接近 bin 边缘的读取。连续
突触异步积分仍保持明确拒绝，尚不能宣称任意异步语义完成。

新增 `NativeLIFTrainer.update_timed_input(bank, values)`；原生层在序列边界检查并
替换冻结表值，保留优化器、携带状态、时钟和随机流，支持首次运行前和 checkpoint
恢复后使用。形状/采样网格固定；GPU 更新拒绝不可表示的 float32 值。

新增测试包含真实 Brian Cython、独立固定噪声/替代梯度差分、refractory 分支、
CPU/Metal 与 2/8 本机 MPI、更新原子性、内存预算、热启动和单位。最终冻结证据在
`mpi-evidence/training-timed-inputs-20261003/`。前面的 fixture 修正和采样时钟修正
仅是探索记录，不与最终验收合并计数。

仍需运行中索引/延迟变化、任意异步连续积分、动态阈值和其它尚未覆盖的 Brian
语义；定长外部时间序列及序列边界表值更新已经实现，但任意外部回调/流式协议
不因此视为完成。最新版 CUDA 尚待 NVIDIA 实机，跨机器依用户要求暂缓。

本阶段最终冻结验收：8 个模块 **381 passed / 109 CUDA hardware skipped / 0 failed**，
705.58 秒；新增 TimedArray 套件 60 passed / 13 skipped。177 个源码/依赖哈希、
原生程序哈希与测试唯一性全部核对通过，所有跳过项均明确为 CUDA。目标继续 active，
本轮没有云作业、跨机器测试或多物理 GPU 验证。


### 状态/时间/输入依赖的动态比较阈值（2026-10-03）

本轮实现单个 `>`、`>=`、`<`、`<=` 比较，左右两侧可以包含状态、参数、链接/共享
存储、时间、TimedArray 和支持的数学函数，展开神经元子表达式并检查单位一致。
比较坐标不再必须是电压，可表达 `t >= onset`。零/负阈值以及优化器更新后过零
均可执行；未改变静态 v3/v4 和手写旧式阈值引用的约束。

前端生成定向差值 margin，原生 action 在阈值位置计算 scratch cell，然后施加
严格/非严格发放与替代梯度。margin 的 VJP 传播到比较两侧；共享状态保持唯一
物理存储。原生验证要求对应 owner 的先行无条件写入，不允许把神经元电压 cell
冒充 margin、使用 detached/初值参数绑定的 margin 或把比较标志放在普通 action。
GPU 使用既有 16 槽 action header 的第 14 槽，CPU/Metal/本机 MPI 共用算法。

独立固定噪声/替代梯度差分覆盖参数、输入表和初始状态；真实 Brian Cython 覆盖
Euler/RK2/RK4、子表达式、不应期、时间单位与共享突触链接。检查点/carry、输入
更新、四种比较的相等边界和梯度符号、SGD 阈值过零都纳入验证。
最终冻结验收在 `mpi-evidence/training-dynamic-thresholds-20261003/`。旧的常量链接
“负阈值必须拒绝”测试改为接受非正动态阈值并核对独立梯度，测试数量没有变化。

复合布尔/相等/链式阈值、运行中索引/延迟变化、任意异步连续积分以及任意外部
回调仍待继续；CUDA 最新路径待 NVIDIA 实机，跨机器依用户要求暂缓。整体目标 active。

首轮完整回归的 36 项失败已保存在上述证据目录 `attempt-1/`。原因是旧不应期
测试假设辅助状态固定为 4 个，以及独立 oracle 未写入新 margin cell。修正后仍
检查全部最终状态、参数梯度和初始状态梯度；36 项定向复测全部通过。完整冻结
回归重新执行，最终结果以 `verification.json` 为准，不能合并两轮通过项计数。

本阶段最终冻结验收完成：10 个模块 **439 passed / 105 CUDA hardware skipped /
0 failed**，750.52 秒；新增动态阈值套件 53 passed / 9 skipped。178 个源码/依赖
哈希、原生程序哈希、测试身份唯一性和完整数量全部验证通过。CPU、Metal 与
2/8 本机 MPI 为实际运行；没有云作业或跨机器验证。整体目标继续 active，下一步
仍需补齐复合布尔阈值、运行中索引/延迟变化、任意异步连续积分和外部回调。


### 复合布尔阈值和短路梯度（2026-10-03）

本轮实现 `and`、`or`、`not`、布尔常量和相等/不等判断；保留从左到右短路，
未执行分支不计算有域限制的表达式。逐比较检查单位，展开布尔子表达式。
链式比较为前端便利扩展：当前 Brian renderer 不接受链式语法，使用展开为显式
有序 and 后的真实 Cython 执行作对照，不能称 Brian 直接执行了链式字符串。

梯度契约写入 provenance：有序比较沿 SI 差值使用 fast-sigmoid surrogate；
已执行 and/or 分别沿 a*b 与 a+b-a*b 求导，跳过右侧时伴随穿过左侧（右侧仅在
该导数约定中取中性值）。not 取负伴随，==/!= 保持精确离散值并停止比较梯度。
最终 threshold action 直接传递 predicate 伴随，避免对二值输出重复施加 surrogate。
这是明确选择的事件替代梯度，不声称离散真值函数存在这样的经典导数。

新增 SSA opcode 26–32 和 action predicate 模式，CPU/Metal/本机 MPI 使用一致契约；
原生验证检查 Boolean 类型、SSA 引用、系数与计划一致及同 owner 先行写入。
独立决策树差分、固定噪声/可塑性/full-TBPTT、真实 Cython Euler/RK4、不应期、
短路/执行分支域错误、相等边界、输入更新和 checkpoint/carry 均已加入专项测试。
本轮冻结证据将保存在 `mpi-evidence/training-boolean-thresholds-20261003/`，仅以
最终验证文件中的结果为准；不把探索运行相加。

运行中索引/延迟变化、任意异步连续积分、可变整数/布尔存储及任意外部回调
仍未完成；CUDA 新路径待 NVIDIA 实机，跨机器仍暂缓。整体目标继续 active。


### 后续离散状态的独立 Brian 参考（2026-10-03）

在复合阈值完整回归仍运行时，保存了独立参考
`mpi-evidence/training-discrete-state-reference-20261003/`。真实 Brian NumPy（事件
采用顺序 Python loop fallback）结果与手算逐 tick 轨迹一致：神经元 enabled/count、
逐边 ready/quota、路径写入后 reset 再翻转 Boolean、以及正负小数 int 向零截断。
默认 count 为 int32，enabled 为 bool。源码和探针哈希一并保留。

这只是下一阶段的参考数据，当前 native 前端仍拒绝非浮点可变状态；不计入 CPU/GPU
通过数量。实现还需覆盖离散赋值/强制转换、停止离散状态伴随、溢出/精度边界、
随机幅度读取，以及 carry/checkpoint/migration 和 Metal/MPI。复合阈值冻结源码未改动。


复合布尔阈值阶段最终冻结验收已完成：12 个模块 **512 passed / 120 CUDA hardware
skipped / 0 failed**，pytest 耗时 2039.22 秒。新增套件 56 passed / 11 skipped。
632 个收集身份与执行 XML 逐项一致，179 个源码/依赖哈希、原生程序哈希及成功
退出状态全部核对通过。首轮意外中断原因未知，原始记录保留在 `interrupted-1/`，
没有合并其部分结果。CPU、Metal、2/8 本机 MPI 为实际运行；没有新增云作业。

该阶段实现已经验收，但可变整数/布尔状态仍只有 Brian 参考、没有 native 实现；
运行中索引/延迟变化、任意异步连续积分、外部回调和新版 CUDA 实机仍未完成。
跨机器按用户要求暂缓，整体目标保持 active。

### int32／Boolean 可变状态实现（2026-10-03，本地验收完成）

上文“只有 Brian 参考、没有 native 实现”是前一阶段状态。本轮已加入 native
int32/Boolean 读写、顺序赋值转换、离散比较和停止梯度，并接通 Brian 神经元、
动态突触、shared/linked 别名和冻结常量 bank。整数加减乘、取负和 min/max 使用
int32 环绕语义；显式 int 转换向零截断，越界或非有限值拒绝。类型不匹配的手写
SSA、可训练整数参数和非法状态快照也拒绝。

GPU 将整数作为原始 32 位位模式保存，避免超过 2^24 时丢精度；浮点混合运算仍
遵循 Metal/CUDA 的 f32 精度。v5r3 ABI 用 metadata[28] 标记整数 cell，本地 MPI
delta stride 为 130，最后一格单独传递错误状态。合法负整数的 NaN 位模式不再与
域错误混淆。主机在归约前解码为精确 f64 整数，归约后重新编码。旧 v5r2 动态
入口不能替代新入口，否则会少分配 scratch。

Brian 前端保留默认 int32 与 Boolean dtype；每条状态赋值立即插入转换，再进行
后续语句替换。浮点权重和受这些离散值影响的连续路径仍可训练，但不对计数器或
Boolean 开关求连续梯度。携带状态／检查点保存原值，剪枝清零被拥有的离散突触
cell，新一代恢复声明的初值，神经元别名继续保留其物理生命周期。

新专项探索已覆盖原生 CPU、实际 Metal、2/8 本地 MPI、独立权重差分、Euler/RK4
真实 Cython 逐 tick 轨迹、整数上下界、非法分支原子失败，以及离散开关控制的
Heun 随机幅度、full/TBPTT 和 checkpoint/carry。Cython 对隐式浮点赋给整数会
编译失败，因此真实 Cython 对照使用显式 int(v)，并断言其 reset block 已编译。
原生的隐式截断扩展只与独立预期／Brian NumPy 行为对应，不冒称 Cython 支持。

最终验收保存在 `mpi-evidence/training-discrete-state-20261003/`：16 模块、912
个测试身份、181 个源码/依赖哈希。原始运行和最终修复后的运行分别冻结、保存
完整 XML；通过总数按身份替换后核算，不能将探索结果或重叠运行相加。

整除／取余的下一阶段 Brian NumPy/Cython 参考另存于
`mpi-evidence/training-integer-division-reference-20261003/`，两者与独立 Python
整数运算一致；这只是参考，不代表原生已实现。除零、INT_MIN/-1、混合精度另需
定义和测试。运行中索引／延迟变化、任意异步连续积分、外部回调及新版 CUDA
实机仍未完成；跨机器继续暂缓，总目标保持 active。

本轮最终验收已完成：**743 passed / 169 NVIDIA hardware skipped / 0 未解决失败**，
覆盖 16 模块的 912 个唯一身份。记录在
`mpi-evidence/training-discrete-state-20261003/final-verification.json`；
`verify_final.py` 已核对 181 个最终源码/依赖、实际程序哈希及所有原始／替代身份。

完整长进程在最后模块因垃圾回收明显变慢。单次进程采样显示栈在 gc_collect；
已优雅中断并确认退出，前 15 模块的 775 个测试身份完整保存。最后 RK/refractory
模块独立完整运行得到 125 passed / 12 skipped；原有部分该模块结果全部排除，
不重复累计。未来广泛回归应按模块启动干净子进程，仍使用同一冻结源码和程序。

最终审查另发现 GPU 打包 collect 可能自动扩容，已改为精确预分配；使用独立构建
先验证，没有修改运行中的基线。两项非 root MPI 原子失败测试因新错误文字丢失
旧 `nonfinite` 关键词而失败。已在 Metal/CUDA bridge 恢复兼容关键词，最终原生
整数及动态 GPU MPI 重测 **74 passed / 47 skipped / 0 failed**，包含 2/8 rank
失败后状态、时钟和 optimizer 不变的断言。基线／分配候选的失败日志全部保留，
由相同身份的最终重测替换；通过数量不把重叠运行相加。最终仅 GPU 打包 Rust
文件和两个 bridge 的错误文字与基线不同，完整差异及前后版本均存档并校验。

因此可变 int32/Boolean 状态已接通并获本地验收，不再只是参考数据。整除／取余
仍只有独立 Brian 参考；int64/unsigned 状态、超出 int32 的整数字面量、位运算、
运行中索引／延迟变化、任意异步连续积分和外部回调仍待实现。新版 CUDA 没有
NVIDIA 实机证据，跨机器继续暂缓，整体目标保持 active。

### 整除／取余与顺序赋值（2026-10-03，本地验收完成）

上文“整除／取余仍只有参考”是前一阶段状态。本轮已实现 int32 floor division／
remainder、浮点 floor division／remainder 和取余分段 VJP，接通 `//=`、`%=`、
`floor`、随机系数、时间表达式、神经元 reset 和动态突触事件。整数运算保持
精确 int32，`INT_MIN/-1` 显式环绕；执行到除零时整体失败，未执行分支不求值。

真实 Cython 参照揭示 `%` 与 `%=` 的边界差异：Brian 表达式 renderer 额外执行
`((a%b)+b)%b`，增量赋值直接执行余数。前端已保留这一区别，包含中间整数
环绕与浮点消去；不冒称 NumPy 边界等价。Brian 单位检查补充 `//=` 和右侧括号。

Metal/CUDA 动态入口升级为 v5r4，沿用 v5r3 buffer 布局，新增 integer selector
5/6 与浮点 opcode 46/47；旧库不能静默把新 selector 当成 max。CUDA 源码转换
已检查，但没有 NVCC 编译／NVIDIA 实测，也没有新增云作业。

冻结验收在 `mpi-evidence/training-division-20261003/`：**454 passed / 174 NVIDIA
skipped / 0 failed**，10 模块、628 个唯一身份、187 个源码／依赖哈希。原生程序
哈希为 `0edc668de99944069a85495e70bd76c77af6821f28753ee4f54aed2fc5cd3492`。
全部模块独立启动干净进程，验证收集身份、完整 XML 和正常退出；早期探索失败
另存，不混入最终统计。CPU、实际 Metal 和 2/8 进程本地 MPI 为实测。

新专项覆盖 103 passed / 50 NVIDIA skipped：独立整数参照、两个浮点操作数的
有限差分、实际 Cython 顺序事件、懒分支／除零原子失败、full/TBPTT 与固定噪声
checkpoint/carry。既有整数状态、动态 GPU/MPI、SDE、Brian 动态突触、积分器、
原生 SSA 与表达式解析回归均通过。

运行中索引／延迟变化、任意异步连续积分、外部回调、int64/unsigned、超出 int32
的整数字面量与位运算仍待实现。新版 CUDA 仍无 NVIDIA 实机证据；跨机器继续
暂缓，整体目标保持 active。

### 运行间延迟更新（2026-10-03，本地验收完成）

已实现原生 `NativeLIFTrainer.update_delays`：按 pathway 设置标量或原始边顺序的
延迟，旧排队事件保留到达时间和顺序，新发出的事件使用新延迟。队列重建支持
多代同时到达、独立 batch 历史、空路径、半步取整、长历史拆分及空槽回收；
神经元／突触物理索引、优化器、时钟和 RNG 序列保持不变。配置边界为离散、
detach 操作；下一段 full/TBPTT 保留历史幅度和模型参数的正常 VJP。新 checkpoint
使用更新后的 trainer.plan，旧 plan 的 checkpoint 明确拒绝。

真实 Brian Cython 轨迹和独立延迟/STDP 数值差分已核对。首次回归发现空 summed
路径被过度拒绝，以及旧独立 oracle 未记录 threshold margin scratch cell；均已
修复并保留原始失败、修复前源码和精确差异。第二轮冻结 19 模块验收取得
1,031 passed / 224 NVIDIA skipped / 0 failed，记录在
`mpi-evidence/training-delay-update-final-20261003/`。

另发现事件里的 `delay = ...` 被误当成局部变量而静默丢弃。完整基线正常退出后，
只增加 pathway-owned storage 写入检查，并重跑新专项及两组受影响模块。
pre/post 直接与增量赋值现在给出明确的 pathway 错误，普通局部变量的轨迹和 VJP
保持原样。三模块重测为 97 passed / 17 skipped，重叠身份替换基线而不重复累计。

最终为 **1,043 passed / 224 NVIDIA skipped / 0 failed**，20 模块、1,267 个唯一
测试身份、191 个源码／依赖哈希。完整记录及可重复校验脚本位于
`mpi-evidence/training-delay-pathway-guard-20261003/`，可执行文件 SHA-256 为
`c2c0f16ffae3b756cf258e0977bd3822757b3ede6a8930a7cd1a145e2d9e5708`。
CPU、实际 Metal、2/8 进程本地 MPI 为实测；没有启动云作业。

事件内修改 pathway delay 的原生语义仍待实现。目前已有独立 Brian 参考，确认
变量立即改变、队列到下一次 run 才锁定新延迟；需要继续接通独立物理存储、
每 batch 不同延迟及自动事务迁移。参考和要求在
`mpi-evidence/training-runtime-delay-reference-20261003/README.md`。运行时索引变化、
任意异步连续积分、外部回调、int64/unsigned、超出 int32 的整数字面量和位运算
仍待实现。新版 CUDA 实机验证未完成，跨机器继续暂缓，整体目标保持 active。


### 事件内动态延迟（2026-10-03，本地验收完成）

上一阶段为防止静默丢弃而拒绝 `delay` 赋值；现在已实现 pre/post pathway 的独立
物理延迟存储、SI 单位读写与顺序/增量赋值。事件执行时变量立即改变，队列仍按
本次 run 开始时锁定的延迟发出事件；下一次调用准备新的路线，旧排队事件保留
到达时间和排序。每个 batch 可以为同一条边计算不同延迟，不能用全局广播替代。
共享标量延迟用一个规范单元读取，Brian 不允许的标量事件写入继续明确拒绝。

原生准备阶段有界重建路线与历史，保持模型索引、参数 bank 和优化器不动。
成功训练原子提交 plan/live state；evaluate/gradients 不修改 trainer，返回的
updated_dynamic 描述新 final_state 布局。初态伴随映射回请求原始布局。物理延迟
参与模型表达式时正常求导，整数到达时刻、路线选择和空历史回收明确 detach；
结果通过 gradient_scope 声明这一约定。无效延迟、预算不足和非主 MPI rank 中的
执行错误均保留原来的 plan、状态、时钟、RNG 和优化器。

冻结最终验收 **792 passed / 119 NVIDIA skipped / 0 failed**，19 模块、911 个唯一
测试身份、197 个源码/依赖/文档哈希；CPU、实际 Metal、2/8 进程本地 MPI 已实测。
原生程序 SHA-256 为
`da68c2cd25e17b3b91184bb39c1eea7064cef7184c72892013b255ff543f5b70`。
四个新增/更新专项在该总数内为 67 passed / 15 skipped。实际 Cython 多次 run 轨迹
及独立事件桶有限差分覆盖参数和每个原始初态单元，并核对随机 carry/full/TBPTT、
mask 迁移、checkpoint、refractory、RK4、异步事件时钟、summed 与共享延迟梯度。
早期探索及旧版本结果不混入本次统计，没有运行云作业。

完整证据：`mpi-evidence/training-event-delay-final-20261003/README.md`。所有测试及
原始校验完成后，仅修正文档中两处过时的 shared/linked 限制；前后文件和哈希
单独保存，最终校验仍严格匹配全部原始实现、测试和可执行文件。

运行时链接索引变化仍待原生实现；已补充真实 Cython 和独立顺序递推参照，记录在
`mpi-evidence/training-runtime-index-reference-20261003/`。它确认 reset 中修改索引
可能使 RHS 读取旧单元、写回新单元，后续神经元又看到前一次写回，不能静态展开
后假称支持。该目录只有参考验证，不计入原生完成。任意异步连续积分、外部回调、
int64/unsigned、超出 int32 的整数字面量和位运算仍未实现。新版 CUDA 实机验收
未完成，跨机器继续暂缓，整体目标保持 active。


### 运行时索引：原生 CPU/MPI 核心（2026-10-03，阶段验收完成）

新增原生有界索引表与间接读写。动作读取时保存真实物理单元，写回可使用顺序赋值后
的整数输出索引；所有写回地址先解析，再按输出顺序提交，碰撞时最后一个写入生效。
反向保存/使用实际读写地址与旧目标值，只对最终写入传播浮点伴随。整数索引与选择
detach，gradient_scope 显式声明；确定性、固定 counter-noise、full/TBPTT、分样本
索引、嵌套查找、carry/checkpoint 与 CPU 2/8 进程 MPI 已验证。

独立递推对每个浮点初态和参数做有限差分；实际 Brian Cython 轨迹核对在 reset 中
读取旧索引单元、修改索引并写回新单元的语义。此处原生 IR 为显式构造，不能视为
Brian 前端已经自动转换可变索引。GPU 新索引寻址尚未接通，目前明确拒绝，避免
静默使用固定地址。新索引的 GPU 项仅验证这个拒绝边界，既有 Metal/MPI 功能另有
实际设备回归。

索引表类型、深度、范围和内存有界检查；非主 MPI 计算的越界也保留完整事务状态。
新入口不能绕过受管延迟队列保护；重建预算包括克隆索引元数据。掩码迁移归属枚举
全部候选写入目标，补充 CPU/2/8 MPI 的清空、恢复及缺失候选归属拒绝测试。

主冻结验收为 321 passed / 106 NVIDIA skipped；完成后补充六项候选迁移测试，
实现和原生二进制不变。精确身份去重及哈希交叉校验后的合计为
**327 passed / 106 NVIDIA skipped / 0 failed**，10 模块、433 个唯一测试身份、
198 个源码/依赖哈希。新核心和迁移专项在该总数内为 44 passed / 1 CUDA skipped。
原生程序 SHA-256 为
`7c88dbee9a3502260c0c6adf42cb6bae443586f215e1daa72e6b9298fc4ff81b`。
完整记录：`mpi-evidence/training-runtime-index-core-final-20261003/README.md`。
原始夹具失败及修复前源码保留，未混入最终统计。没有云调用或跨机器测试。

下一步仍须接通 GPU 寻址及 owner-compute MPI、Brian 可变索引自动转换、运行时
常量 bank gather，以及 shared/linked/summed/refractory/delay 的完整组合。
`NATIVE_TRAINING_RUNTIME_INDEX.md` 和证据目录 `DEVICE_NEXT.md` 明确这些接口。
任意异步连续积分、外部回调、int64/unsigned、超出 int32 的整数字面量、位运算和
新版 NVIDIA 实机验证也尚未完成。跨机器继续暂缓，整体目标保持 active。


### 运行时索引：GPU 与本地 MPI（2026-10-03，阶段验收完成）

原生运行时索引已接入 v5r5 Metal/CUDA 共享内核及 owner-compute MPI。设备保存真实
读写地址和旧目标值，先解析全部目标再写回，碰撞时最后写入生效；反向只对最终写入
求导并向实际读取单元累加伴随。整数/布尔值保持精确类型，索引选择 detach。旧 v5r4
库会因缺少 v5r5 符号被拒绝，避免静默忽略地址元数据。

MPI owner 在 GPU 上计算值和目标地址，其他进程按统一顺序应用；整数值与地址采用
不同传输解释。最大 64 读/64 写、分样本/嵌套索引、固定噪声 full/TBPTT、carry、
checkpoint、mask 迁移、非主进程越界回滚和内存预算均有本地实测。前向和 VJP 未用
CPU 动作执行器替代 GPU。

最终冻结验收 **407 passed / 230 NVIDIA-only skipped / 0 failed**，11 模块、637 个
唯一测试身份、199 个源码/依赖哈希。新增 GPU 寻址模块在总数内为 51 passed /
50 skipped；实际 Metal 和 2/8 进程本地 MPI 均运行。原生 SHA-256：
`e982a79af601199b1913520bbf388b249f0f2f1a6a322e93d1511ebe5902d903`。
报告：`mpi-evidence/training-runtime-index-gpu-final-20261003/README.md`。
最初三项测试夹具错误及原始日志保留，修正后全模块重跑；探索结果未混入最终计数。
所有模块和主执行进程均已退出成功。

CUDA 源码转换检查覆盖六个 kernel 和十三个动态设备函数；尚未用 NVCC 编译或在
NVIDIA 实机运行。本轮没有上传、云作业或跨机器测试。自动 Brian 可变链接转换、
运行时常量 bank gather 及 shared/linked/summed/refractory/delay 完整组合仍待实现。
`FRONTEND_NEXT.md` 记录了实际阻断点和编译 Brian 参照要求。任意异步连续积分、
外部回调、int64/unsigned、超出 int32 的整数字面量及位运算也仍未完成。
跨机器按用户要求继续暂缓，整体目标保持 active。


### Brian 可变物理链接自动转换（2026-10-03，阶段验收完成）

`lower_brian_dynamic_training` 现已把选择范围内的可变 int32 链接转换为原生寻址，
覆盖神经元积分、reset、阈值准备及突触事件变换。读取逻辑局部变量后执行顺序赋值，
再按 Brian 的逻辑写回顺序解析目标；最终被另一个固定别名覆盖的整数局部值仍可
决定其他写入的地址，不能在转换时提前删除。原生 indexed 动作现允许并正确处理
这种固定目标碰撞，固定地址动作的原有唯一目标检查仍保留。

仅为实际使用的表达式/写回目标解析索引。旧索引未被表达式读取时，可以先设为有效
常量再执行常量链接写入。索引源须属于选中的 canonical storage；地址表、provenance
及动作元数据均计入预算。迁移/保留规则枚举全部候选单元。新的 runtime_index_layout
记录可变视图，常规状态布局中的对应单元是 inert placeholder。

冻结最终验收 **394 passed / 127 NVIDIA-only skipped / 0 failed**，9 模块、521 个
唯一测试身份、200 个源码/依赖哈希。新增自动转换专项在总数内为 64 passed /
25 skipped，包括实际 Brian Cython 的 Euler/RK2/RK4 轨迹、阈值读取、整数别名及
突触索引变更；每个浮点初态/训练参数的独立有限差分覆盖固定噪声 full/TBPTT。
CPU、实际 Metal 和 2/8 进程本地 MPI 的批次索引、carry/checkpoint 与越界回滚均通过。
原生 SHA-256 为
`1a33608cf36640b4ae53f50c45023744f397d1b02c852bcce6f7a15a19c647c8`。

首次冻结运行在新模块完成后因磁盘写满退出 120。保留该完整模块，未采用中断模块
的部分结果；其余模块按完整收集列表分成每批 8 项，进程退出即释放 Cython 缓存。
最终 verifier 核对保留文件、每批原始结果、精确身份去重、源码和二进制哈希，全部
通过；主执行进程也已退出 0。早期夹具/参考模型错误和探索结果未混入最终计数。
报告：`mpi-evidence/training-runtime-index-frontend-final-sharded-20261003/README.md`。

仍待实现：运行时常量/optimizer bank gather、可变链接指向 refractory 限制目标的
条件写入，以及 shared/linked/summed/delay/随机积分器完整组合。目前该 refractory
边界也会明确拒绝只读突触访问。任意异步连续积分、外部回调、int64/unsigned、超出
int32 的整数字面量、位运算及新版 NVIDIA 实机验收仍未完成。下一接口见该证据目录
`NEXT.md`。本轮没有上传或云作业，GPU shader/CUDA wrapper 未改变，跨机器仍暂缓，
整体目标保持 active。


### 可变链接的不应期条件与隐式标志（2026-10-04，阶段验收完成）

可变链接指向 `(unless refractory)` 状态的只读访问和条件写入已接通。前端按 Brian
排序标识符解析共享条件名，条件索引取最后一个引用该条件的变量（包括只读变量）。
事件先读取逻辑局部值，再执行顺序赋值；最终写回可使用修改后的索引。因此条件为假
时仍可能把保持的旧逻辑值写到新的物理目标，不能直接丢弃整个写回。

显式修改同名 `not_refractory` 会更新本事件后续语句使用的布尔值；编译器仅对明确
声明的布尔 guard 开放这种修改，默认只读约束保留，未执行分支仍惰性求值。
Brian 代码生成才引入、原本不在 `Synapses.variables` 中的条件标志也绑定真实存储。
同时保留 Brian 自身的 scalar/vector 限制：常量初始化的临时标量在向量语句之后
再次赋值仍会拒绝；向量表达式初始化的合法形式通过实际 Cython 对照。

神经元 updater 使用自身 counter 计算活动标志，再通过 Brian 解析的索引写入。
阈值读取解析后的标志，但发放时清除自身物理标志，与 Cython 模板一致。动态阈值
使用 detached 布尔准备单元。初态采用实际 Brian 快照中的标志值，不从 counter
重建，以保留重复索引映射下未被写入的单元。

修复隐式标志前的 r2 完整基线为 **666 passed / 185 NVIDIA-only skipped / 0 failed**，
10 模块、851 个唯一身份。隐式标志修复后的 r3 独立运行四个完整模块，固定每批 8 项，
最终 **327 passed / 90 NVIDIA-only skipped / 0 failed**，417 个唯一身份。全部批次和
主执行进程退出 0；严格验证器核对完整测试身份、XML、213 个源码/依赖/检查脚本
及二进制哈希通过。旧版基线单独保留并验证，没有与新结果合并。原生程序 SHA-256 为
`69bd30927f5d993ab8d15a05955a5c55704e1e5c257dc837b5e8634dd1928c57`。
最终记录：`mpi-evidence/training-runtime-refractory-final-r3-20261004/README.md`。

测试覆盖实际 Cython Euler/RK2/RK4、独立固定噪声 full/TBPTT 递推、所有浮点初态和
参数的有限差分、pre/post 路径、重复目标、快照标志、延迟、掩码、carry/checkpoint
及非主进程越界回滚。CPU、实际 Metal 和本机 2/8 进程 MPI 参与验证。

两处预先存在的 Rust `0i8` 到 `0 as c_char` 修改已保留，重建原生程序后验证；旧源码
通过重建和哈希匹配保存。GPU shader/CUDA wrapper 未改。本轮没有上传或云作业，
NVCC/NVIDIA 实机验证仍未完成，跨机器按要求暂缓。

剩余本地实现重点为运行时 optimizer bank gather，以及可变链接与 shared/summed、
Heun/Milstein 和 event-driven 的更完整组合。状态依赖不应期、任意异步连续积分、
外部回调、int64/unsigned 和位运算也尚未覆盖；接口要求见最新证据目录 `NEXT.md`。
整体目标保持 active。


### 运行时 optimizer bank 读取（2026-10-04）

原生 `parameter_gather` / `integer_parameter_gather` 已实现，CPU 与 v5r6 Metal/CUDA
共享内核使用 int32 SSA 选择参数位置，惰性检查边界，反向按保存的动作上下文向真实
bank 槽累加梯度。索引 detach，重复读取累加，未选位置不产生梯度；int32 保留精确
位模式。旧 v5r5 库被拒绝。当前二进制 SHA-256：
`0a330bd4a81ec3c64ae319388cd0729ce0950ef24290d108427e3fdf4a933742`。

Brian 前端已接入直接索引、默认索引别名及指向选中 canonical storage 的显式引用，
覆盖积分、阈值、reset 和突触路径。参数通过独立的语句前索引读取，后续索引赋值不
改变已读取的参数；optimizer 更新之后的 carry/checkpoint 使用最新 bank 值。
本阶段最终冻结验收：**507 passed / 183 NVIDIA skipped / 0 failed**，12 个完整模块、
690 个唯一用例、221 个源码/依赖文件。包含实际 Cython、CPU/Metal、所有浮点有限
差分和本机 2/8 进程 MPI；主运行与同版本补充运行均已退出 0，严格核验通过。

主验收目录 `mpi-evidence/training-parameter-gather-final-20261004/` 的
`combined-verification.json` 汇总主运行和同版本常量链接完整模块。报告明确保留并排除探索失败：掩码测试初值
错误、参考模型遗漏阈值 scratch、Brian 不支持的多级显式索引生成代码问题。
多级显式索引曾因 Cython 未加载索引或读取顺序错误导致参考进程退出 138；具体修复
位置和证据见该目录 `NEXT.md`。当时该边界尚未验证；后续生成器修复和实际 Brian
验证结果见下一节，未用原生递推替代 Brian 参考执行。

实现契约见 `NATIVE_TRAINING_PARAMETER_GATHER.md`。更多 shared/summed、Heun/Milstein、
event-driven、delay/refractory 组合及先前列出的广泛 Brian 语义仍待实现/验收。
没有新云调用；NVCC/NVIDIA 及跨机器验证未完成。整体目标继续保持 active。


### Brian 多级索引依赖（2026-10-04）

修复 CodeObject 只收集一层索引的问题，并让 Cython、C++、NumPy 按依赖顺序读取。
条件写入的 flag 也收集完整索引链；NumPy 修改索引后的写回保留语句前读取快照。
普通读取的循环索引提前报错，SpikeGenerator 模板自管的自索引元数据保持可用。
此前 SIGBUS 的路径先经生成代码专项验证，再运行实际 Cython。最终生成器专项
23 项通过，多级参数前端 24 项通过、12 项 NVIDIA 跳过；包括实际 Metal、Cython
Euler/RK4 轨迹、固定噪声、full/TBPTT 和所有浮点初态/参数有限差分。

最终验收在 `mpi-evidence/training-nested-index-final-20261004/`：**133 passed /
54 NVIDIA-only skipped / 0 failed**，四个完整模块、187 个唯一用例、233 个冻结
文件，父进程退出 0。另有原有 Brian 代码生成/linked 子集在 NumPy、Cython 下分别
47 项通过，未与上述数字或旧版验收混合。既有直接链接和 implicit-refractory/MPI
回归也已通过。本阶段不修改原生二进制，不进行云调用。
更多组合与广泛 Brian 语义仍按该目录 NEXT.md 继续，整体目标保持 active。


### 状态依赖 refractory 与精确时钟（2026-10-04，v5r7）

已实现字符串 Boolean 和当前状态决定的 duration refractory。积分前更新活动标志，
每步重读时长；保留阈值、条件神经元写入、突触可塑性及 reset 的 Brian 顺序。
新增 int32 饱和 elapsed age、浮点 lastspike 与两个保留完整时间位模式的 int32 槽，
支持 warm snapshot、linked runtime 条件、optimizer/carry/checkpoint、MPI 失败回滚。
门、时间戳与时长的离散决策停止梯度，物理状态/参数继续 full/TBPTT。

真实 Metal 前向测试发现 `.0014-.0008 < .0006` 因 float32 舍入而改变门决策。
修复增加 time_word/elapsed_compare IR；设备以整数算法执行 binary64 相减/比较，
保留简单 t-lastspike 的六种比较及反向两侧，常量阈值保留 binary64。普通模型状态
与动态比较 RHS 仍使用后端 dtype。更广泛非线性时间算式没有自动获得 binary64。
动态 GPU ABI 升至 v5r7，拒绝旧库；原失败测试未弱化。

最终证据：`mpi-evidence/training-state-refractory-final-r2-20261004/README.md`，
871 passed /262 NVIDIA-only skipped /0 failed，1133身份、12模块、245文件哈希；
父进程43222 exit0。原父进程74472在旧 CUDA 文本断言失败，现已修正断言并完整
重跑 gather 模块；未改实现的9模块经清单和逐记录哈希验证复用。随机积分/RK
也全部完成，未汇总早期 pilot 或旧实现的验收数字。实际 CPU/Metal/本机2/8 MPI
覆盖新增与受影响语义，CUDA 仅源码转换检查，没有 NVCC/NVIDIA 或跨主机执行。

新增表达式 refractory 与 Heun/Milstein、summed、event-driven、mutable-delay 的
更多组合，以及前述 nested synaptic selector 等仍待继续验收；异步连续积分、
外部回调、int64/unsigned/bitwise 等广泛 Brian 语义仍未完成。完整剩余范围见
本轮 `NEXT.md`；整体目标保持 active，跨机器仍按用户要求暂缓。


### 神经元／突触乘性噪声与可塑性联合验收（2026-10-04）

前一阶段待验证的 Heun/Milstein、summed、event-driven、mutable-delay 与状态依赖
refractory 已在同一合法 Brian 模型中完成独立验收。神经元支持共享或交叉状态依赖
Heun 噪声及对角 Milstein 噪声；同一突触对象同时具有独立乘性 SDE、指数事件迹、
可塑性 w、非线性 summed 输入和事件更新的 pre/post 延迟。冷启动及已运行两步
网络转换、在途事件接续、优化器、检查点、full/TBPTT 和本地 MPI 均覆盖。

实际 Cython 只替换外部 randn 缓冲区补充数据，保留全部生成的积分/队列/路径代码；
抽样顺序、次数和独立噪声域均检查。独立向量/事件桶递推不读取原生 IR，遍历全部
浮点参数和初态有限差分，并要求神经元 sigma 和突触 rho 梯度非零。错误回滚测试
验证非 root 表达式域错误，不能用形状拒绝代替执行时失败。

最终记录：`mpi-evidence/training-stochastic-refractory-combined-final-20261004/README.md`。
84 passed /42 NVIDIA-only skipped /0 failed，126身份、2完整模块、251文件哈希；
父进程6647 exit0。前端和原生实现无需修改，先前245份冻结文件/二进制完全一致。
新增数目独立记录，不与上一阶段1133项或中间pilot相加。实际设备是Metal，本地
MPI为2/8进程；没有新云调用、NVCC/NVIDIA 或跨主机执行。

仍待继续：nested synaptic selector、非恒等 bank/subgroup 映射、可变路由表与
共享/summed/delay 的更广组合。当前联合模型为两条一对一输入边，不证明任意
共享状态/嵌套寻址。任意异步连续积分、外部回调、int64/unsigned/位运算和新版
NVIDIA 实测仍未全部完成；跨机器暂缓，整体目标保持 active。下一接口见本轮 NEXT。


### 多级突触寻址、共享路由与可变延迟（2026-10-04）

已修复真实前端缺陷：解析参数 gather 和间接写入根索引可能在延迟触发门后追加
上下文，导致 `delay gate mismatch`。现将私有门放在全部模型上下文之后，并保留
64 槽预算和原生严格布局检查。原生 v5r7 二进制不变；此前251份冻结文件中仅
`training_brian_dynamic.py` 这一份实现改变。原始失败和修复前源码完整保留。

新增模型覆盖 pick → 可变共享 route → 非恒等常量映射 → 规范 gain 参数库，
同时修改索引和别名目标、四条非平凡排序连接、summed、神经元／突触乘性噪声、
pre/post 可塑性、可变 pre 延迟、冷／暖快照与跨块接续。实际 Cython 以共同噪声
对照前向，独立事件桶与模型公式对全部浮点参数／初态做 full/TBPTT 有限差分；
整数路由梯度为零。优化器、双样本、检查点、本机2/8 MPI和执行时越界回滚也覆盖。

最终验收记录在 `mpi-evidence/training-nested-synapse-final-20261004/README.md`，
458 passed /184 NVIDIA-only skipped /0 failed，642身份、8完整模块、257文件，
父进程50785 exit0；不累加中间 pilot。实现范围详见
`NATIVE_TRAINING_NESTED_SYNAPSE.md`。当前显式 Variables 引用不代表高层 linked_var
限制已解除，也不代表实际 Subgroup 端点已支持；后者是下一项具体开发工作。
任意异步连续积分、外部回调、int64/unsigned/位运算与新版 NVIDIA 验证仍有缺口，
跨机器继续暂缓；整体目标保持 active。


### 实际 Subgroup 突触端点（2026-10-04）

已接入选中父群体的连续 Subgroup 端点：父群体物理 pre/post 地址、MPI owner、
触发脉冲和状态保持一致，i/j/N 则保留子群坐标。summed 仅清空相应视图，空边和
双方向均覆盖；不重叠子群写入可组合，重叠与未选中父群体仍拒绝。普通静态投影
也使用相同父群体映射。只在突触中被读取的显式选中神经元常量可以进入优化器。

真实 Cython 对照发现模板循环 i 与子群相对 i 重名。队列局部变量全部改为保留
名称，保留原始编译错误与修复前模板。此项修复不改变原生二进制或 GPU ABI。
独立参考还明确处理了 Subgroup summed 的 order 0 时序，不能假定它总早于积分。

三个新增模块验证 source／target／双侧子群、共享系数、神经元／突触乘性噪声、
pre/post 可塑性、双向可变延迟、不应期、暖快照、检查点、full/TBPTT 和本机 MPI。
正式结果见 `mpi-evidence/training-subgroup-final-20261004/README.md`：332 passed／
108 NVIDIA-only skipped／0 failed，440身份、8完整模块、270冻结文件，父进程3318
exit0；原有 Brian Cython 单独10 passed／0 skipped，父进程52060 exit0。两个终态
均已观察并通过验证，中间 pilot 不计入最终验收。

实现范围见 `NATIVE_TRAINING_SUBGROUP.md`。下一项是标准 run_regularly 更新动作；
任意异步连续积分、外部回调、int64/unsigned/位运算及新版 NVIDIA 实测仍有缺口。
没有云调用，跨机器继续暂缓，整体目标保持 active。

### 标准同步 run_regularly（2026-10-04）

已把选中神经元、连续子群及 Synapses 的标准定时代码接入原生动作图，按实际
slot/order/name 执行。Brian 自身的语句分析器区分标量／向量段：共享数组先以独立
逻辑局部值读取，标量代码只执行一次，排序写回后向量代码仍复用局部值。这样能
正确处理 shared 别名、临时量和子表达式重算；运行时 linked 索引与条件写入保持
原有读写时机。只在 runner 中使用的可训练 shared/array 神经元参数保留 bank。

定时修改的突触参数成为可携带状态，可训练初值和 Adam 常量更新不覆盖 carry。
逐边定时动作遵守结构 mask，共享标量更新仍按时钟执行；零连接也能更新共享值。
显式 randn 调用获得独立调用点流，标量和向量不复用抽样地址；t/dt、TimedArray、
暖快照、检查点、full/TBPTT 与本机 MPI 共用现有原生 v5r7，不修改二进制或 GPU ABI。

实现与边界见 `NATIVE_TRAINING_REGULAR.md`；最终证据入口为
`mpi-evidence/training-regular-final-20261004/verification.json`。
两个新增完整模块已通过 57 项，24 项 NVIDIA 硬件用例跳过；完整回归结果以该证据
文件为准，不累加 pilot 或旧阶段计数。异步时钟、uniform rand、int64/unsigned/
位运算、任意 Python 回调和 Synapses-to-Synapses 端点仍未全部实现，整体目标保持
active。NVIDIA 新版实机仍待验收，跨机器继续暂缓；本阶段没有云调用。

本阶段完整回归终态：232 passed / 56 NVIDIA-only skipped / 0 failed，288 身份、
6 完整模块、278 冻结文件；session 74065 exit 0 已观察，verify.py 通过，无在途测试。

### 原生均匀流与 rand 定时代码（2026-10-04）

原生 v4/v5 新增 uniform_noise，采用独立 domain 的 53 位计数均匀样本；GPU 采用
原有噪声表 ABI，但 float32 舍入时保持 [0,1) 上端开区间。NormalNoise 旧序列不变。
原生更新／reset 支持显式 UniformNoise 绑定，同步 run_regularly 支持 rand/randn
混合调用点；共享标量与向量、临时量重用和不同调用点保持各自的抽样身份。

真实 Cython 对照发现 masked 突触向量临时量遗漏 restart 描述，现已补全其多 owner
及 initial 重建规则，覆盖全部剪枝、regrowth、检查点与 Adam 后的随机状态续跑。
独立固定噪声 full/TBPTT 差分覆盖全部权重及初态；新模块 36 passed／17 NVIDIA-only
skipped，包含实际 Metal、CPU/Metal 本机 2/8 MPI 梯度及恢复、2 万余原生样本逐地址
与分布检查。另有 2 项 Rust 随机数和 float32 边界测试通过。

完整最终结果以 `mpi-evidence/training-uniform-final-20261004/verification.json` 为准，
新二进制保存在该目录，旧 runner 未覆盖。实现契约见 `NATIVE_TRAINING_UNIFORM.md`。
一般 Brian equation/threshold/reset/pre/post 中的任意显式随机调用仍有前端缺口，
同一 tick 的同一边多次投递尤其需要独立且可恢复的事件抽样身份。异步执行、外部
回调、int64/unsigned/位运算及 Synapses-to-Synapses 端点仍待补齐；NVIDIA 新版
实机尚未验收，跨机器继续暂缓，整体目标保持 active。

本阶段完整验收终态：207 passed / 95 NVIDIA-only skipped / 0 failed，302 身份、
5 完整模块、283 冻结文件；session 82114 exit 0 已观察，最终 verify.py 通过。
2 项 Rust 单元测试另计，旧二进制保留，临时 Cargo 缓存已清理，无在途作业。


### 随机事件身份与显式突触调用（2026-10-04）

标准 pre/post 路径已支持 rand/randn，先物化 Brian 子表达式再分配调用点，保持
局部随机临时量重用。新事件以发射 tick 而非到达 tick 抽样，队列压缩保留原始
发射延迟；转换前 pending 条目获得稳定 ID，避免同 tick 同边多次到达复用样本。
Rust CPU/GPU 主机层共用身份，GPU 内核 ABI 仍为 v5r7。连续突触 Euler 和 summed
表达式的显式随机调用也已通过独立验算与实际 Brian Cython 对照。

新增两模块正式验收39 passed／16 NVIDIA-only skipped，包括冷／暖队列、可变
延迟与重复到达、独立 full/TBPTT 全部权重／浮点初态差分、批次与极值seed、
结构掩码迁移、检查点、实际Metal和本机2/8MPI。原生对错误发射延迟、重复导入
身份和多重live投递有拒绝检查。3项Rust随机数测试另计。

完整最终回归311 passed／134 NVIDIA-only skipped／0 failed，445身份、11完整
模块、289冻结文件；父进程39928 exit0已观察，verify.py通过。证据入口：
`mpi-evidence/training-event-noise-final-20261004/verification.json`。本版本机runner
保存在该目录，旧二进制保留。本轮Cargo缓存已清理，无在途测试或云调用。

实现边界见`NATIVE_TRAINING_EVENT_NOISE.md`；延迟仍在调用边界重新锁定，TBPTT
边界不重设路由。一般神经元 equation/threshold/reset 中任意显式随机调用仍待
补齐，异步连续积分、外部回调、int64/unsigned/位运算和Synapses-to-Synapses端点
仍有缺口；新版NVIDIA实机未验收，跨机器继续暂缓，整体目标保持active。
下一具体入口已写入本轮验收目录的NEXT.md。


### 神经元显式随机调用与 typed reset（2026-10-04）

微分方程、threshold、reset 已接入标准 rand/randn 调用点流；命名 xi 流保持原位。
Brian 语句分析器先物化 reset 临时量／子表达式，再分配调用点，保留重用语义。
原始随机方程在 SymPy 替换之前校验，即使替换缓存已填充，也拒绝 Brian 不允许
的重复 stateful 调用和缺少 constant-over-dt 标记的随机子表达式。

v4 支持精确 int32／Boolean 局部表达式，物理状态仍为浮点，GPU ABI 升至 v4r7；
v5 沿用 v5r7。GPU 复用现有动态表达式求值器，保留 lazy 分支、整数载荷和 VJP。
本轮没有云调用；跨主机／多物理 GPU 继续暂缓。

新增完整模块 51 passed／23 NVIDIA-only skipped，覆盖实际 Cython 固定样本对照、
Euler／混合 xi／RK2／RK4、冷暖快照、refractory、独立 full/TBPTT 差分、批次、
检查点、实际 Metal 和本机 CPU/Metal 2/8 MPI。另有 4 项独立 v4 边界检查通过。

正式完整回归 575 passed／164 NVIDIA-only skipped／0 failed，739 身份、12 完整
模块、302 冻结文件；父进程 session 24468 exit 0 已观察，verify.py 通过。证据入口：
`mpi-evidence/training-neuron-noise-final-r4-20261004/verification.json`，新 runner 在
同目录，旧二进制保留。早期磁盘不足和编译选项覆盖造成的失败记录单独保留，
不计入正式结果；已恢复 Python 默认的 -fno-strict-overflow，仅追加 -g0。

实现边界见 NATIVE_TRAINING_NEURON_NOISE.md。下一项是 constant-over-dt 的真实
SubexpressionUpdater 调度与缓存状态，之后是直接随机 refractory 表达式。异步
连续积分、外部 Python 回调、int64/unsigned/位运算、Synapses-to-Synapses 端点
及新版 NVIDIA 实机仍有缺口。整体目标保持 active，下一具体入口见本轮 NEXT.md。


### 每步随机缓存与直接随机 refractory（2026-10-04）

标准 SubexpressionUpdater 已接入，神经元和突触 constant-over-dt 表达式保存为
可携带、可微的 typed 状态。共享/向量分段和真实调度、依赖链、Boolean、时间项、
RK 复用、延迟投递时刻、冷暖快照、full/TBPTT 及检查点均有独立对照。外部输入
群体的标准内部缓存保持在训练图之外，修复新 updater 识别引入的兼容性回归。

直接随机 refractory 已绑定独立 rand/randn 流，保留命名 xi 地址、布尔锁存短路、
每步时长求值与精确时钟；控制梯度保持硬门控。实际 Cython 样本注入覆盖两种条件、
Euler/RK2/RK4；独立补充检查覆盖 xi 混合的 Euler/Heun/Milstein 及 end-slot 缓存。

正式结果 344 passed／132 NVIDIA-only skipped／0 failed，476 身份、10 完整模块、308 冻结文件；
新增三个模块 89 passed／42 NVIDIA-only skipped。另有 16 项独立检查通过，不并入
pytest 计数。实际执行 CPU、Metal、本机 CPU/Metal 2/8 MPI；父进程 session 10248
exit 0 已观察，verify_completion.py 通过。证据入口：
`mpi-evidence/training-cached-refractory-final-20261004/README.md`。

复用原生 v4r7/v5r7 runner（46be8add60b0b5671d34819dff83a40347b774a7683edca4b7d04074190b5a92）；
本轮所有测试进程已结束，无云调用。早期磁盘失败和 pilot 未累计到正式结果。
异步连续积分、外部 Python 回调、int64/unsigned/位运算、Synapses-to-Synapses 端点
及新版 NVIDIA 实测仍有缺口，跨机器继续暂缓。整体目标保持 active；下一入口见
本轮 NEXT.md，不再把已经接入的随机方程、时间依赖和动态突触笼统称为“未实现”。


### 原生时钟访问核心与同刻边界（2026-10-04）

新增 clock.rs 共享访问游标，区分完整时钟访问与主步采样，保存 active 集合、
pending ticks/时间、执行计数和可恢复状态。已有 samples/boundary 已接入。
前端保存 Brian clock set 的实际同刻优先级；修复不同 dt 的非传递 epsilon 合并
可能改变第三时钟执行位置的问题。完整区间检查还保留了 Brian 对 active 时钟
越过自身结束步的拒绝，以及暖快照结构边界的舍入规则。

正式回归 230 passed／43 NVIDIA-only skipped／0 failed，273 身份、5 完整模块、313 冻结文件；新模块
75 passed／6 NVIDIA-only skipped。另有 10 项 Rust 测试和 8 项独立 CPU/Metal
2/8 MPI 优先级与检查点检查通过。父进程 session 51939 exit 0 已观察，
verify_completion.py 通过。证据：`mpi-evidence/training-clock-itinerary-final-20261004/README.md`。

新 runner SHA256：7c4370b6bfc8c0524f0f40a20ce5b84ce46ab8b6021551021c9dfd189732f3f9。
设备 kernel ABI 不变，仍为 v4r7/v5r7。本輪构建缓存已清理，验证进程全部结束，
无云调用；之前的源码快照、runner 和失败 pilot 记录保留。

完整异步连续积分仍未完成：动作尚需绑定 clock，并把实际访问序列接入 CPU/GPU
前向、反向 tape、脉冲/延迟队列和完整训练 checkpoint。新的执行计数尚未用于
训练随机样本地址。下一接口与验收计划见本轮 NEXT.md。其余 Python 回调自动
求导、扩展整数/位运算、Synapses-to-Synapses 端点和新版 NVIDIA 实测仍有缺口。
跨机器继续暂缓，整体目标保持 active。


### 多时钟训练 checkpoint 与边界恢复（2026-10-04）

clock_state.rs 已把完整游标接入 Request/Output、carry、手动续接和检查点。
保存 next_tick/start/initial_calls/calls/ticks/visits；恢复前由原生核心有界
重放验证。新 run 重新对齐 ticks、保留执行次数，evaluate/gradients 不提交游标。
CPU、Metal、MPI 共用已恢复的主机时钟表。修改延迟、时间输入和结构掩码时
原样保留 cursor；结构重生使用实际 pending clock 时间。新的动态 checkpoint
不允许缺失游标。主机执行器须更新，GPU kernel ABI 仍为 v4r7/v5r7。

正式回归 339 passed／68 NVIDIA-only skipped／0 failed，407 身份、6 完整模块、
315 冻结文件；新增模块 39 passed／12 NVIDIA-only skipped，另有 13 项 Rust
clock/checkpoint/noise 测试通过。实际覆盖 CPU、Metal、本机 CPU/Metal 2/8 MPI。
父进程 session 39376 exit 0 已观察，verify_completion.py 通过。证据入口：
`mpi-evidence/training-clock-checkpoint-final-20261004/README.md`。

新 runner SHA256：83e9a86675aa91c6cc9a7119a841932e3f451ce18883c8b13f29ca28f8c0dea9。
构建 target 缓存已清理，原生执行器、源码快照和 pilot 记录保留。所有验证进程
已正常结束，本轮无云调用。

完整异步动作、visit/frame 前向与反向 tape、快慢时钟脉冲缓冲/队列，以及动作
随机抽样 calls 尚未接入；不能把本轮 checkpoint 完成视为完整异步训练。
其余外部 Python 回调自动微分、扩展整数/位运算、Synapses-to-Synapses 端点和
新版 NVIDIA 实测仍有缺口。跨机器继续暂缓，总体目标保持 active；下一入口
见本轮 NEXT.md。


### 异步连续突触、定时动作与实际访问 VJP（2026-10-04）

连续突触、标准 run_regularly 和 SubexpressionUpdater 已按实际 CodeRunner
clock 调度，支持快于、慢于和非整数倍于主时钟的步长。新增 itinerary tape
在分配前计数并检查预算，保存各 clock 的时间/calls/active 与主输入 frame。
CPU 和设备前向、反向、随机样本地址、loss 归一化和 TBPTT 均接入实际访问。
t_pre/t_post 读取端点时钟，定时更新的 t/dt 保留 owner 绑定。

GPU 动态 ABI 升为 v5r8；Metal/CUDA 主机跳过不活动动作，MPI phase 8 独立
执行 TBPTT 截断，避免依赖 action 0 的活动状态。随机 Euler 与 RK2/RK4、缓存、
rand/randn、冷暖快照、独立有限差分、batch、分段/carry/checkpoint 均有验证。

正式回归 283 passed／136 NVIDIA-only skipped／0 failed，419 个身份、7 个完整
模块、317 个冻结文件；新增模块 58 passed／27 NVIDIA-only skipped，另有
16 项 Rust 测试通过。CPU、实际 Metal、本机 CPU/Metal 2/8 MPI 已实测。父进程
session 12762 exit 0 已观察，verify_completion.py 通过。证据入口：
`mpi-evidence/training-async-actions-final-20261004/README.md`。

执行器 SHA256：64a317354267a109063bdcf515e9cf55bdafedcf937e5ccd882005a16d55c43f。
CUDA 翻译后六个内核通过 stub header 的主机 C++ 语法检查，不代表 NVCC 或
NVIDIA 设备验收。当前执行器与源快照保留，pilot 不累计，无云调用或上传。

仍需多时钟神经元/阈值/重置、各源时钟的 spike buffer 生命周期、异步 pathway
延迟队列及跨访问事件伴随。当前验证器明确拒绝非主时钟的 spike threshold、
直接 spike trigger 和 event-noise pathway。外部 Python 回调 AD、扩展整数/
位运算、Synapses-to-Synapses 端点及新版 NVIDIA 实测也未完成。跨机器继续
暂缓，总体目标保持 active；下一入口见本轮 NEXT.md。


### 持久脉冲缓冲与逐访问发放（2026-10-04）

新增原生 dynamic.spike_buffers。各时钟阈值覆盖独立、可微的 binary state；
消费者显式使用 state trigger，允许按指定调度读取最近的结果。反向在阈值覆盖
处合并并清除缓冲伴随，包括被 refractory 活动条件禁止发放的情况。TBPTT 在
主帧边界切断缓冲伴随，carry/checkpoint 保留其前向值。新增 event_visits 保存
逐访问发放和实际时钟；原 spikes 在缓冲模式下按主帧累计发放次数，可大于 1。
每次实际发放都参与 loss，归一化仍使用主输入帧数。margin 必须在同一时钟准备。

CPU、Metal、CUDA 源码及 MPI 已同步接入，动态设备 ABI 升为 v5r9。正式回归
294 passed／129 NVIDIA-only skipped／0 failed，423 个身份、6 个完整模块、
318 个冻结文件；新增模块 39 passed／17 NVIDIA-only skipped，另有 17 项 Rust
测试通过。覆盖真实 Brian Cython 冷暖事件时间、独立 full/TBPTT 全权重和初态
差分、跨访问缓冲、禁用阈值、每帧多次发放、checkpoint，以及实际 Metal 和
本机 CPU/Metal 2/8 MPI。父进程 session 52740 exit 0 已观察，证据核对通过。

证据：mpi-evidence/training-spike-buffers-final-20261004/README.md。
执行器 SHA256：f2838fcc2f0a8491db0014b0084655a2b163340bc1d66f52f760a7fcdc323e9e。
CUDA 翻译后内核通过主机 C++ 语法检查，尚不构成 NVCC/NVIDIA 设备验收。
本轮无云调用、无上传；pilot 和历史结果不累计。

本轮完成多时钟事件的原生执行基础，前端的群体同 dt 限制尚未移除。下一步须
同时接入 group dt 下的积分/随机扩散/refractory、暖 spikespace 采集、路径自身
时钟的延迟量化与重建、结构迁移和事件身份，详见本轮 NEXT.md。外部 Python
回调 AD、扩展整数/位运算、Synapses-to-Synapses 端点和新版 NVIDIA 实测仍未
完成。跨机器继续暂缓，总体目标保持 active，不将原生基础阶段视作完整实现。


### 多时钟神经元前端与路径延迟自动转换（2026-10-04）

动态前端已移除群体共同 dt 限制；积分、随机扩散、固定/状态 refractory 使用各
群体自己的 dt。暖快照按各自 pending 时间验证并采集 spikespace；内部触发
连接持久缓冲。根据真实 Brian Cython after_run 行为，前端运行结束后清空返回
状态的脉冲缓冲，保留事件输出与梯度。修复共享 reads/writes 列表被触发操作数
原地追加而破坏 refractory 动作布局的问题。

Path.clock 贯穿延迟量化、队列移位、动态 routing、pending 保留、update_delays、
结构迁移和 checkpoint。校验器要求路径、动作与源阈值时钟一致。设备 ABI
仍为 v5r9；新增主机字段必须使用本轮配套执行器。

正式回归 530 passed／199 NVIDIA-only skipped／0 failed，729 个身份、九个完整
模块、319 个冻结文件；新增模块 130 passed／63 NVIDIA-only skipped，另有
17 项 Rust 测试通过。真实 Brian Cython 冷暖状态/发放、独立固定噪声 full/TBPTT
全权重与初态差分、Euler/RK2/RK4/Heun/Milstein、事件随机与动态突触、外部输入、
固定/状态 refractory、batch、结构迁移、新进程 checkpoint，以及实际 Metal 和
本机 CPU/Metal 2/8 MPI 均已覆盖。父进程 session 78072 exit 0 已观察，完整证据
核对通过；pilot 和旧结果不累计。

证据：mpi-evidence/training-multiclock-frontend-final-20261004/README.md。
执行器 SHA256：be260dfbbf16a3cd8ff896d4ad362d1684a518eb833deffc110498968289d539。
CUDA 仅完成翻译后主机 C++ 语法检查，未完成 NVCC/NVIDIA 实机验证。本轮无云
调用或上传，跨机器继续暂缓。

仍需近邻时钟与第三时钟组合、暖源 dt 变更的队列重采样组合、多时钟连续突触与
summed/linked/间接索引/随机缓存组合；自定义对象/函数与 Python 回调 AD、扩展
整数/位运算、Synapses-to-Synapses 端点仍未完整覆盖。详情见本轮 NEXT.md。
总体目标保持 active；本阶段通过不等于任意 Brian 语义全部实现。

### 暖时钟变更与多时钟组合（2026-10-04）

已修复合法 deferred dt 变更被旧 pending 数组拒绝的问题。动态前端只读调用
Brian check_dt 并计算下一次运行时间，应用到输入 origin 和固定/状态 refractory
初态；时钟数组、旧 dt 标记与 queue 均不修改。非法步长和陈旧普通快照仍拒绝。

两个新增完整模块已通过 89 项，44 项 NVIDIA-only 跳过，无失败。包含较快/较慢
时钟转换、待发送队列重采样、随机路径全权重和初态 full/TBPTT 差分、近邻与
第三时钟、连续 SDE/summed/正态缓存、共享/linked/间接索引、CPU/实际 Metal 与
本机 2/8 MPI、batch/carry/checkpoint。九个模块的正式完整回归为 615 passed／
198 NVIDIA-only skipped／0 failed，813 个身份、321 个冻结文件；父进程 session
11439 exit 0 已观察，verify_completion.py 已核对源码、执行器、完整身份及终态。

证据入口：mpi-evidence/training-multiclock-transitions-final-20261004/README.md，
续接说明：该目录 CONTINUE.md。原生源未改、执行器复用前轮冻结二进制；此前的
Rust 17 项及 CUDA 主机语法证据明确继承，不冒充本轮新运行。无云调用/上传。

审计又确认 13 个标准数学函数缺少编译支持，包含 expm1/log1p/exprel、反三角、
abs/sign 等；具体拒绝表达式见 standard-function-audit.json。poisson 随机调用、
自定义函数/回调 AD、扩展整数/位运算和 Synapses-to-Synapses 端点等尚未完整
覆盖，新版 NVIDIA 实机仍待验证，跨机器继续暂缓。目标保持 active。


### 标准数学函数基础（2026-10-04，组合验收待续）

完成上一阶段九模块正式回归及终态核对后，整合独立暂存目录中的数学实现：
tan/cosh/sinh/log10、expm1/log1p/exprel、反三角、ceil/abs/sign 共 13 个此前
缺失函数，以及 floor 标量路径。原生 SSA 追加 math opcode，CPU 和设备都
直接计算值与 VJP。Metal 使用稳定的 expm1 Taylor 与补偿 log1p；exprel 的
可去奇点值和导数显式定义，导数的小值/大值分支避免消差和中间溢出。

新增 b2_train_math_v1 GPU 能力标识，布局仍 v4r7/v5r9；新主机拒绝旧能力库。
整合后 CUDA 语法检查捕获并修复辅助函数名含数字时漏加 __device__ 的问题。

完整暂存数学模块 112 passed／53 NVIDIA-only skipped／0 failed，165 个身份，
CPU 与实际 Metal；三种执行路径、90 位参考值、解析 loss VJP、定义域、惰性
求值、整型控制和旧库拒绝均已覆盖。父进程 session 5748 exit 0 已观察，冻结
暂存源码、执行器及全部身份核对通过，核对进程 session 27049 exit 0 已观察。
整合后的 19 项 Rust 训练测试通过；CUDA 六个翻译后内核通过主机 C++ 语法检查。
这些结果不代表 NVIDIA 实机或新增函数在完整 Brian SDE/MPI 组合中的验收。

证据：mpi-evidence/training-standard-math-20261004/README.md。
当前独立执行器 SHA256：4a3914cf55916b4b9655f51210be8d09b203279390dd324f88b33dc1bbe954c4。
所有本轮测试/构建均已终止。下一步为真实 Brian Cython、Heun/Milstein 幅度、
动态突触连续/事件路径、多步 full/TBPTT 与 2/8 MPI 的完整组合验证。poisson、
自定义函数/回调 AD、扩展整数/位运算、Synapses-to-Synapses 等仍未完整实现。
NVIDIA 实机待验收，跨机器继续暂缓，无云调用/上传。目标保持 active。


### 标准数学与随机动态突触联合验收（2026-10-04）

新增真实 Brian Cython 联合模型，覆盖 14 种函数在随机 drift/diffusion、阈值、
reset、连续突触与延迟 pre/post 可塑性中的 Heun/Milstein 行为。独立标量公式和
显式 FIFO 对照冷暖状态与发放，固定噪声 full/TBPTT 差分遍历全部参数及非 detached
初态；本机 CPU/实际 Metal 2/8 MPI、双样本、carry/checkpoint 也已通过。

正式十模块回归 649 passed／301 NVIDIA-only skipped／0 failed，950 个身份、585 个冻结文件；
修复后父进程 session 10304 exit 0 已观察，verify_completion.py
已核对全部 collection/JUnit、源码/执行器、分片及父进程终态。初轮 session 13782
因旧 dispatch 次数断言 exit 1；只修复对应测试文件，三个模块续跑，七个无依赖
完整模块按哈希保留，原始失败片段不计入最终结果。新增联合模块 240 项通过，
120 项 NVIDIA-only 跳过。19 Rust 测试、六内核 CUDA 主机语法证据明确继承自前轮
同源码/同二进制数学实现，不冒充新运行。默认 target/release 未替换。

证据：mpi-evidence/training-standard-math-final-r2-20261004/README.md。
执行器 SHA256：4a3914cf55916b4b9655f51210be8d09b203279390dd324f88b33dc1bbe954c4。

新增入口审计确认 poisson 和 timestep(t, dt) 在六种代码位置均仍未转换，exprel
六个阳性对照均接受。timestep 返回 int64，不能简单替换成既有 int32 cast。
后续还包括 Poisson rate 的明确梯度规则、自定义函数/回调 AD、扩展整型/位运算、
Synapses-to-Synapses 和非标准语义；详见本轮 NEXT.md。NVIDIA 实机尚未完成，
跨机器继续暂缓。本轮无云调用/上传，总体目标保持 active。

### Poisson 数值核心与 CPU 训练接入（2026-10-04）

运行时 Poisson int32 SSA、独立 counter 内部 draw、跨输出去重的 likelihood-score
反传已接入 CPU v5；rate 可以依赖参数/当前状态，整数样本路径保持 detached。
反向为实际执行的每个调用注入停止梯度的单样本 loss/batch × (K-rate)/rate，
随后反传 rate 表达式；旧 hard-threshold surrogate 契约保持明确，不能把两者组合
宣称为任意硬脉冲模型的无偏导数。被屏蔽动作、未执行的惰性分支不增加 score。
整数输出和零普通伴随仍会计分，indirect 路径使用记录的实际地址。

Brian 的 neuron ODE、threshold、reset、cached、regular、连续突触和 pre/post
调用已转换。真实 Cython regular 神经元/突触对照逐 draw 验证，并覆盖冷暖启动。
独立 fixed-count likelihood + local-surrogate 差分遍历参数和初态，full/TBPTT、
双/多样本、2/8 MPI、恢复、嵌套抽样和事件身份通过。单独的可解析期望梯度测试
区分 score 与固定随机数样本路径导数。Metal 数值核心另有实际设备分布、score、
极小 rate、计数精度测试；修复了真实 FTZ 将正 subnormal 当成零而不消耗 draw 的问题。

正式 11 模块最终合并 **480 passed／139 NVIDIA-only skipped／0 failed**，619 个
身份、714 个源码/依赖文件；另有当前 Rust 训练 23 项通过。原首轮 session 24764
exit 1，仅旧库 ABI 测试夹具缺少数学 capability；保留首轮哈希与原文件，补 capability
后完整重跑 parameter_gather 模块（45 passed／21 skipped，session 49934 exit 0），
其余十模块未改源码/依赖直接保留。verify_completion.py exit 0（8cd948）核对全部
身份、替换范围、二进制与终态。默认 target/release 没有替换。

证据：mpi-evidence/training-poisson-ssa-20261004/README.md。
契约：NATIVE_TRAINING_POISSON.md。独立执行器 SHA256：
47a1e77e28465890e4dd44bcf24338eaa05bb0d99459bf7af56ebe80285fd739。

尚未完成：Poisson GPU expression/score 接入、连续零 rate 的边界梯度（当前明确拒绝）、
跨 action 重复 emission site 的缓存/延迟迁移扩展、静态 v3/v4 Poisson 和更广泛组合验收。
int64/timestep、扩展整数、自定义函数 AD、Synapses-to-Synapses 等仍待实现。当前只承诺
int32 计数，Brian 隐式 int64 regular 局部变量仍拒绝，不能静默缩窄整型语义。
NVIDIA 实机未验收，跨机器继续后置；本轮无云调用或上传。目标保持 active。

### Poisson GPU 训练接入与本地验收（2026-10-05）

Poisson 已接入 Metal 动态表达式和 reverse interpreter；opcode 53 保留 int32
payload，host 只传精确拆分的 counter key，rate 在设备上计算并抽样。单 GPU 和
owner-compute MPI 都使用单样本 loss/batch 的 likelihood score，去重实际执行的
stream，覆盖 detached 整数输出、间接地址、惰性分支及 full/TBPTT。新增 context
计入 staging/device budget，b2_train_poisson_v1 拒绝旧库/错误能力版本。
CUDA 共享体完成相应接入与 host C++ 语法检查，尚无 NVCC/NVIDIA 实机证据。

正式 14 模块 881 个身份：627 passed／254 NVIDIA-only skipped／0 failed；额外混合
Gaussian/uniform/Poisson 模块 12 个身份：8 passed／4 skipped。合计 **635 passed／
258 skipped／0 failed**，当前 Rust 23 项通过。另用四次 CPU/实际 Metal、串行/局部 MPI
检查计数位模式属于 float NaN 时的完整整数传递和似然梯度。真实 Metal 包括参数/初态
独立 VJP、full/TBPTT、2/8 MPI、恢复、极小正 rate、计数精度、旧库拒绝、预算与回滚。
Brian regular 冷暖 Cython 对照已扩展到 Metal，其余七种代码位置也完成编译与执行。

父进程 session 12857 exit 0（cbe7b1），715 个冻结源码/依赖哈希一致；补充模块
session 46170 exit 0，四次大计数 probe session 46352 exit 0。verify_completion.py
exit 0（78425a）核对全部身份、源码、程序、翻译源哈希、skip 参数和终态。
没有沿用上一轮测试计数，也没有修复后替换模块。默认 target/release 未替换。

证据：mpi-evidence/training-poisson-gpu-20261005/README.md。
独立执行器 SHA256：6b7d4f3f496e10bc4d63f5e1f9c59db955b8cf8ce48dbfda4a83103a3da22765。

下一步仍包括连续零 rate 的边界梯度（不能用零 score 代替）、跨 action common-draw /
延迟迁移组合、int64/timestep、扩展整数、纯自定义函数 AD、Synapses-to-Synapses 与
其它未覆盖语义。NVIDIA 实机未验收；跨主机按用户要求继续暂缓。本阶段无云作业或
上传，整体目标保持 active。

### 2026-10-05：CPU Poisson 零 rate 反事实轨迹与本地 MPI

原生 CPU 在实际执行的连续零 rate 抽样点重放完整单样本轨迹，将该点计数固定为 1，
按原 batch/事件/时钟 key、改变后的后续 rate 运行，使用归一化损失差注入 baseline
rate VJP。MPI 各 rank 进入相同重放顺序，保留 owner compute。重放只 evaluate，
baseline 状态才会提交；预分配前保守预留两倍执行内存，多个重放顺序执行。
GPU 零 rate 重放未实现，没有 CPU fallback。

新增 33 项覆盖硬轨迹解析损失、参数/初态、TBPTT、初态参数绑定、2/8 MPI、嵌套/惰性
抽样、间接目的地址、无效分支回滚、内存、事件触发、异步时钟、checkpoint、SGD 和
实际 Brian Cython 一计数轨迹；另两项独立用例覆盖同轨迹正 rate 与零 rate 的项叠加。

10 模块初次正式运行 594 身份、716 冻结源文件，session 69047 exit 1（be5d53）：
424 passed / 168 NVIDIA skipped / 2 failed，哈希一致。两处仅为成功 CPU 零 rate
计算更新 last_result 后，测试仍要求全 snapshot 不变。保留原测试/日志，只修正该
成功断言；整模块加两项混合用例重跑 98 身份，session 24398 exit 0（26cffc）：
67 passed / 31 skipped。9 个源码不变模块复用本轮初跑。最终 **428 passed /
168 NVIDIA-only skipped / 0 failed**，596 身份，另 23 Rust 测试通过。
verify_completion.py exit 0（b27c7a）核对身份、仅一个测试文件变更、最终 hash 和终态。

额外审查得到待修 CPU 反例：draw(scale-scale) 恒为 0，但保守依赖标记仍要求计数 1
重放；v=1/(1-k) 因此误报 domain error。其 forward 正常，参数梯度应全 0。
probe_zero_vjp.py 已复现并记录，未计为通过功能。下一轮先实现实际 baseline rate VJP
为零时的判定，再推进 GPU 零 rate、int64/timestep、迁移/缓存组合、自定义函数 AD 与
其它未支持 Brian 语义。前端普通 v:1（无微分 v）也确认仍被拒绝，记录在 NEXT.md。

当前证据：mpi-evidence/training-poisson-zero-cpu-20261005/README.md。
独立 runner SHA256：1a062a82e8f6a0ab44464eb2df1936f35aa095f5da3976ac93a09ce76f0b8ed2。
722 文件源码/过程归档逐项验 hash，SHA256：
189bdd0b8301fcceb9825ff0d5fa24f0ea3c3c16a4c9d0f0e25229ce0bfada12。
所有进程已观察到终态；无云端执行/上传；跨主机仍暂缓；整体目标保持 active。

### 2026-10-05：修复 CPU Poisson 零 rate 的实际 VJP 判定

上一阶段的 draw(scale-scale) / v=1/(1-k) 误报已修复。零 rate 的 unit VJP 先沿
实际访问分支计算可达导数，合并 canonical 参数 bank 与实际物理状态地址，应用
mask/detach，再用精确非零判定决定是否重放。不使用 epsilon，1e-300 非零路径仍
要求重放。优化器冻结不抹去被请求的梯度。unit probe 对全屏蔽/detached 的导数路径
进行裁剪，避免无关 sqrt(0) 奇异导数；真实 active 奇异导数继续报错。

新增 57 项覆盖取消/零因子/惰性/最小值分支、mapped/gather/neuron 参数别名、直接/
间接状态别名、mask/detach、2/8 本地 MPI，以及相同值不同地址、极小非零系数、
冻结 optimizer bank、剩余参数路径与奇异路径等负对照。重跑此前零 rate、Poisson
SSA、CPU/实际 Metal/MPI 和混合正/零 rate 模块，合计 **180 passed / 32 NVIDIA-only
skipped / 0 failed**，212 个完整测试身份，另 23 Rust 项通过。

正式 session 62308 exit 0（364abc），717 个源文件 hash 一致。控制脚本沿用了
仅索引 tests 目录的过滤器，因此 expected-tests.json 有 210 项；原始 collection.log
及 JUnit 包含全部 212 项（另两项 evidence 目录混合 rate 用例确实执行且通过）。
verify_completion.py exit 0（3f1bfb）核对并生成完整身份文件，没有隐去执行或替换模块。
旧误报的精确 probe session 86994 exit 0（80b646）现返回全零参数梯度，旧失败证据
保持原样。最终 release build 和 23 Rust 测试均已观察到 exit 0。

证据：mpi-evidence/training-poisson-zero-vjp-20261005/README.md。
Runner SHA256：741aa55fe3b427ac8b9ac7f751a5eaffa5b02de94a03ea2ae27c7115627b7cee。
721 文件归档逐项验 hash，SHA256：
f456da607539b6a97633900e54c72818271ccabb725450a6b12cd0ef273a396e。

仍未完成 GPU 零 rate 重放；普通/正 rate VJP 尚未共用上述 inactive-path 裁剪，应
继续审查更一般的 mask/detach 奇异路径。int64/timestep、common-draw/迁移组合、
自定义函数 AD、未支持 Brian 状态/对象/endpoint 语义保持原范围。CUDA/NVIDIA 未
实测，跨主机暂缓，本阶段无云作业/上传。所有进程已结束，整体目标保持 active。

### 2026-10-05：GPU Poisson 零 rate 实际 VJP 判定

共享设备解释器现用 unit rate VJP 合并 canonical 参数槽和物理状态地址，应用
mask/detach、实际访问分支与 inactive-path 裁剪。结果确实为零时无需重放；不使用
任意 epsilon。TimedParameter 非零时间行、mapped/gather/neuron 别名和直接/间接
状态别名均有实测。私有 probe 不写普通 gradient，混合正/零 rate 回归通过。
用 float bit 判零，保留 Metal 对极小正 rate 的既有处理。新增库 capability
b2_train_poisson_vjp_v1，动态 GPU 每 lane scratch 预算提高至 16 KiB，ABI 不变。

5 个完整模块共 **353 passed / 206 NVIDIA-only skipped / 0 failed**，559 个
完整身份，718 个源文件 hash 一致；另 23 Rust 项通过。实际 Metal 和本地 MPI
已执行，CUDA 仅完成生成代码的 host C++ syntax 检查。正式 session 98612 exit 0
（f508b5），Rust 74491 exit 0（8cbeeb），最终 build 16237 exit 0（d7e727），
CUDA syntax exit 0（70134e）。验收 verifier exit 0（6d0975）。全部进程已观察终态。

证据：mpi-evidence/training-poisson-gpu-vjp-20261005/README.md。
Runner SHA256：467fb8c6fb471bff9cf1e07289044abda4aa242a6637784cf9f788e3848b6133。
721 文件源码/过程归档逐项核 hash，SHA256：
22e464481255a3f0be3446ea8e5deb4c67dcebf771deda5e67dfad126aef4ce7。

零 rate 的非零 VJP 仍显式拒绝：下一步是完整 GPU counterfactual 轨迹重放，
具体设备系数上下文、MPI 顺序、预算与测试路线已写入本阶段 NEXT.md。普通正 rate
inactive 奇异路径、跨动作公共 draw/重复事件、int64/timestep、自定义函数 AD 与
未支持 Brian 状态/对象语义仍在原范围，不因本阶段通过而缩小目标。无云作业或上传，
跨主机继续暂缓，整体目标保持 active。

### 2026-10-05：实现 GPU Poisson 零 rate 的完整轨迹重放

Metal 动态执行器现先在设备收集需要非零边界梯度的 action/visit/sample/stream，
MPI 汇总 owner mask 后，各 rank 以相同顺序从原始状态重放完整 GPU batch。每次只
强制选定计数为 1，后续 live-rate 采样、事件、惰性调用和间接地址仍在设备执行。
原始 batch 随机键保留。备用单样本 GPU loss 存回专用槽，最终基线 GPU reverse
计算 loss 差/batch；缺少 ready bit 显式失败。只提交基线状态、optimizer 和 cursor。
新增 b2_train_poisson_boundary_v1 capability、header bit 512、97 槽随机上下文、
每动作一个 tape mask 和两个 force control。预算纳入 host/device context、mask
及 MPI scratch；所有 replay dispatch 都纳入结果计数。无 CPU 替代 GPU 轨迹。

6 个完整模块 **482 passed / 268 NVIDIA-only skipped / 0 failed**，750 个完整
身份，719 个源文件 hash 一致。另 23 Rust 测试通过。实际 Metal／2、8 本地 MPI
验证 full/TBPTT、batch、nested live rate、不同动作/owner、间接写入、后续事件、
异步 idle visit、carry/checkpoint、SGD、预算、非法备用轨迹、缺失系数及旧库拒绝。
正式 session 12371 exit 0（38ffb3），316.21 秒；Rust 54330 exit 0（63ff6d）；
release build 81451 exit 0（0bc74f）；CUDA host syntax exit 0（20974e）。
verifier exit 0（52355e），完整核对所有身份、source/procedure/runtime hash 与终态。

初次 pilot 的 6 个失败均为新增测试预期问题：event 用例累计电压而 oracle 预期
替换，以及成功训练后 next_noise_sequence 应为 20。原测试、日志、XML 和 exit 1
终态完整保留；修正预期后全新模块重跑通过，再纳入正式全模块回归。

另独立复现待修缺陷 draw(scale+sqrt(r))：scale=r=0，r detached。unit VJP 正确
识别 scale 的有限活动导数，但 CPU／实际 Metal 普通 reverse 仍遍历无关 sqrt
奇异导数并失败。probe session 51764 exit 0（72ac48）只表示成功记录缺陷，绝不
计为通过功能。下一步修复普通 VJP 的 inactive-path 裁剪。int64/timestep、公共
随机 draw／迁移组合、自定义函数 AD、未支持 Brian 状态/对象/endpoint、static
Poisson 等保持原范围。CUDA 无 NVIDIA 实测，跨主机暂缓，无云作业或上传。

证据：mpi-evidence/training-poisson-gpu-boundary-20261005/README.md。
Runner SHA256：d6a8fcbee9903c06a05f080a72cff4fd52e2aa32fcf657f85a7f42a92e3c8755。
724 文件归档逐项验 hash，SHA256：
8e17f8213cb2e0b5dbcd1f855f1e24b10d409b1f39d36d198e7bc9be8c59c704。
全部进程已观察终态；整体目标保持 active。

### 2026-10-05：CPU／动态 GPU 普通 VJP 活动路径裁剪

CPU 普通 VJP、Poisson 似然/边界 seed 和 unit rate probe 现共用 actual visited
branch 的可达性判定。动态 direct/indirect 路径传入真实物理 read 地址及 detach；
canonical 参数 mask 覆盖 mapped/gather/neuron/timed。共享 GPU helper 在普通及
probe 模式均裁剪无请求叶子的奇异导数路径。无 epsilon；活动奇异路径仍回滚；
forward domain 与 trainable=False 的梯度契约不变。CPU 固定 bool 数组取代 JSON
节点遍历，dynamic budget 新增 256 字节串行 scratch；GPU 保留 16 KiB/lane。
动态 GPU gradient/train 强制 b2_train_vjp_activity_v1 == 1，capability 范围为 v5。

原 draw(scale+sqrt(r)) / scale=r=0 / r detached 反例已修复：CPU 与实际 Metal
分别返回 -0.5253611511736794、-0.525361180305481。最终 shader 重测 probe session
36745 exit 0（4d44cb）。新增 260 项覆盖普通连续、正/零 rate、正态/均匀噪声幅度、
参数/state 屏蔽、间接读、选中分支与 active 奇异负对照。9 完整模块最终
**694 passed / 352 NVIDIA-only skipped / 0 failed**，1,046 身份，720 源文件 hash
一致，另 23 Rust 项通过。

初次新测试 pilot 有 109 构造失败，原测试/日志/XML 保留；修正函数名替换、变化
计数索引、对称标签和 scalar gradient key 后新模块全过，再加入 noise 用例并完整
纳入正式回归。第一次正式运行 session 3450 exit 1（dc3eee），680/352/14，原因
为我把活动图用于借用 dynamic evaluator 的旧 v4 vector math 入口，错误读取 v5
元数据。只改 training_dynamic.metal 区分 v5 活动入口与 legacy 兼容入口；完整
math module 重跑 112/53（98479 exit 0，84f93b），再重跑 9 个完整模块。

最终 session 65047 exit 0（5b9abc），431.21 秒。Rust 61584 exit 0（e63f8d）；
release 38893 exit 0（2b9ccf）；CUDA host syntax exit 0（411619）；verifier exit 0
（642221），核全部原/现身份、仅一个修复差异、源/过程/runtime hash、probes 与终态。
initial-formal/ 保存原始失败来源、归档、日志与 JUnit，不复写历史阶段。

独立范围审查仍确认旧 scalar/vector **GPU** 的 masked sqrt(0) 失败；CPU 两模式
已有限返回。static-gpu-activity-probe.json 的 exit 0 仅表示成功记录缺陷，绝不算
通过。gpu.rs 目前只在 host 应用 mask，下一步需设备 mask staging、独立 legacy
capability、正确 legacy 活动模式和预算。具体 route 及 metadata 占用已写 NEXT.md。
int64/timestep、公共 draw／迁移、自定义 AD、未支持 Brian 状态/对象/endpoint、
static Poisson 与 broader SDE 组合保持原范围。无 NVIDIA 实测，无云作业/上传，
跨主机仍暂缓；所有进程已观察终态，整体目标保持 active。

证据：mpi-evidence/training-vjp-activity-20261005/README.md。
Runner SHA256：4219f9a8838962ee1faa0fb7e9b8ac5e175fa0938927f497faf4aaf4657a2b63。
最终 726 文件归档逐项验 hash，SHA256：
988134e038a80f821886416fef7454e1c628fe637756646d04f187a62be597e6。
初次正式运行归档独立保留，SHA256：
009b67ada39e3a0a198f4dac4f0a74d2c06e2f60b44b1b3c11ec86c25a7987c5。

### 2026-10-05：旧 scalar/vector GPU 普通 VJP 活动路径补齐

旧 v3/v4 SSA GPU gradient/train 现要求独立 b2_train_static_vjp_activity_v1 == 1。
Host 在 m[9] 设置 bit 2，把 canonical masks 放在 constants/clock/noise staging
之后，设备使用 m[13]-m[5]；不占用已有 metadata word。Evaluation 与无 SSA
的 built-in reverse 保留原能力契约。GPU budget 纳入 16 KiB/lane scratch 和两份
mask。Scalar/vector 直接解释器裁剪无请求叶子的奇异导数；共享 helper 分 mode 1
（真正 v5 的 physical cells/detach/m20 masks）与 mode 2（legacy v4 local states/
mask suffix），数学及惰性 v4 程序正确使用后者。无 epsilon，活动奇异导数和非法
forward 仍失败回滚。

原 masked sqrt(0) scalar/vector GPU 反例 CPU／实际 Metal 均有限返回，gradient
分别 -0.15280073461891633 / -0.15280072391033173。Probe session 83186 exit 0
（55d944）。新增 150 身份涵盖 sqrt/pow/arccos、neuron canonical slices、reset、
clock、normal/uniform staging、full/TBPTT、detach、冻结 optimizer、checkpoint、
惰性分支、预算与旧库能力负对照；106 passed / 44 NVIDIA-only skipped。

17 完整模块最终 **1,038 passed / 486 NVIDIA-only skipped / 0 failed**，1,524
精确身份，721 source hashes 一致；另 23 Rust 项通过。最终 formal session 8143
exit 0（512c98），742.82 秒；Rust 23226 exit 0（2925d4）；release 55230 exit 0
（6265d3）；CUDA host syntax exit 0（e3bc88）。Verifier exit 0（9c4ad5），逐项核
原/现身份、source/runtime/procedure hash、四次 pilot、独立 probe、终态与两份归档。

首次新测试 pilot 有 116 / 58 / 8 / 4 个测试构造失败，原测试、日志、XML、exit 1
终态保留；修正 clock/time binding、无噪声模型的序列参数、类型描述和惰性编译入口
后完整新模块通过，再纳入正式回归。首次正式 session 55002 exit 1（3e3edd），
1,037/486/1，唯一失败为旧 metadata ABI 测试模拟库未提供早已要求的 math capability，
尚未抵达其要测的缺失执行符号。补齐 math ABI 1 的 stub，保留缺失 metadata entry；
同时仅改 Poisson descriptor docstring 的过时说明，核 executable AST 未变；再次
运行全部 17 模块。首次 formal 全部输入和 742 文件归档独立保留在 initial-formal/。

证据：mpi-evidence/training-static-vjp-activity-20261005/README.md。
Runner SHA256：ac05ae2c15a7167643d9c100afbfb27da0e0decbe86e190818410a26a3af6daa。
最终 756 文件归档全部逐项验 hash，SHA256：
4ac6f6628be6e2de6dcc211c68a2534bf6fd3647ee719d1a924eb09cfef12c56。
初次 formal 归档 SHA256：
2b499db01f953bcd634b09cc03721349fcdf8b762bd5b82e51531481843f78d4。

整体目标仍 active；共享 Poisson draw／cache 跨 action、延迟迁移／重复 pending
事件身份、完整 int64/timestep/unsigned/bitwise、自定义函数 AD、未支持 Brian
状态/对象/endpoint、static Poisson 与 broader SDE/migration 组合保持原范围。
下一实现路线及正/零 rate 只计一次与全别名回放要求写入本阶段 NEXT.md。无 NVIDIA
实测、无云作业或上传；跨主机按用户要求暂缓。所有本阶段进程已观察终态。

## 2026-10-05 CPU shared Poisson draw cache

继续原目标，未缩减范围。新增 CPU 显式 draw identity 缓存：首个实际访问保存 count
与 rate；后续 action 跳过 rate 子树，反向仅原上下文计分。MPI 只广播 owner 新记录；
零 rate 回放从输入缓存开始并强制同一 identity 的所有消费者。carry/store/restore
保留 b2-poisson-draw-state-v1；恢复原生校验身份、counter count、rate、batch、sequence
与工作量，提交前完成。缓存按 768 bytes/记录加 overhead、双回放及历史总量准入；
目前保留历史到预算上限，安全退休仍待集成。共享 GPU 缓存仍明确拒绝。

新增完整共享模块 68 项通过，独立正率固定样本/score FD、全别名零率弱差分、
full/TBPTT、batch、2/8 本地 MPI、覆写无效 rate、首次惰性访问、延迟/固定 pending、
新进程恢复、indirect、异步 idle、只读/新序列、预算/回滚/篡改/不兼容率负对照。
27 Rust training 单元通过，包括 scoped 回放出错恢复及 carried draw 无计分 owner。

首次 28 完整模块 session 12318 exit 1（86fd23），1098.93 秒，1484 passed /
538 NVIDIA-only skipped / 1 failed，2023 身份，723 source hash 保持一致。唯一
失败为旧迁移测试改 clock_tick 却保留 committed clock_state，正确的早期边界拒绝
先于预期 work-budget 错误。仅改该测试文件：snapshot 加 clock/Poisson 缓存；
参数化 saved-checkpoint 与 bare-prefix 两条回滚/重试路径。完整迁移模块 session
2917 exit 0（86f6a6），50 passed，32.02 秒。生产与其余 27 模块输入逐字节不变；
原始测试与替换字典保存，verifier 重构测试改动并验证确实只改这一文件。

最终合并覆盖 1486 passed / 538 NVIDIA-only skipped / 0 failed，2024 精确身份，
28 完整模块；不是第二次重跑全部 28 模块的声明。旧 pilot 的触发器漏设、错误率
测试修改错 operand、CUDA admission 提前构建硬件库等失败均保留原测试与日志。
Verifier exit 0（53fd06），归档 exit 0（cbe653）：783 文件逐项 hash 验证。
Runner SHA256：c068502d0f3d7cdaa82c3f712b5a28d5d48e254a8934756773feeede0eaf9134。
Archive SHA256：521d81a424431b975fa1900a9a3ee92c54d0520a01a2dccd10742b1bc425872f。
证据：mpi-evidence/training-poisson-shared-cpu-20261005/README.md。

整体目标仍 active。下一阶段为共享 GPU 缓存及 Brian generated delay/migration
集成（NEXT.md 已写设备 identity/计分/全别名回放、版本能力和预算要求）。完整
int64/unsigned/bitwise、自定义函数 AD、未支持对象/endpoint、static Poisson 与其他
SDE/migration 组合仍保留。无云作业或上传；无 NVIDIA runtime 验收，跨主机按用户
要求暂缓。所有本阶段进程已观察终态。

## 2026-10-05 GPU shared Poisson invocation cache

继续原目标，未缩减范围。共享 Poisson 地址已接入实际 Metal 和本地 MPI 设备缓存：
host 仅 intern 显式 site/instant 身份与 uint32 offset，不抽样或计算 rate VJP。
每个 batch lane 的 tape tail 保存 valid、int32 count bits、f32 rate 与首次 noise
origin bits；后续消费者跳过 rate 子树，反向仅原物理上下文计分。零率反事实选择
同一 cache slot 并强制所有别名，collection/alternate/final baseline 各次缓存独立。
布局由 bit 1024、metadata 34/1 与 b2_train_poisson_shared_v1==1 保护；113 noise
words 和 194-word MPI lane 纳入预算，count/origin 经精确整数 collective 传输。
GPU carry/incoming cache 及 shared start_tick>0 仍明确拒绝，尚未算持久恢复完成。

新增完整 202 身份模块：136 passed / 66 NVIDIA-only skipped，覆盖独立 fixed-count
surrogate 与一次 score FD、全别名零率弱差分、full/TBPTT、batch、2/8 local MPI、
首次惰性访问、覆写无效率、延迟 emission、重复 pending、异步 idle、indirect 原物理
rate、nested stream、inactive singular VJP、大于 2^24 精确 count、预算/回滚与旧库
capability 负对照。GPU carry 拒绝也检查完整事务回滚。30 Rust training 单元通过。
两个早期 pilot 112/56 与 128/64（通过/跳过）的原始测试、XML、日志与终态保留。
CUDA 为 translated kernel host C++ syntax 检查，无 NVCC/NVIDIA runtime 验收。

首轮完整回归 session 74215 exit 1（e2bf41）：1460 passed / 603 skipped /
163 failures-or-errors，1042.41 秒。163 项 message 均对应 No space left 或无法创建
临时目录；725 source hashes 一致，全部失败与输入保留在 disk-full-formal/。
首个失败 Metal case 在空间恢复后单独通过（session 15813 exit 0，edebcf）。
仅删除本任务可重建 Cargo incremental cache 145053019 bytes；记录见 cache-cleanup。
第二次 session 29496 exit 1（55cb17）在低于 1 GiB 预检阶段停止，未启动 pytest；
记录保留在 preflight-failed/。两次输入、运行器和最终测试身份与最后成功运行完全一致。
控制器随后增加短时空间复检、PID 与磁盘采样及 monitor write-error 子进程监督；
生产源码及测试未因资源错误改变。本轮成功运行期间最小可用空间 1617231872 bytes。

最终 session 58304 exit 0（894640），1180.97 秒：29 完整模块 **1622 passed /
604 NVIDIA-only skipped / 0 failed**，2226 精确身份、725 source hashes 全部一致。
Finalizer exit 0（751838），archive/verifier 子进程均 exit 0；787 文件逐项归档 hash
核验，并验证旧 Metal host prefix、两个 pilot 身份、全部资源失败分类及全部终态。
Runner SHA256：fde22ff449fa0f2d7066e9a6e624acf641bf8078e2b5abfbeab21bd7f2f4eeee。
Archive SHA256：2a1e4e380b6e6471410c889d02a11ddcae52aed1419da3b1d1511d508e4781bf。
证据：mpi-evidence/training-poisson-shared-gpu-20261005/README.md。

整体目标仍 active。下一阶段为 GPU numeric-profile 持久缓存、设备 count 验证、
store/restore/carry 与 Brian generated delay/migration 集成。NEXT.md 保留 CPU/GPU
profile 不混用、incoming/orphan 预算、所有回放从同一输入记录开始和 detached carried
score 的要求。完整 int64/unsigned/bitwise、自定义函数 AD、未支持对象/endpoint、
static Poisson 与 broader SDE/migration 组合仍在原范围。无云作业或上传；跨机器
按用户要求暂缓，NVIDIA runtime 验收未完成。本阶段全部进程已观察终态。

## 2026-10-05 GPU persistent Poisson checkpoint

继续原目标，未缩减范围。GPU Poisson carry/store/restore 已接入，独立单 pending
站点也使用持久缓存。CPU b2-poisson-draw-state-v1 不变；GPU v2 带显式 Metal/CUDA
f32 sampler profile，跨 profile 导入拒绝。恢复先查 header、identity、batch/sequence、
rate/count 与预算，再在设备上用保存的 rate/key 重算 count；host 不抽样或求导。
每 dispatch 最多 100 条记录，累计 uniform 工作量最多 10M；MPI 汇合校验错误。
空缓存也要求正确 profile、capability 与 validation symbol。

新 marker m[13]==36/m[33]==2，由 shared_v1/persistent_v1 能力保护。输入历史按
batch lane 独立 intern，含 orphan 记录并按最大 lane stride 分配；incoming params
和双端 buffers/identity/output history 纳入预算。每次 collection/alternate/final
baseline 都从相同不可变 incoming 记录初始化，origin=0 不重新计分。只输出最终
baseline 记录；delay/input/mask 更新保留缓存。数值状态/优化器及 restore 提交仍事务化。

270 身份共享模块 pilot：204 passed/66 NVIDIA-only skipped。新完整模块 188 身份：
134 passed/54 skipped，含独立 fixed-count surrogate FD、carried rate 无新 score、
full/TBPTT、batch、2/8 MPI、正/零 pending、无效 live rate/参数覆写、延迟 emission
split/restore、篡改/重试、大于 2^24 及 float-NaN bit 范围精确 count、异步 state trigger、
fresh sequence、不均衡 orphan 历史、profile/f32/work/旧库负对照。Generated Brian delay
migration 在正/零 rate 下与独立 CPU 比较；零率另与实际 Cython/真实队列比较。
未声称正率实际 Cython replay 或所有 Brian 组合。另 30 Rust 项通过，CUDA translated
kernels 与 launch-erased host C++ syntax 通过，不是 NVCC 或 NVIDIA runtime。

首轮 pilot 65/66/139：113 MPI bind 权限、25 Metal 不可见、1 last_result 只读断言误设。
保留日志、XML 与原测试；获准本地硬件执行及修正仅 latest-result 断言后全模块通过。
新 checkpoint 两次 pilot 都 114/48/4，async fixture 缺正确 non-primary state trigger/
可微 spike buffers；之后独立 probe 4 failures/2 skips 为 trigger 漏 external=false。
各原测试/错误/终态均保留；修正后完整模块及实际集成 probe 通过。旧 Poisson 能力
模拟库仅添加新前置 capability，仍测原缺失项；实际 Metal ready-mask 损坏回滚也通过。
Cargo 首条命令误指定不存在的 training feature，未编译即退出，错误记录保留。

最终 30 完整模块 session 63541 exit 0（a2561d），1250.98 秒：**1756 passed /
658 NVIDIA-only skipped / 0 failed**，2414 精确身份，727 输入源码 hash 在执行期间
保持一致。终态后只修 PoissonNoise docstring 的过时描述，AST 排除该文档常量后逐项
相同；保存原文件/原 manifest，verifier 从归档重建全部原执行输入与最终源码。
Finalizer exit 0（5a706e），archive/verifier 子进程均 exit 0；805 文件逐项 hash 核验。
另核原失败分类、全部 pilot 身份/终态、CUDA 输入/过程 hash 与旧 Metal host prefix。
仅删除 Rust 测试终态后的本任务 incremental cache 144444825 bytes；成功回归期间
最低磁盘可用 1443233792 bytes。无其他目录清理。全部本阶段进程已观察终态。
Runner SHA256：7053f2d69098c61b9e9d7b7a5409d6caf71b79f5d87b810e8a7ed0041de750b6。
Archive SHA256：a131f3f9204108529a2a1165efe00c578a1ee6681c8fd63052b00c8a0ea07e78。
证据：mpi-evidence/training-poisson-gpu-checkpoint-20261005/README.md。

整体目标仍 active。NEXT.md 保留正率实际 Cython arrival/uniform replay、warm pending/
collision/异步迁移、cached threshold/reset 与更广 SDE/migration、static v3/v4 Poisson、
完整 int64/timestep/unsigned/bitwise、pure custom function/callback AD、剩余 Brian
对象/endpoint 与安全历史退休的要求。无云作业或上传；跨机器继续按用户要求暂缓，
NVIDIA runtime 验收仍未完成。不能用本轮局部通过宣称原完整目标达成。


## 2026-10-05 Positive Poisson event / cached-clock acceptance

继续原目标，未缩减范围。新独立物理方程/Python FIFO oracle 按 emission/pending
counter key 生成抽样及 Cython uniforms，不读取 native counts，不解释 native SSA。
32 CPU + 32 实际 Metal 事件用例覆盖 serial/2-rank MPI、cold/warm、fixed/mutable
延迟、continuous/event trace、pre/post 顺序；实际 Cython 原 Poisson 实现及真实队列
逐边核对。保存/恢复、两次固定延迟迁移、480 分段边界、256 imported pending 及
48 碰撞用例已验证。另 8 CPU/Metal 用例验证 .4/.2 或 .1ms updater 与 .2ms neuron
下的 Poisson cache threshold/reset 消费、idle 边界、full/split restore 与 MPI。
所有 Poisson rate 固定 1.25/2.5；未声称任意 sampler 等价或所有 rate/weak VJP。

首轮 8 完整模块：631 精确身份、728 源码输入 hash 执行期间不变，496 passed /
134 NVIDIA-only skipped / 1 failed。session 12319 exit 1（759a83）；唯一失败是
旧 test_cached_updater_contract[clock-clock] 期待不同 updater clock 被拒绝。
现有多时钟前端已支持此行为，替换为正向 runtime 验证。原复杂 mixed uniform /
normal RK2 fixture 的 .4ms updater 另经独立闭式物理更新、native CPU、实际 Cython
比较通过（session 2913 exit 0，038956）。原失败 XML/logs/两份旧测试源码均保留。

两个改动的完整测试模块重跑 session 94271 exit 0（9b1c65）：143 身份，97 passed /
46 skipped / 0 failed，728 新输入 hash 不变。精确替换首轮两模块 131 身份并添加
12 新 clock 身份，其他 500 身份及源码不变；联合 8 模块验收 **505 passed /
138 NVIDIA-only skipped / 0 unresolved failures**，643 唯一身份。补充 event pilot
16 passed、clock pilot 2 passed，进程均终态。执行器/全部生产源码无改动；第一轮
parent 727 源码全一致，最终仅一份旧测试修正及新测试文件，与已验收 runner 同 SHA。
Prearchive verifier exit 0（746a28）；首轮失败未伪装成全量绿色。

证据：mpi-evidence/training-poisson-event-replay-20261005/README.md。
整体目标继续 active。继续 static v3/v4 Poisson（保留 analog 输入与同步状态契约）、
完整 int64/unsigned/bitwise、自定义 function/callback AD、异步队列及更多 trainable
cache/weak/migration/SDE 组合、剩余对象/endpoint 与安全历史退休。跨机器仍按用户
要求暂缓，无云作业或上传；NVIDIA runtime 仍未验收。

本阶段归档补记：finalizer session 52863 exit 0（644534），920 文件逐项 hash 核验；
archive SHA256 11774f96a32ecd9e3e1f4175d186caaf9f3a6cdd3f3e8997ffdb8a00c97acff9。
归档保存上述验收快照，此终态补记在归档之后。全部本阶段进程终态，整体目标仍 active。


## 2026-10-05 Static v4 CPU Poisson acceptance

继续原完整目标，不缩减范围。静态 v4 多状态 CPU 执行器直接加入 Poisson，
保留 analog 负数/分数输入、simultaneous 更新、synapse/reset 上下文、tied/feedback
投影、full/TBPTT、reset surrogate/detach、refractory clamp 与旧 normal/uniform key。
update/reset 共享声明 stream 只抽样/评分一次；实际首访阶段保存 rate；
reset-first 用 post-projection 上下文。正率 stopped-loss likelihood score、零率
实际单位 VJP 分类及 full single-sample one-count weak replay 均接入 CPU/MPI；
全 rank 同步新记录和失败，counterfactual 从原 initial/immutable incoming cache
开始，保持 global batch identity；RAII 恢复 baseline。只有 baseline 可提交。
恢复记录 origin detached，人工 rewind 复用旧 count/rate 不重新评分。

102 新静态用例覆盖独立物理方程和全部参数/初态 fixed-count surrogate FD、
正/零率、batch、full/TBPTT、2/8 MPI、nested/duplicate stream、reset-first、
取消/mask/singular VJP、预算/rollback、carry/checkpoint/tamper、refractory reachability
及实际 Cython 原 sampler uniforms（warm/cold、serial/MPI，rate 1.25/2.5）。
12 补充时间函数用例覆盖非零 tick 7/17、正/零率、constant-bank rate VJP、
full/TBPTT、batch 1/3、MPI 2/8 和分段保存恢复。首次 time probe 尝试静态
TimedArray，被现有 TimedParameter admission 拒绝；原源码/log/终态保留，
不得声称静态 TimedArray 已支持。另 30 Rust unit 项通过。

35 完整模块 session 34818 exit 0（2b539b），2031.59 秒，730 主源码输入 hash
期间不变：2066 passed / 705 CUDA-only skipped / 0 failed，2771 精确身份。
两份 MPI C 源码在运行期间额外记录，mtime 早于 formal start，终态 hash/mtime
再次核验。完整回归后只把新模块 GPU 拒绝测试改为直接 native CLI，避免 Linux/
无 GPU 环境先遇到 Metal library build。原模块保留且 hash 等于原 manifest；
AST 排除该函数和 subprocess import 后逐项一致。整个 102 项模块重跑 session
91967 exit 0（21105b），80.07 秒，新输入 hash 不变；精确替换模块身份后联合
结果仍为 2066/705/0，35 模块/2771 身份。生产源码无重跑间改动。

开发期两次 pilot 为测试 fixture 的 conditional compiler mode 错误，修复后
串行30、MPI93、refractory3、nested6 均通过；各诊断/终态保留。Prearchive verifier
首轮错误要求所有 skip reason 带 CUDA 关键字；705 skip 身份全部 CUDA，旧模块
存在 generic GPU/hardware reason，现按 CUDA identity + allowed reason 核验，
原诊断保留。仅清理已终态的本任务 Cargo target 315488721 bytes，已复制 runner
和证据保留；完整回归最低磁盘剩余 611115008 bytes，无 I/O 失败。

Runner SHA256：0781e5ed5f66170508186cae93486b63a6f334db0d0cdca1dcaffe3cd280f339。
证据：mpi-evidence/training-poisson-static-cpu-20261005/README.md。

整体目标仍 active。NEXT.md 明确保留 static v4 TimedArray、static GPU Poisson、
static v3（pre-projection subtraction 与 public layouts 必须保持）、完整 int64/
unsigned/bitwise/custom function AD、剩余对象/endpoint、安全 history retirement
和更多 cached/weak/SDE/queue/migration 组合。无云作业或上传；NVIDIA runtime
仍未验收，跨机器按用户要求暂缓。不能以本轮局部实现宣称原完整目标达成。

本阶段归档补记：finalizer 首次 copy 使用原 manifest 比较已改动的 guard 测试，
在归档前退出；原错误和终态保留，改用 rerun manifest 后成功。finalizer session
86681 exit 0（a79200），801 归档成员逐项 hash 核验。Archive SHA256：
dd575a5eb51851d144dc94cdabab6cb989dd3aa16cea30f9cc7db9e7ee3514df。归档保存补记前的验收快照；此补记在归档之后。
所有本阶段进程均已观察终态，整体目标仍 active。


### 静态 v4 TimedArray、SDE 扩散与 Poisson 表值率（2026-10-05）

静态 v4 前端识别 TimedArray（包括字面时间），保留物理 bank 和名字别名。
CPU/legacy GPU 执行使用现有采样/VJP 节点；GPU 新增描述符和独立 capability，
旧 ABI 缺失/版本不符明确拒绝。时间/列索引 detached，表值/普通参数/所有初态
VJP 保留，原静态模拟投影/复位顺序不变。CPU MPI 同步局部列索引错误。
序列边界可更新冻结 bank，状态/optimizer/时钟/RNG 和已观察 Poisson rate/count
保留；更新预算合计包含缓存，任何失败不提交。

新增 278 精确身份，198 passed /80 CUDA-only skipped；物理 RK、1D/2D、
别名、literal time、全参数/初态 full/TBPTT、analog batch、feedback、carry/restore、
更新/预算/owner rollback，以及 Euler/Heun/Milstein 的独立物理扩散梯度和
实际 Cython 原 integration/event code 均验证。CPU Poisson 表值正率 score /
零率 full one-count weak VJP、mask、旧记录 rewind 和原 Cython sampler 也验证。
首次 Euler fixture 被 Brian 原 .3ms 网格检查拒绝，预算 fixture 先命中 metadata
guard；原错误/终态保留，修复后 63 预检全通过，最终 formal 覆盖完整模块。

21 完整受影响模块 1394 passed /533 NVIDIA-only skipped /0 failed，
1927 精确身份；733 源码输入（包括初始 C manifest）始末 hash 相同，
正式 session 34554 exit0 (7dc415)，
2372.02 秒。原 TimedArray+Poisson 时间 probe 在最终输入上
12/12 通过（非零 tick7/17、batch1/3、full/TBPTT、MPI2/8）；40 Rust unit 通过。
补充实际 Metal 表值 VJP 使用更紧容差独立 FD，36/36 通过（六种确定性/随机
积分器组合、full/TBPTT、serial/MPI2/8），最大绝对误差
1.37186127e-07；与 formal 同一源码输入/runner。
CPU、实际 Metal、本地 MPI 验收；CUDA host capability controls 不等于 NVIDIA。
清理已终态且 binary 与留存 runner 相符的本任务 Cargo target，证据保留。
Runner SHA256：5dcebba31ee9428d5b6507434db086430b2d7e9660d201a2ee3841baa0d47fb8。
证据：mpi-evidence/training-static-timed-input-20261005/README.md。

整体原目标继续 active。NEXT.md 保留 static GPU/v3 Poisson、完整整数和 custom
function AD、未完成对象/endpoint、安全 history retirement 与 broader cached/
weak/SDE/queue/migration 组合。无云作业/上传，跨机器按用户要求暂缓。

本阶段归档补记：finalizer session72553 exit0 (25f897)，802 归档成员逐项 hash
核验。Archive SHA256：874a700cd8e04edef44af77dbb7fc545ad0bb0a728d43c23df2c07df74dfd405。
归档保存此补记前的验收快照；此补记在归档之后。全部本阶段进程均已观察终态，
原整体目标仍 active。后续使用本阶段留存 runner 或重新构建源码；默认旧 release
二进制未替换，不能把旧二进制当作本轮验证的运行时。

2026-10-05 scalar v3 development checkpoint (not accepted phase): direct native
CPU/Metal time, normal/uniform/Poisson, 1D/2D TimedArray, scalar threshold/reset
schedule, carry/cache/weak/VJPs and local MPI are implemented. All-parameter and
initial FD pilots passed, as did corrected negative cache controls and 44 Rust
units. Complete immutable 39-module regression collected 3,500 identities and
is running as session6078. The preceding static-v4 full run session28622 remains
live with two missing-reference-executable harness failures plus a separately
recorded strict two-case passing rerun. Neither full run is relabelled green.
The overall goal remains active; cross-host/GPU cloud resources remain deferred.

2026-10-05 int32 bitwise development checkpoint (not accepted phase): native
CPU/Metal/CUDA-shared integer AND/OR/XOR/invert/left/arithmetic-right shift, typed
AST/augmented dynamic assignment and versioned GPU capability are implemented.
Exact large signed values, stochastic dynamic VJPs, nonroot invalid shifts, lazy
branches, integer NaN payloads and scalar/vector float VJPs:77 development passes
with33 explicit NVIDIA-only skips; Rust45 pass. Frozen739 inputs/1644 assets
are prepared; its full40-module controller has not been launched. Earlier full
regressions28622 and6078 remain active. The original full goal remains active.

2026-10-05 pure-function development checkpoint (not full phase acceptance):
bounded pure expression descriptors and source-inspected stateless Brian
callbacks now inline into native SSA at neuron/synapse/regular resolvers, with
single argument evaluation, lexical closure constants/SI quantities and native
VJPs. Loaded/source semantic and fixed standard-function identity guards prevent
silent replacements. Final frozen complete module54 pass13 NVIDIA-only skip0fail,
67 exact identities,741 source inputs unchanged;46157 exit0 chunk036544.
Earlier mutable-source, threshold-fixture and CPython-encoding failures remain
retained. Latest full48 modules/3910 IDs is prepared, not launched; it includes
complete older bitwise110 IDs plus seven affected Brian front-end modules. Use
this latest run after28622 terminates instead of an extra older bitwise full.
Overall goal remains active; cross-host deferred; no cloud/NVIDIA execution.

后续元数据保护补记：不透明包装器及类属性先按精确类型拒绝，不执行用户
getter。最终68身份模块55 passed/13 NVIDIA-only skipped/0failed，84774 exit0
chunkfeafa9，101.23秒；741冻结源码保持不变。原67身份快照与开发归档保留。
旧完整28622已终态exit1 chunk1363f6：2443 pass932 skip2缺少参考程序的失败，
3377精确身份和736源码hash核验，独立补齐资源后的2项通过，原full不改记为green。
最新48完整模块3911身份已启动，**LIVE46085**，冻结于
/atlas-storage/0002/b2-pure-functions-validation-o77bo80u；原scalar39模块**LIVE6078**。
最新预计2845 pass1066 NVIDIA-only skip，仅实际终态/XML/hash审计通过才接受。
旧bitwise40完整控制器未启动，其110身份完全纳入最新48模块。原完整目标仍active。

2026-10-05 scalar full acceptance and sequential-function development:
Scalar full6078 observed exit0 chunk3404a3:2530 passes,970 NVIDIA-only skips,
3500 exact IDs across39 complete modules;738 original executable inputs and
1642 original assets unchanged. Raw inventory also retained736 unchanged binary
AppleDouble metadata files; audit checks format magic and logical counterparts.
The phase acceptance and per-member-verified completed T7 archive are recorded.
Old static full28622 remains failed with its original two harness failures.

Sequential local assignment/zero-arity/unused-argument inlining now uses native
Sequence/IntegerSequence with eager primal and right-only VJP semantics. Brian
array argument/alias mutation and augmented dtype changes are refused, including
helper-returned aliases. Pure body modulo matches NumPy at negative int32 bounds.
Final frozen181 IDs:137 pass44 NVIDIA-only skip0failed,97775 exit0 chunk3562fb,
143.48sec. Exact IDs,742 source hashes and1650 assets passed module audit; Rust48
units passed, generated CUDA host C++ syntax only, default runners unchanged.
Earlier14 failures and subsequent173-ID development result remain retained.
Latest49 complete modules/4024 IDs is LIVE78304 under
/atlas-storage/0002/b2-pure-statements-validation-b02expqg; expected2927 pass1097 skip
only after a real terminal/XML/hash audit. Older48-module3911-ID full46085 is
still live. This is progress within the original complete stochastic/dynamic
objective, not whole-goal completion; cross-host deferred, no cloud/NVIDIA run.

2026-10-06 conditional stochastic-function and dynamic-synapse checkpoint:
Scalar pure descriptors now admit lazy conditional/operand-returning Boolean
expressions; source-inspected Brian callbacks admit identity-checked eager NumPy
selection/logical operations. Boolean selections preserve threshold surrogates,
eager AND/OR VJPs use both real gate values, and GPU hosts require the new
versioned eager-Boolean capability. Old untagged selection JSON remains stable.
Boolean function units are inferred before result-kind flags; NumPy no longer
invokes the Boolean marker as a callable unit rule. Genuine callable unit rules
and wrong return-type validation have explicit regression coverage.

Rust50 units/release passed; actual complete three-module development63872
exit0 chunk708a33:242p88 NVIDIA-only skips330 exactIDs. Complete core function
module81382 exit0 chunk3ed8ad:50p6 named Cython-only skips56 exactIDs, including
real standalone C++ execution. Strict audit8495 exit0 chunkf8772d verifies743
source hashes,1652 original assets, all exact identities and native/runtime
provenance. APFS source-only core snapshot avoids automatic T7 AppleDouble
regeneration; its798 logical assets equal the native snapshot's logical assets,
and854 original metadata entries remain retained/format-verified. Original24
Boolean-fixture failures, missing-plugin/config/collection and metadata failures
remain preserved. CUDA checks are translated host syntax, not NVIDIA hardware.

Latest52 complete modules/4302 plannedIDs is LIVE61441, initialchunk0fddf7,
`/atlas-storage/0002/b2-pure-branches-validation-vc43319d`; await actual terminal and
its own strict audit. Older48/49-module full46085/78304 remain separate and
live. Evidence and remaining work are in
`mpi-evidence/training-pure-branches-20261005/{README,NEXT}.md`.
The original full stochastic/dynamic implementation goal remains active;
cross-host deferred, no cloud/NVIDIA run or default-runner replacement.

2026-10-06 cached weak/SDE replay and Metal host development (not accepted):
Independent hand recurrences now cover cached trainable Poisson, shared Gaussian
noise, two coupled states, delayed pre/post plasticity, warm Cython import,
all-parameter/initial FD, zero-rate full counterfactual weak derivatives and
positive-rate joint-distribution scores including idle clock observations;
carry/read-only/restore/invalid refresh checks are included. Complete133 run
returned91 pass41 NVIDIA-only skip1 Metal/MPI8 client timeout. The same exact
case reproduced120s and300s timeouts; failures are retained, not accepted.

Finite positive configurable request_timeout keeps default120 and checkpoint
independence; corrected actual process-group timeout tests passed11 cases.
Metal host now reuses immutable shader/pipeline objects across full weak replay
calls within one loaded library. Buffers/queues/model state remain per call.
Cleanup test sleeps are isolated from subprocess polling. Actual final40-ID
Metal/MPI8+transport diagnostic is LIVE67312; source744/assets1654 frozen under
training-metal-pipeline-cache-20261006. Complete53-module4375-ID regression is
prepared, not launched; numerical/performance acceptance remains pending.
Older full46085/78304/61441 handles are now missing and corresponding processes
absent:48 has controller exit0/XML3911 but missing tool receipt,49/52 are partial.
None is fabricated as accepted. Original full goal active; no cloud/NVIDIA job.

后续实际终态补记：Metal缓存诊断67312 exit1 chunk4ecec2，39p1F40IDs；原
Metal/MPI8 full weak/pathwise/all-initial差分用例通过。唯一失败是 .3s超时fixture
在外置盘尚未启动；加入就绪同步后79040 exit0 chunk0e1f8c，1p38deselected。
生产transport仍限制启动时长。最终同生产源码的四完整模块172IDs已启动LIVE59851；
使用自有APFS编译临时目录和T7不可变源码/证据，源744/资产1654，严格终态审计
前不接受。完整53/4375回归尚未启动，记录于training-metal-pipeline-full-20261006。
原完整目标继续active，剩余语义与CUDA实机缺口仍需完成；跨主机暂缓，无云作业。

2026-10-06 完整回归进展更正：原133实际90p42skip1F；四模块59851实际
exit1 chunk2a20da，129p42CUDA skip1请求超时。恢复测试显式300秒后
原失败99649实际exit0 chunk8c79cc，1p17.70秒，生产输入保持一致。
新独立快照sxfsoa5_的完整53模块4375身份已启动exec68504初始chunk244d57，
本次自身收集已核对，尚未验收。历史失败保留。冻结源码之外的新开发独立验收；
原完整目标active、跨主机暂缓、无云/NVIDIA作业。

2026-10-06 动态突触对象端点：实现物理写入预发现、Synapses源/目标端点、
summed写入目标突触、声明顺序无关的运行时/优化器初值区分与目标链MPI owner。
开发CPU/MPI2/8 10p、延迟/暖队列4p、三级对象调制1p实际终态已保留；
测试fixture/非法Brian别名失败亦保留。最终745源码/800逻辑资产的APFS快照
14完整受影响模块CPU回归LIVE64790初始chunk57a397，尚未终态验收；新增eta
断言由本次冻结验证。新版端点Metal仍未运行。原Metal阶段full53 LIVE68504
属于旧前端源码版本。完整目标继续active，无云/NVIDIA作业，跨主机暂缓。

端点回归更正：64790发现regular owner重复目标偏移，SIGINT停止，实际exit2
chunk584ee4，部分185（73p86skip26F），原失败保留。修复后15887实际exit0
chunk451f12，2p包含MPI8。新freeze98928实际exit0 chunka2b805；当前14/705
CPU完整回归LIVE20333初始chunk781817，原53GPU回归LIVE68504另行等待。
当前原目标active，尚未完整验收；源新版端点Metal未运行，无云/NVIDIA新作业。

端点CPU阶段20333已actual exit0 chunkf89c48：433p272明确硬件skip/705，
14模块。strict audit实际exit0 chunkf81036，745源/800资产等身份/runtime匹配。
此后独立开发已补齐动态索引参数与shared整数/布尔字面量；CPU/Cython8个
新开发用例分别通过，所有非法Brian fixture失败保留。最新729/14准备并另行
冻结验证，不能把旧705验收算作最新main或Metal通过。原full53 GPU68504仍运行。
原目标active，无云/NVIDIA新作业，跨主机暂缓。

最新参数索引/类型版729/14 CPU回归LIVE82989初始chunk04e5b5；原53/4375
GPU仍LIVE68504。归档前一已验收CPU705，实际60503 exit0 chunk421037，
857成员逐哈希相等，sha256038688146f6e918de0e8d8b7c1745c52ac9300d2ab12b1b3cc27a9ece48d38c1。
归档不代表最新参数索引或Metal验收；原目标active，不能缩减全语义范围。

2026-10-06 随机突触端点组合：边状态Euler SDE、缓存uniform/normal、跨对象
权重调制、异质delay、full/TBPTT全参数/初态差分、实际Cython随机buffer重放，
本地MPI2/8 carry/read-only/restore及prune/regrow迁移。
完整模块CPU32实际85257 exit0 chunkc94065，12p20GPU硬件skip；strict audit
actual exit0 chunk240725，源746/801资产和全部身份/runtime匹配。生产与729
阶段一致，只新增一个测试模块。旧729 CPU82989、原53 GPU68504仍保持真实handle。
这些组合已支持，验收未宣称全部完成；原目标active，Metal/NVIDIA等实际门仍待验证。

随机端点CPU32已可恢复归档：77183 actual exit0 chunkfde730，848成员全部
hash匹配，sha2565db16652c512ba5e5faabc84ef9bd46d5ac45c8db3390e18cf4f437322b2b909。
原归档说明文件并发变化导致hash校验失败，失败23422 exit1 chunk690c81已保留；
原始源码/证据未删除。原729 CPU82989和53 GPU68504真实handle仍live，
本轮新增组合证明归类progress，完整目标active且未缩减，Metal/CUDA门仍需完成。

2026-10-06 后续实际验收：729/14 CPU82989 exit0 chunk72b427，441p288明确
GPU skip；strict audit exit0 chunk9e25dc。Heun/Milstein 端点完整81身份79630
exit0 chunkc5883f，31p50硬件skip；strict audit exit0 chunkccee7e。
两份 CPU-only 归档全部成员/source-after-check 相等：12725 exit0 chunk9a2c5b
843成员；5978 exit0 chunk36a692 846成员。各阶段 cpu-archive.json 保存路径及
SHA256。旧32与705验收继续保留，未混为最新实现数字。

新增显式 external_state_inputs：输入组可变浮点字段通过同单位时间表、固定物理
列与消费者 clock 读取，接入连续 SDE/pre/post/summed/regular 的同一 native
timed_parameter ABI；不执行源信号生成器，也不静默折叠初值。训练图写入拒绝。
共享 batch、表 VJP、边界替换和checkpoint 契约详见 NATIVE_TRAINING_EXTERNAL_INPUTS.md。
开发 CPU/Cython/MPI19实际55043 exit0 chunk61240a。初始 namespace、Brian
标量/向量顺序与独立参照的 optimizer/状态区分错误保留原失败 XML，并已修正。
新增时钟/只读边界与最终冻结验收尚未完成；旧 Metal53 exec68504继续等待实际终态。
原完整目标 active，任意外部 linked/运行时索引和所有硬件验收仍未宣称完成。

外部状态后续：4个边界检查28823实际exit0 chunk9ce1d0，包括异步消费者clock
和公开dynamic入口。审查复现输入alias指向选中内部存储会忽略输入表：14439
actualexit1 chunk73d134；新增物理owner拒绝后65704 exit0 chunk5ff208，1p。
原248完整CPU77476冻结版本继续运行；新owner修复freeze98663 exit0 chunkf5a1ba，
747源/802资产。当前完整五模块249 CPU95379初始chunk68dc33已启动，Metal尚未
实测新接口。旧full53 Metal68504仍在原版本运行，无云/NVIDIA作业，目标active。

最新249 CPU95379 actualexit0 chunkec54e4：148p101明确硬件skip；外部接口44中
24p20skip。strict audit actualexit0 chunk76f3a5，747源/802资产、完整精确身份
与runtime/controller哈希一致。修复前248 actual77476 exit0 chunka64620，147p101skip
仅代表旧快照；当前已包含所有权拒绝。新外部接口 Metal/CUDA 仍未实测，原目标
active，不能以CPU子集验收取代全部实现及硬件门。

2026-10-06 外部字段运行时索引：新增冻结 int32 列路由，通过已有 native
integer_parameter_gather 与 timed_parameter 保持代码块前物理索引快照；列选择
停止梯度，表值仍有VJP，越界有原子失败。独立连续Heun/事件驱动指数推进、
full/TBPTT全表/参数/初态差分、实际Cython、batch2、MPI2/8与失败回滚扩展检查
14567 actualexit0 chunk0ee09a，23p12明确硬件skip28deselected。
冻结 actualexit0 chunkc0b756，748源/803资产，源相对owner249仅两处前端改变和
新增一个测试模块；native/kernel/runtime未变。六完整模块312 CPU已启动，
旧Metal53 exec68504仍独立等待真实终态；原完整目标active，无云作业，跨主机暂缓。

运行时索引最新CPU312 actual18203 exit0 chunkf0296b：171p141明确硬件skip；
新增模块23p40skip/63。strict audit actualexit0 chunk1d1cb7，全部精确身份/
748源/803资产/runtime/controller一致，所有前249身份保留原顺序。当前仅CPU/MPI
验收，旧Metal53仍live，最新前端Metal/CUDA未实测，完整目标active。

运行时索引CPU312可恢复归档实际85018 exit0 chunk862d0c，871成员全部与manifest
及源文件二次hash一致，SHA2566ef82352fb06bc98c88b86dd0de2358b698a10b6f8efa072eb627ca76b3cbb62。
来源与原始runtime证据亦保留；归档仅代表CPU阶段。旧Metal53真实handle68504仍
running，已推进到98%，不能伪造终态或重启。当前312覆盖旧249/81，后续只运行
当前312和未覆盖的729 GPU门；原完整目标active、跨主机暂缓、无新云作业。

旧Metal53/4375实际exec68504 exit0 chunk487309：3194p1181skip，源哈希一致。
strict audit actualexit0 chunk5a1356（/private/tmp/b2-metal-full-audit.log），744源/
1654资产、53模块所有精确身份与runtime一致。这是旧cache/weak/SDE阶段来源，
不能替代新端点/外部输入前端。现729/14 Metal已启动62563初始chunkaffbd7，
其后执行最新外部前端GPU验收。原完整目标active，无云/NVIDIA作业。

2026-10-06 神经元外部linked：同一显式表接通神经元积分、margin/threshold、
reset、regular与refractory（固定/信号条件），保持同块前索引快照与只读、输入
VJP/零alias初态伴随。非法未暴露drive_post参考fixture19失败保留，修正21p。
扩展固定/field refractory独立物理/全梯度/实际Cython与MPI开发23946 exit0
chunka64bdc，45p60deselected。冻结actualexit0 chunk64efc2：749源/804资产，
七完整模块417身份CPU已启动；新增105，仅dynamic frontend改变，native/kernel
runtime未变。当前前端GPU与自身最终CPU验收尚待完成，原目标active，无云作业。

神经元linked最新七模块417 actual59196 exit0 chunk30f69e：216p201明确硬件skip；
新增105中45p60skip。strict audit actualexit0 chunkb47e08，所有前312身份保留、
749源/804资产/runtime/controller一致。原Metal53归档19310实际exit0 chunk7d1442，
1746成员全部hash/source-after-check相等，SHA8c4af530fbcb52d486be7cd308a70845bf210150985179e0c749bd005e2971cf。
新端点729 Metal62563仍live；当前linked最新GPU需其结束后单独运行。原目标active。

2026-10-06 外部int32/Boolean：单银行 high16/low16 + 现有 typed IR 保持GPU精确
整数，不新增native/kernel/ABI；linked神经元积分/门/reset/regular和synaptic路径
共享consumer-clock与前块索引快照。新增边界update_external_state_input校验原形状/
精确离散值，一次原生银行更新，不推进状态。离散输入VJP停止，原float VJP保留。
最终开发22980实际exit0 chunkba7f65：22p24硬件skip，含Heun乘性噪声/fullTBPTT
全梯度/固定normal原始Cython/MPI2/8/每个prefix精确状态/shared/只读/恢复。开发
失败保留，未伪装通过。冻结55729 exit0 chunk43ab25：750源/805资产，4个Python
生产改动+1模块；Rust/GPU/runtime不变。最新463/8 CPU88549与Metal72609已启动。
原729/14 Metal62563实际exit0 chunk4d6773：585p144 CUDA-only skip，strictaudit
exit0 chunkded1e9，全部身份/745源/runtime相等；该旧版本不能代替最新外部前端。
linked417 CPU归档原process终态因工具输出丢失而不可取，不伪造；独立readback
actualexit0 chunk0ebe66确认873成员全部manifest匹配，SHA5541dba763322456d2199c9027ccd15ebecae25905ca6bc2385d79c0654f5dd6。
原完整目标active；按样本外部表、外部整数根索引、非int32/任意mutable callbacks、
安全历史自动回收及最新NVIDIA门仍未宣称完成；跨机器暂缓、无新云作业。

最新discrete463/8 CPU88549实际exit0 chunke5de04：238p225GPU硬件skip，strictaudit exit0 chunkecd73d；全部身份、750源/805资产与runtime/controller一致。Metal72609仍在自身冻结快照运行，不能由CPU推定GPU通过。完整目标active，无新云作业。

最新discrete463/8 Metal72609实际exit0 chunkec8ce3：353p110 CUDA-only skip，
strictaudit exit0 chunk721d45；463完整身份、750源/805资产和runtime/controller一致。
新增模块34p12 CUDA skip，精确整数控制、Heun路径导数和localMPI2/8 GPU恢复通过。
本阶段覆盖前417/312/249/81当前前端；不重复被覆盖旧快照。整体目标仍active，
不能由CPU/Metal通过宣称剩余语义或NVIDIA已验收；跨机器暂缓，无新云作业。

最新discrete CPU/Metal可恢复归档11406实际exit0 chunk24d767，906成员全部hash/readback/source-after-check一致，SHA3afb5eaca435a329b89559bafe6581a144b13f90a1c937cfe2622e918c8e6c80。原证据保留；整体目标active，继续尚未实现语义与NVIDIA实测，不将本轮通过缩减成总体完成。

2026-10-06 外部int32根selector：新增consumer-clock只读detached缓存供native
间接读写，按每个reader动作clock刷新，保持owner pendingclock采样；不新增Rust/
GPU/ABI。sampling前缀平移全部delay布局start/end/event，延迟更新保留旧pending
与restore已实测。开发失败保留；合法Cython/全VJP/fullTBPTT/Heun/事件驱动、
MPI2/8、table更新/atomic-oob回滚与恢复通过。冻结实际exit0 chunk88ead3，751源/
806资产，唯一生产更改dynamic frontend+64新身份；所有前463顺序保留。
CPU完整527 actual5524 exit0 chunk4ee99d：262p265硬件skip，strictaudit exit0
chunk661838。Metal完整actual23226 exit1 chunkf2134a：393p130CUDA-only skip4F，
全为错误matcher不匹配nonfinite dynamic GPU result；原失败未删。
仅更正一个测试字符串，AST证明其余fixtures/物理/回滚断言和750其它源不变；
新freeze自身collect527完全一致，四原失败身份1488 actualexit0 chunkf2b2c1：4p。
合并audit actualexit0 chunkc94439：397p130CUDA-only skip，各case有实际来源；
不伪造一次GPU全量exit0。原目标active，剩余per-sample/history/更广语义与最新CUDA
仍须继续，跨机器暂缓，无新云作业。

外部根索引阶段可恢复归档5628 actualexit0 chunk565b57，1755成员全部hash/readback/source-after-check相等，SHA07329c562b523e301401d97d93788d2fcc32374cd8261e20d7f3d4fddbb69db9；main751源匹配修订snapshot，检查actualexit0 chunk025502。原目标active，下一步继续per-sample表/history与剩余语义/最新NVIDIA。

### 按样本外部输入阶段（2026-10-06）

BatchTimedArray提供独立样本的float/int32/Boolean外部状态表，可混用共享表。
原生detached只读sample_index保存行编号，custom initial/carry必须匹配；直接/
间接写及迁移拒绝。单位、银行更新、原形状VJP、Poisson零率全轨迹重放保持契约。
只改两Rust生产源、五Python生产源，新增78身份；GPUshader/ABI和默认bin未改。
新隔离native由对应全部Rust源构建，Rust lib50实际通过。

十完整模块605身份：CPU2531实际exit0 chunkf0eaeb，296通过309硬件跳过；
Metal44268实际exit0 chunk95b57a，453通过152CUDA-only跳过。audit.py分别实际
exit0 chunk1f7ab3/f26f79，752源码807资产及逐身份/native/controller一致。
独立原始Cython、全银行/初态有限差分、MPI2/8、更新/新训练器恢复、失败原子性
均覆盖。原始开发fixture失败保留。证据见training-batch-external-inputs-20261006。

整体目标保持active；安全Poisson历史回收、非int32/任意mutable Python、最新
NVIDIA实机仍待实现或验收，跨机器继续暂缓。本阶段没有云调用。

### 显式随机历史回收开发阶段（2026-10-06）

Poisson回收及精确continuation证书已接入原生CPU/Metal/CUDA共同的边界接口；
GPU抽样/恢复验证仍由原后端执行，没有shader/ABI或默认bin替换。已证明不可达
的clock/emission/pending记录才删除，未知记录保留；旧检查点仍可恢复，回收后
非法分支/回退原子拒绝。delay/input/mask成功结果重新绑定证书。

新Rust lib50实际通过；暖队列原始Cython CPU/Metal2项、rate/mask更新4项、
pre-zero发射及delay alias CPU/Metal/MPI2/8 6项专项均有实际exit0。
完整22模块2101身份已冻结754源809资产，CPU70544仍live，尚不能宣称完整验收；
Metal待GPU开发作业结束后启动。最新NVIDIA未运行，整体目标active。详情与原句柄
见training-poisson-retirement-20261006/NEXT.md。初始MPI沙箱失败和开发fixture
缺失未删除，最终完整回归必须用自己的实际终态及逐身份源码审计。

### 回收预算修正（2026-10-06）

实际反例证明原回收只分别检查缓存/模型预留，8KiB时忽略合并工作空间并报告0。
已增加复制/抽样校验前的checked合并预算；site临时表仅覆盖已有缓存身份，成功
报告保守预留。新native单独构建，原CPU70544与开发45792继续，原冻结输入不变。
新Rust50实际通过；CPU/Metal static/dynamic预算专项4通过2CUDA跳过，失败不提交、
提高预算后carry成功。新22模块2107身份版本另行冻结，尚未完成完整验收；目标active。
证据training-poisson-retirement-admission-20261006，原版本反例与日志保留。

### 回收检查点测试入口及最终回归（2026-10-06）

原完整冻结CPU中唯一新进程测试未找到冻结目录/.venv，child未启动。测试现改用
调用方lexical sys.executable（实际main/.venv/bin/python），AST证明只有此路径
与import sys变化，生产源码/native/所有断言不变。CPU/Metal主workspace实际2项
新进程通过；重新冻结后唯一CPU身份68486 actualexit0 chunk332b1c，1项通过。

修正版CPU60720完整22模块2107身份继续，最终Metal49818完整相同身份已启动且
自身collect相同。原CPU失败将保留，只在唯一入口错误且其它2106结果无失败时
允许用1实测复验替换；不是一次完整CPU exit0。最终GPU及CPU组合仍未验收，
整体目标active。最后证据入口training-poisson-retirement-checkpoint-20261006。

### 回收阶段CPU最终组合验收（2026-10-06）

修正版60720完整22模块2107身份实际exit1 chunkfb79b3；唯一失败为冻结目录新进程
启动路径，所有其它模块exit0。仅该身份68486 actualexit0复验通过，断言/生产源
AST及native不变，组合严格audit实际exit0 chunk50a771：963通过1144明确硬件跳过。
原exit1/失败与逐身份来源保留，不能宣称一次完整CPU exit0。新回收模块33通过66
硬件跳过。最终Metal49818仍live，整体目标active；最新CUDA仍未实测。

### 回调提前返回和浮点控制流（2026-10-06）

提前返回已贯穿静态 if/for/while/else，原始 Brian 正向、全部参数和初态
有限差分、共享多状态 SDE 与动态计划均有实际证据。修正版完整8模块666身份：
CPU95469实际exit0 chunk69f6d6，严格audit ed109b：282通过384硬件跳过；
Metal85377实际exit0 chunkc635ff，严格audit8f1a22：475通过191CUDA跳过。
759源815资产全部匹配。原663身份两个过时拒绝断言失败保留，不伪造原结果。
归档20386实际exit0 chunkb05252，899成员逐字节回读一致，SHA
a54e0db2f54b46388f7f25bdfa8b8e8a2f5294636f1c3903478e31cde9a1646f。

旧回收Metal49818句柄、PID、冻结临时目录均已消失且无真实终态，前文live
状态不再适用；其部分结果不能作为完整GPU验收。已核验恢复暖回调CPU归档
904成员，并将新冻结源码、编译和证据路径改为持久T7。

本轮继续支持有限不可变浮点常量/已知标量局部变量控制分支与有界while，
保留range的int32限制及未知数组/opaque对象拒绝；无native/shader/ABI改动。
开发77274实际exit0 chunkd4690c：117通过188硬件跳过，完整9模块CPU/Metal
冻结验收待运行。更广Brian语义、旧回收阶段完整GPU与最新CUDA仍未全部验收；
整体目标active，跨机器暂缓，无云上传或计费作业。

浮点控制流边界复验发现 NumPy float32(.1)/float32(.3)循环被转成Python
双精度后多执行一次，并改变实际脉冲；原始反例69782真实exit1 chunk122275
保留。静态计算已保留NumPy实数标量原精度，13551真实exit0 chunkdc751f
同反例1通过2GPU跳过。新freeze760源816资产9完整模块897身份；CPU11595
真实exit0 chunk1e8797，严格audit42137a：363通过534硬件跳过。修正版
Metal20939仍live，前一894身份Metal65587自身快照继续，不能代替修正版。
无native/shader/ABI/default bin变化，无新云作业，原完整目标仍active。

修正版浮点控制流897/9最终Metal20939实际exit0 chunk62355b，严格audit
2d7100：631通过266仅CUDA硬件跳过；CPU为363通过534硬件跳过。两边
完整身份、760源816资产和native/controller均匹配；NumPy单精度物理反例
实际CPU/Metal通过。无新云作业，默认二进制和native/shader/ABI未改。
最新CUDA、更广离散存储/可变Python语义、旧历史回收完整GPU重验仍待推进；
整体目标active，跨机器暂缓。详见training-pure-real-control-precision-20261006。

本轮最终归档44555真实exit0 chunk898fb7，910成员全部hash/readback相等，SHA100f07f9834a1085c5cb81aba63022e83a9118c0576fdeeb7b06cd86cf8a18fe；最终main760源与验收快照一致（73a24e exit0）。整体目标active，剩余语义与硬件验收未冒称完成。

### 私有浮点数组别名与原地更新（2026-10-06）

显式numpy.array(x,dtype=float/np.float64)创建已知私有浮点存储，copy=True
可省略；增强赋值刷新同一分配的所有活跃别名，独立副本、先前派生值和重新绑定
保留独立值。标量调用的零维数组语义亦有原始Brian物理验证。copy=False、
asarray、未知/窄精度、借用调用者存储写入和helper逃逸仍拒绝；无Python
构造器执行、native/shader/ABI/default bin改变或云调用。

开发80098实际exit0 chunk322ac0：146通过238硬件跳过。新模块271身份
覆盖静态/动态Heun/Milstein共享/混合噪声、全部银行和初态VJP、TBPTT/reset、
本机MPI2/8及pre/post/regular/synaptic ODE正向。冻结761源817资产10模块
1168身份；CPU26789实际exit0 chunk85c10b，strictaudit476324：458通过
710GPU硬件跳过。实际Metal新模块183通过88CUDA跳过，完整45364仍live。
整体目标active；更广离散存储/可变Python、最新CUDA和旧回收全量GPU仍待继续；
跨机器暂缓，无新云计费作业。证据training-private-array-aliases-20261006。

私有数组完整10模块1168身份最终Metal45364实际exit0 chunk9e1302，
严格audit5f536f：814通过354仅CUDA硬件跳过；CPU458通过710GPU硬件跳过。
761源817资产、原生运行器和controller/完整逐项XML匹配；主源码也与快照一致。
全梯度及原始Brian别名/独立复制/重绑定/标量零维行为已实测。整体目标active，
不冒称更广可变Python/离散存储或最新CUDA完成；跨机器暂缓，无云调用。

私有数组最终归档13712实际exit0 chunk5bccfe，912成员全量hash/readback一致，SHAa52445a1f3c449dadd78fe3f14993afdae20210c937f1cdb84b842398739726c；main761源一致。原完整目标active，继续剩余语义与硬件验收。

### 数值包装函数语义修复（2026-10-06）

实际原始Brian反例发现任意追随__wrapped__会丢失数值装饰器操作，7/20脉冲
不同；57989真实exit1 chunk1325f1原始日志/XML保留。现只识别有原始绑定的
规范check_units代码，其他数值包装器读取并验证其实际code body；有界
固定/转发/静态索引参数包不执行Python而绑定，普通元组求值不能被静默删去。
首轮物理反例及原纯/暖回调5077实际exit0 chunk7a9f27：51通过34硬件跳过。
扩展开发77747实际exit1 chunkcd49e9：455通过814跳过，唯一失败为我在进程
加载后追加测试导致源身份正确拒绝而与matcher不同，原失败保留。新不可变
762源818资产11模块1855身份CPU14383运行中；Metal尚待运行。原9417/118
全套继续自己的修复前快照，不代替新修正版验证；最新119模块全套仍需自己的
修正版输入及真实终态。原完整目标active，无native/ABI/默认bin改动或云调用。

包装函数修正版11完整模块CPU14383真实exit0 chunk4da7e4，严格audit80600c：
689通过1166硬件跳过；1855身份/762源818资产匹配，无单项替换。最新119模块
已自行CPU/Metal完整预检10104身份一致（89552 exit0 chunk84ea7e），仅收集
不算执行通过。最新CPU97729已启动，原9417/118两套作业保持原源继续；
修正版完整Metal待本机GPU容量。整体目标active，未使用云或跨机器资源。

### 位置专用参数与验证进程恢复（2026-10-06）

上一轮修正11模块CPU真实exit0及严格审计证明689通过1166硬件跳过。
后续119模块CPU97729/Metal21887本轮核实工具句柄缺失、系统无对应进程且无完整
终态，原始部分XML/日志保留，不算全量成功。同一冻结764输入新建持久监督的
CPU/Metal执行，真实子进程退出由监督进程独立记录并纳入严格审计。
本轮另补固定与变长包装回调的位置专用参数 `/` 支持；保持参数顺序和包绑定，
随机方程、动态突触前向与VJP验证在独立12模块冻结版本执行，尚未验收。
最新CUDA缺实机证据，跨机器暂缓，完整目标保持active。

本轮位置专用参数新增模块首跑4个未包装SDE例失败，原因为测试幅度与独立参照
不匹配；原日志/XML保留在原阶段。修正测试幅度未改生产源码，独立冻结最终版本
完整50身份CPU真实exit0（34689/150ebc）：18通过32硬件跳过。逐身份XML、
763源/819资产及原生运行器哈希审计通过（d79fa7）。同一50模块Metal77487正在
本地实测；12模块完整回归及全目标验收仍未完成，不能由单模块结果代替。

新增50身份Metal77487真实exit0（918acf）：34通过16CUDA跳过。两模式严格
审计07e995验证全部身份、763源/819资产及原生/参考运行器哈希。位置专用参数
新增语义已有CPU和实际Metal证据；12模块完整回归、最新全量及全目标验收
仍未达成。旧失败12模块继续原冻结测试，不把其4失败改称通过。

### keyword-only默认值与原始Brian复制语义（2026-10-06）

细节见mpi-evidence/training-keyword-defaults-physical-20261006。首个SDE控制
在保留单位模式通过；明确丢弃单位的独立原反例89992/88ccd9失败，随后修复
Brian/core/functions.py复制默认值，并由训练前端从实际函数对象绑定不可变
默认标量。首次两套124模块2/4个warm失败因测试用了非微分v参数，原exit1
及XML保留。修正测试为显式零导数微分v，生产/原生源码未变；两套完整新增
模块真实exit0并通过逐身份/源码/运行器/监督退出审计。较早12模块原始幅度
版本已真实exit1：703通过4原测试幅度失败1198跳过；其余11模块exit0，不能
把该旧整套运行记为成功。完整目标保持active，最新CUDA及更广可变Python
语义尚未全部验收，跨主机仍按用户要求暂缓。

### 当前完整语料准备与完成证据归档（2026-10-06）

新增124身份的CPU/实际Metal完成证据已单独归档；真实归档进程73898/69167e
exit0，878成员全部输入/回读哈希一致。范围明确排除仍在运行的13/119模块，
保留原始单位丢弃失败、保留单位通过的控制以及warm模型失败。
当前所有121个训练模块已完整纳入新的持久冻结版本；764核心源加2离线依赖
共766输入/822资产，预期10278身份。原四个具体本地回归继续运行，未因准备
新完整语料被停止或重启。最新121模块CPU/Metal执行尚未启动；完整预检与
当前整套真实终态及严格审计仍必需，不能缩减目标为新增回调子集。

### 可变回调的调用点语义调查（2026-10-06）

原始Brian12项前向实测真实exit0（76032/8c7331），覆盖两种单位模式与
stateless False/True。saved=x; x*=.8; return saved在借用v时将v原地变更，
不可消去的v+.125临时数组只变更临时存储，标量字面量则重绑定x并保留saved
旧值。初轮1.*v被Brian优化为状态引用，不当成私有临时数组证据；原记录保留。
这些是独立原始前向证据，不是原生实现、VJP或GPU/MPI通过。当前转换器仍
明确拒绝参数修改及stateless=False；剩余实现必须保留调用点所有权、别名、
原表达式顺序和状态写回的反向规则，不能偷偷复制参数改变模型。
证据见mpi-evidence/training-mutable-callsite-investigation-20261006。本调查
位于冻结训练源码/测试目录之外，现有13/119作业与准备的121语料未被修改。

当前13模块CPU完整源码/逐项XML/真实监督退出审计041e2e成功；最新完整
121模块CPU84359已经启动，实际监督进程980fce确认存活，自有完整收集
仍在执行，尚无全121终态。较早119 CPU5737/Metal5738及当前13 Metal51492
继续原作业。新的可变调用点调查12前向检查不是原生可变回调或VJP的实现；
当前源码仍拒绝，剩余工作保持完整目标范围，不把此项降为仅文档限制。

### 可变回调状态效果的原生实验（2026-10-06）

隔离实验已将借用状态写回、数组别名、临时数组、标量重绑定与表达式求值
顺序转成现有原生SSA/Sequence，保留所有未使用算术的执行。公共前端尚未
接入，当前公共转换仍拒绝参数修改，不能宣称任意可变回调完成。原首个副本
假设失败由独立Brian确认顺序赋值返回数组可继续别名，原失败源/终态保留。
修正后的不可变45身份CPU19通过26GPU跳过、实际Metal32通过13CUDA跳过，
均真实exit0及严格审计2d61dd。覆盖固定normal随机流、初态/参数FD、full/TBPTT、
MPI2、实际GPU调度及未使用无效算术必须失败。现有766公共输入未变，正在运行
的121/119/13语料未被改写。证据见
mpi-evidence/training-callback-effects-prototype-20261006。后续必须接入真实Brian
状态/动作布局，并补充只读/捕获/链接/共享/跨神经元效果、语法/单位/元数据、
恢复和失败原子性验证；这些仍属于未完成项。完整目标保持active。
