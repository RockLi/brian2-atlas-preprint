# constant-over-dt 随机子表达式

动态训练前端接入标准 Brian `SubexpressionUpdater`。神经元和突触中标记
`constant over dt` 的子表达式保存为可携带状态，由原有更新器的 slot、order、
name 决定执行顺序。转换不会把缓存表达式在每次引用处展开。

```python
group = NeuronGroup(
    2,
    '''dv/dt=(-v+drive)/ms : 1
       u=rand() : 1 (constant over dt)
       drive=gain*u : 1 (constant over dt)
       gain : 1 (constant)''',
    threshold='v > .6 + .02*u', reset='v = .1*u',
    method='rk2', dt=.2*ms,
)
```

通过 `lower_brian_dynamic_training` 或 `lower_brian_training(dynamic=True)` 转换。
静态 v3/v4 前端仍拒绝这些额外更新器，避免忽略缓存行为。

## 调度与求导

- 缓存更新器沿用 Brian 的标量／向量语句分离。共享神经元缓存先计算一次，
  再写回物理共享值，向量段保留其局部引用。普通缓存按神经元或突触逐项计算。
- 标准默认时机为 `before_start`。改变更新器的 `when` 时，动作按实际网络调度
  执行，例如 `after_groups` 读取积分后的状态，`end` 读取 reset 后的状态。
- 缓存依赖链可以包含浮点、Boolean、时间依赖和已支持的函数。缓存供方程、
  阈值、reset、pre/post 以及 refractory 表达式读取。RK 各 stage 复用本步缓存。
- 每个随机调用点按 seed、sequence、batch、updater domain、entity、tick、stream
  定址；共享段与向量段使用不同的 stream 区间。每个更新器最多 16 个调用点。
  `regular_runner_layout[updater.name]['draws']` 记录缓存更新器的抽样布局。
- 缓存更新是可微动作：固定抽样路径上的状态及可训练幅度参与 full BPTT／TBPTT。
  Boolean／整数控制继续停止梯度。缓存值随 carry、store/restore 保存。
- 延迟事件读取投递当步的缓存值。缓存中的 rand/randn 与事件路径中的直接随机
  调用具有不同语义，后者仍使用已有的稳定发射事件身份。
- 零连接突触不执行逐边缓存抽样。其余结构 mask 与缓存重建使用已有的动态状态
  机制；更新器的注册不依赖存在 spike 或非零连接。

本轮保留原生二进制及 GPU ABI：v4r7／v5r7，不需要新增原生指令。

## Brian 约束与剩余范围

仅接入实际所属对象的标准 `SubexpressionUpdater`，保留模板、活动状态、时钟、
单位、数据类型、预算和函数身份检查。同步更新器须使用网络 timestep。
Brian 本身禁止突触子表达式同时使用 `shared` 和 `constant over dt`；前端不放宽
这个约束。神经元共享缓存可用。

外部输入群体由调用方提供 spike；它自身的标准缓存更新器与积分器一样不进入训练图。
此时没有被训练网络使用的输入缓存不会占用噪声流或改变转换结果。突触若读取输入
群体的可变缓存，仍显式拒绝，避免把一个运行时变量冻结成快照常量。

缓存驱动的布尔／时长 refractory 与直接在 refractory 表达式中调用 rand/randn
不同；后者现已接入，见 `NATIVE_TRAINING_RANDOM_REFRACTORY.md`。
异步连续积分、外部 Python 回调、int64／unsigned／位运算、
Synapses-to-Synapses 端点及新版 NVIDIA 实测仍有缺口。跨机器／多物理 GPU 验证
继续按用户要求暂缓。本轮没有云调用。

## 验证

`test_training_cached_noise.py` 用独立逐步公式、实际 Cython 注入样本与梯度有限
差分检查共享缓存、依赖链、Boolean、时间项、RK、冷暖快照、refractory、可训练
突触初值和 pre/post 复用；覆盖 carry、检查点及 CPU／Metal 本机 2/8 MPI。
`test_training_cached_noise_edges.py` 检查延迟／暖队列读取当步缓存、零连接、
更新位置、缓存驱动 refractory 与非法更新器拒绝。

完整正式回归已通过：344 passed／132 NVIDIA-only skipped／0 failed；另有 16 项独立检查通过。
源码、终态与逐项记录入口：`mpi-evidence/training-cached-refractory-final-20261004/README.md`。
其中本页两个新模块为 49 passed／22 NVIDIA-only skipped；不累计 pilot 或失败运行计数。


## Distinct cached updater clocks (2026-10-05)

The v5 itinerary admits distinct standard SubexpressionUpdater clocks. Actual
Cython and independent physical equations now verify Poisson caches at .4ms /
.2ms or .1ms with .2ms neuron clocks, including threshold/reset consumption,
idle visits after the last main transition, full/split checkpoints and local
CPU/Metal MPI. The original mixed uniform/normal RK2 fixture also agrees at
its exact modified .4ms updater clock. The earlier negative clock test was
obsolete and is replaced by positive execution evidence; inactive, overridden
and static-runner rejection controls remain intact. See
`mpi-evidence/training-poisson-event-replay-20261005/README.md` for exact frozen
sources, the preserved initial failure, complete reruns and acceptance limits.

## Random draws inside captured callbacks

The dynamic array/vectorised event frontend now authenticates standard
`np.random.randn(len(x))` and `np.random.rand(len(x))` when `x` belongs to the
selected argument domain. The body keeps its original caller statement scope,
argument aliases, capture writes and RNG site order. Native emission-addressed
caches retain the actual stream and arrival identity; subsequent whole capture
updates consume selected arrays. Normal/Uniform/Poisson mixed-body development
checks preserve stream 0/1/2, including Normal in the Poisson rate and restore.

All bank and continuous initial-state full/TBPTT VJPs and post-restore domain
rollback have independent development checks. Affine Normal/Uniform APIs with
selected `len(array)` sizes are implemented below. Other explicit sizes/import
spellings and scalar/whole callback draw domains remain unfinished.
The 56-module full regression is pending; see
[the frozen evidence](mpi-evidence/training-callback-body-noise-20261009/README.md).


2026-10-09 仿射 callback 采样前端

标准 `np.random.normal(loc, scale, size=len(x))` 与 `np.random.uniform(low, high, size=len(x))` 支持 selected array/vectorised 回调中的捕获修改。位置参数及关键字参数按 Python 实际表达式次序求值，缺省 loc/low=0、scale/high=1。采样使用现有原生 Normal/Uniform counter 缓存，再进行可微仿射变换；scale/low/high 以及前后的参数和捕获写入保留 pathwise 导数。Normal 的负尺度及 Uniform 的范围溢出回滚所有训练／时钟／缓存状态。零尺度、零宽度均有效。当前 NumPy 2.5.2 legacy uniform 接受反向区间；[官方文档](https://numpy.org/doc/2.2/reference/random/generated/numpy.random.uniform.html) 将该条件标为未定义，兼容结论限于当前实测版本。

显式尺寸目前要求标准 `len(array)` 且属于实际 selected 域，任意固定／tuple尺寸、scalar／whole随机域与完整组合尚未全部覆盖。冻结证据位于 `mpi-evidence/training-callback-affine-noise-20261009/`，整体目标仍 active。


2026-10-09 隐式 callback 采样尺寸

标准 `np.random.normal(x, spread)`、`np.random.uniform(x, x+spread, size=None)`、`normal(scale=spread)` 与 `uniform(high=spread, size=None)` 已接入 selected array/vectorised 回调。尺寸来自两参数的实际广播形状，保留实参求值次序和缺省首参数，并使用已有 emission-addressed 独立随机流。到达数2→1的实际NumPy恢复对照、全部银行/连续初态full/TBPTT与零宽度梯度、负尺度atomic回滚及独立Python执行均有本轮实测。

全部标量／非singleton whole参数的单次或whole域抽样尚未接入，不会把它们假装成selected逐边抽样。更多尺寸／导入拼写、0-D/singleton组合、mutable-delay lanes及完整GPU组合仍待开发或验收。完整60模块174预检运行中，证据入口为 `mpi-evidence/training-callback-inferred-noise-20261009/README.md`，整体目标active。


2026-10-09 captured callback 标量采样

`np.random.normal(loc_scalar, scale_scalar)`／`uniform(low_scalar, high_scalar, size=None)` 与直接 `np.random.randn()`／`rand()` 在有捕获数组的回调内共享一次scalar样本。counter选择本次raw arrival首事件，refractory过滤为0/1行仍遵循原始NumPy callback实际调用；single physical column保存跨语句scalar，不取inactive行旧值。标量augmented赋值为重绑定，旧alias保留旧值。共享可训练mean/scale、全部银行及连续初态full/TBPTT、边界与完整恢复/atomic回滚有本轮实证，原生runner/ABI不变。

当前证据来自captured callbacks；无捕获/direct-return等scalar形式仍待补齐，whole RNG数组/其他尺寸/import和mutable-delay等剩余范围不变。62完整模块204预检进行中，证据见 `mpi-evidence/training-callback-scalar-noise-20261009/README.md`，整体目标active。

标量callback v2验证更新：首204同版预检发现旧masked Poisson array捕获写入回归（5pass/1fail、SHA完整），已保存失败版；scalar元数据排除仅对scalar_normal_sites生效，旧数组compaction恢复。修复CPU/Metal四项实际通过，独立v2冻结的204预检仍运行，不能称完整接受。证据入口 `mpi-evidence/training-callback-scalar-noise-v2-20261009/README.md`。无capture/direct-return scalar、标量Poisson、whole RNG等仍未全部接入。


2026-10-09 无捕获/直接返回的标准随机回调

无capture的事件callback，以及没有`+=`/`*=`的纯随机函数、RNG直接位于return表达式/直接返回scalar，都可通过已认证Normal/Uniform random_sites发现并进入caller lifting。隐式尺寸的return draw先保存；scalar physical storage不要求外部捕获数组。namespace常量使用已解析的路径变量作为输入，不冻结未知源。四种形式的原始恢复、全部银行/初态full-TBPTT、边界与回滚/过滤和独立调用顺序有45项本轮检查（含17旧路径）；原生ABI/runner保持。

64模块232预检仍执行，证据 `mpi-evidence/training-callback-uncaptured-noise-20261009/README.md`；scalar v2的204冻结接受保留。非event callback RNG、scalar Poisson、whole随机数组、其他尺寸/import和动态delay lane等仍待完成，原始goal active。

标准 callback RNG 别名及Poisson keyword更新：见 mpi-evidence/training-callback-rng-alias-20261009/README.md。标准模块/direct/closure identity认证，无需按拼写放宽loaded-body检查；scalar Poisson及其他明确列出的域仍未接通。

Scalar Poisson callback：每次array/vectorised callback使用首个实际raw FIFO arrival的一个count，lazy inactive-lane guard保留真实event identity，integer scalar private storage维持immutable alias与broadcast。见mpi-evidence/training-callback-scalar-poisson-20261009/README.md；非event及scalar FIFO callback上下文、whole RNG与更广组合尚未完成。

Scalar FIFO随机callback：每个arrival使用真实event stream，可pure/whole capture；refractory Python if覆盖所有提升后的body和sampler。见mpi-evidence/training-scalar-event-rng-20261009/README.md。Whole RNG/任意sizes与非event callback上下文仍待完成。

Scheduled scalar callback RNG支持 Normal/Uniform/Poisson 与混合流，pure/full mutable captures，每scalar时钟块按Python顺序采样。见mpi-evidence/training-regular-scalar-rng-20261009/README.md；vector regular/integrator/threshold/reset callback RNG和whole array尺寸仍待完成。

Scheduled array/vectorised callback RNG已覆盖选定数组的Normal/Uniform/Poisson及mixed，pure/mutable captures；scalar块和逐语句offset避免与普通randn复用stream。见mpi-evidence/training-regular-vector-rng-20261009/README.md。vector callback内部scalar once-per-block采样与whole/任意sizes、integrator/threshold/reset RNG等尚待完成。

定时向量callback内标量Normal/Uniform/Poisson已接通，每实际NumPy整块采样一次再广播；可与数组Poisson混合，率前缀按顺序执行，原始count在continuation复用。共同输入快照和显式写回优先级保留NumPy数组/捕获语义。见mpi-evidence/training-regular-shared-rng-20261009/README.md：124增量开发测试通过，真实MPI2/Metal1198调度，913/969冻结484预检运行中。状态相关混合梯度、链式零边界及更广尺寸/上下文仍待补齐，目标active。

混合Poisson梯度补充：状态相关数组rate的全参数/所有连续初态likelihood梯度及上游零边界引起下游重采样的反事实已通过六项独立CPU/Metal MPI2 full/TBPTT对照。新增测试模块尚未纳入82冻结484预检；证据training-regular-shared-rng-20261009/hybrid-supplemental.json。目标仍active。

定时scalar NumPy fallback按实际每行entity执行随机callback，Normal/Uniform/Poisson/mixed与ordinary randn按表达式顺序占不同stream；whole captures先于显式标量索引写回。共享外层eager Sequence森林保留未使用算术检查和分支惰性。915/971最终27身份严格接受，MPI2/Metal646实际dispatch；84模块501组合预检等待旧484终态，尚未开始或接受。证据mpi-evidence/training-regular-scalar-fallback-rng-20261009/README.md；完整目标active。

突触定时scalar fallback随机回调接通符号ownership规划；完整closure列数独立于行数。每行entity/stream、重复目标先后顺序、全捕获数组写入已由Brian及独立full/TBPTT所有银行/初态验证。w剪枝使inactive非法Poisson率跳过，checkpoint再生一行只新增该行真实clock记录，历史count保持。916/972最终20严格接受，MPI2/Metal720 GPU dispatch，85模块517组合预检等待501终态。证据mpi-evidence/training-synaptic-scalar-fallback-rng-20261009/README.md；完整目标active。

生成积分阶段的随机callback支持选定数组Normal/Uniform/Poisson/mixed，每个真实阶段调用有独立stream，命名xi身份保留；Milstein真实两次callback。神经元/连续突触原始Brian与全bank/全部连续初态full/TBPTT及Poisson score已验证。过预算的mixedRK4共享无条件依赖前缀根，保留全部SSA节点与条件/Poisson rate惰性；ABI不变。917/973最终27严格接受，MPI2/Metal304 GPU dispatch；88模块544组合预检等待517终态，目标active。证据mpi-evidence/training-integrator-callback-rng-20261009/README.md。


2026-10-09：重置选定数组随机回调

认证 callback 的 Normal/Uniform/Poisson/mixed 按 body 调用顺序分配流，流号接续其他已分配调用并写入 neuron_draws/reset。动态重置动作沿用 layer domain 和物理 neuron entity，事件选择不重编号。重置局部副本与整组闭包捕获的执行/写回保持独立。CPU/Metal MPI2 的原始 Brian、恢复、full/TBPTT、参数/初态与正率 score、非法域回滚验证通过；空重置域检查的旧回归通过。此版本完整93模块尚未执行，整块标量/whole/更多size重置回调与其他随机入口仍需补齐。详细证据：mpi-evidence/training-reset-callback-rng-20261009/README.md。


2026-10-09：阈值选定数组随机回调

阈值回调接入 Normal/Uniform/Poisson/mixed，neuron_draws/threshold 保存独立流记录，绑定真实物理神经元 entity。Compare 递归提升及多重 eager Sequence 布尔标记传递保持原生 predicate 验证与 and/or 梯度。三阶段分配时 threshold 先分配、update/reset 后分配，实际执行先 update，再 threshold，再 reset；测试按独立 stream 对照真实 Brian。66项冻结CPU/Metal通过、MPI2/Metal368 dispatch，源码919/不可变资产970；完整96尚未执行，整块标量/whole/更多size随机语义仍需补齐。证据 mpi-evidence/training-threshold-callback-rng-20261009/README.md。


2026-10-09：阈值 once-per-block scalar draws

scalar Normal/Uniform raw draw写入共享float缓存，Poisson scalar count写入共享detached int32缓存；每个神经元读取相同物理缓存而不重复sample/score。Poisson rate依赖SSA子图在共享action计算，前面的shared raw/count依赖替换为缓存读取；原有采样顺序保留，score及零率整组CF只作用一次。921/972冻结66 CPU/Metal验证严格通过，含Normal→Poisson rate链式mean/spread score与非零整组CF、恢复atomic，MPI2/Metal432 dispatch。完整97尚未执行，积分/reset及其他whole/size仍需补齐。证据mpi-evidence/training-threshold-shared-rng-20261009/README.md。


2026-10-09：积分阶段 once-per-block 标量随机回调

神经元与连续突触生成积分阶段支持共享Normal/Uniform raw与detached int32 Poisson count，按实际阶段顺序采样一次并供所有行读取；rate依赖子图及Normal→Poisson链式score保留，零率反事实作用整个调用块。原始Brian/恢复、Euler/RK2/RK4与xi的Heun/Milstein组合、full/TBPTT全银行/连续初态、正率score/非零神经元和突触整组零率CF、非法域atomic已验证。重构导致阈值辅助函数局部导入错误，被旧回归发现，修复后重新冻结并全部重跑。922/973最终82不同身份严格接受（70新8旧+4突触零率补充），真实wait0/collection/XML顺序/请求集合/SHA完整，MPI2/Metal720dispatch，runner/ABI未变。98完整模块collect14063，旧13958保留；完整98未执行/接受。既有85完整CPU与88等待队列保留。reset每整块标量、更多whole/size/跨列和混合尺寸梯度、动态delay、更广Brian语义/最新CUDA/完整组合待补齐；目标active，无云调用，跨主机后置。证据 mpi-evidence/training-integrator-shared-rng-20261009/README.md。


2026-10-09：重置阶段 once-per-block 标量随机回调

Normal/Uniform raw与detached int32 Poisson count在重置事件行写回前采样一次，所有事件行共享缓存；空选择仍调用、校验并更新whole capture。绑定capture/selected/scalar eager三类程序，保持副本、别名与显式写回顺序；Normal→Poisson链式率/score保留。候选与scalar eager率DAG不同SSA排列按表达式结构核对，不同参数率/向前随机依赖仍拒绝。原始Brian/SpikeMonitor、pure/mutable、empty/subset/all、显式randn、capture v别名、refractory、carry/restore、full/TBPTT全银行/连续初态、正率score与非零整组零率CF、非法域atomic通过。首次SSA误拒绝及零率fixture未跨边界的失败保留，修复后923/974冻结70不同身份严格接受（58新12旧），真实wait0/请求集合/collection/XML/SHA完整，MPI2/Metal1168dispatch，runner/ABI未改。99完整模块collect14150，旧14063保留；完整99未执行/接受，既有85完整CPU与88等待队列保留。更多whole/size/跨列/混合尺寸梯度、动态delay、更广Brian/最新CUDA/完整目标组合仍待补齐；目标active，无云调用，跨主机后置。证据 mpi-evidence/training-reset-shared-rng-20261009/README.md。


2026-10-09：共享与逐行混合尺寸随机回调的独立梯度验收

共享Normal与逐行Poisson、共享Normal进入arrayPoisson rate链式score，在threshold/reset/neuron integrator/continuous synapse integrator加入独立原始Python参考；pure/mutable、Euler/RK2、真实stream/domain/physical row、原始Brian/SpikeMonitor/carry/restore、full/TBPTT所有参数银行/连续初态、四种位置非零逐行零率CF、非法第二行率及共享spread恢复回滚已验证。924/975冻结最终88不同身份严格接受（80新8旧），actualwait0/请求集合/collection/XML顺序/SHA完整，MPI2/Metal656dispatch。仅新增完整测试模块，无生产/Rust/shader/runner/ABI改动。首轮synaptic a布局假设及比较列表修改provenance的测试失败保留并修复。100完整模块collect14270，旧14150保留；完整100未执行/接受，既有85 CPU/88等待队列保留。额外原始Brian探测具体复现4列capture供2行neuron调用的Normal/Uniform宽度拒绝，以及Poisson(size=len(captured))认证拒绝，作为下一阶段真实未完成项；探测未启动原生/GPU。Whole/显式size/跨列、动态delay、更广Brian/最新CUDA/完整组合仍待补齐，goal active，无云调用，cross-host后置。证据 mpi-evidence/training-mixed-shape-rng-20261009/README.md。


2026-10-09：显式Poisson长度与整列定时随机回调

支持Poisson(size=len(array))率广播，scalar率生成每输出行独立counts，空输出仍校验scalar率而不读空array率。保留keyword顺序与共享Normal→arrayPoisson链式score。Whole regular4列capture供2行caller保留四列raw/count、整数/负索引/越界校验及全部capture写回；ownership探测正确识别constant capture真实修改，编译仍要求整数索引。926/977最终202不同身份严格接受（190新12旧），实际wait0/collection/XML顺序/请求集合/SHA完整，MPI2/Metal736dispatch。覆盖原始Brian/恢复、Euler/RK2、period1/2、full/TBPTT全银行/连续初态、state-dependent Poisson初态score与非零零率CF、empty/非法率/越界回滚。仅Python4文件+2完整测试模块，无Rust/shader/runner/ABI变化。102模块collect14555，旧14270保留；完整102未执行/接受，85运行/88等待队列保留。Neuron阈值/积分/reset不同宽度capture绑定仍待实现，literal/更多size、boolean/slice选择、dynamic delay、最新CUDA/完整目标组合未齐；goal active，无云调用，cross-host后置。证据 mpi-evidence/training-explicit-size-rng-20261009/README.md。


2026-10-09：神经元整列捕获有序阈值

4 列 capture 供 2 行 neuron 的 Normal/Uniform/Poisson/state-rate Poisson 阈值保留四列抽样与真实物理状态；整数 gather 后生成两行边际，整块快照写回，再按 neuron 绑定原生边际归属。密集骨架仅用于分配与单位校验，真实阈值完整替换；有序浮点边际保留 surrogate，integer/Boolean/refractory 更广整列绑定仍待补齐。928/979 冻结最终 76 不同身份严格接受（64 新、12 旧），实际 wait0/collection/XML 顺序/请求集合/工作区源码与冻结资产 SHA 一致；MPI2/Metal 816 dispatch，runner/ABI 未改。覆盖原始 Brian/SpikeMonitor、carry/store/restore、四种比较方向、full/TBPTT 全参数与全部连续初态、正率 score、非零零率 CF、越界/非法域 atomic。103 模块 collect14675，旧14555 全保留；完整103未执行/接受，旧85运行与88等待保留。整列 neuron update/reset、更多 size/选择/谓词/refractory/随机上下文、dynamic delay 剩余范围、最新 CUDA 与当前完整组合仍未齐；goal active，本轮无云调用，cross-host 后置。证据 mpi-evidence/training-neuron-whole-rng-20261009/README.md。


2026-10-09：神经元整列捕获 Euler/RK2/RK4 更新

实际生成的 NumPy 积分按语句保存 scalar/array 阶段缓冲，避免整个 RK2/RK4 DAG 超过128节点；完整4列 capture 与2行 caller保持独立采样及递增stream。vector-block入口复制读写状态，closure仍绑定真实物理存储，所有阶段后按原写顺序显式写回。完整切片捕获规范化支持膜电位alias，caller copy避免被物理closure污染。929/980最终冻结210不同身份严格接受（126新+72前轮阈值/Euler update+12旧），pytest48839/collection48789实际wait0，root1446 actualexit0 fcfff1，审计0 8bf4bf；请求/collection/XML顺序/工作区与资产SHA完整，无skip/fail/error。覆盖四种随机函数、原始Brian/恢复、full/TBPTT所有参数与所有连续初态、正率score、非零零率CF、越界/非法域atomic、捕获膜电位forward/VJP。MPI2/Metal1392dispatch，runner/ABI未改。104完整模块collect14864，旧14675保留；完整104未执行/接受，85运行/88等待队列保留。原始探测确认whole xi/refractory update/threshold/empty reset各仍调用4次4列callback，当前仍拒绝；另实证局部标量full slice错误接受，typed候选仅standalone修复尚未合入。上述与更多size/选择/谓词/动态delay等原始范围、最新CUDA和当前完整组合仍未齐；goal active，无云调用，跨主机后置。证据 mpi-evidence/training-neuron-whole-integrator-rng-20261009/README.md。


2026-10-09：完整切片类型修复与整列命名 SDE

实际一维数组 full slice 通过 intrinsic 验证，普通/随机提升保留类型校验，scalar/zero-dim/readonly 与原始 Python 对照，caller local alias 和原子回滚通过。整列 Euler/Heun/Milstein 的 xi 按名称保留 stream，实际每行标准噪声和生成阶段系数存入动态缓存，callback 后续流。CPU基础36、状态相关扩散24、多状态共享xi_a/独立xi_b12、非零零率与atomic22均通过；typed首批17+补8通过。原始声明顺序须取实际生成代码并映射噪声名，不能按set顺序冒充stream。旧完整85测试wait0（4150pass/7684skip）但审计1因8个非硬件pure mixed跳过，未接受、Metal未启动；当前补齐真实pure_mixed并取消skip，20 CPU对照通过，不放宽审计。full88已实际接续sequence89642/supervisor89645，ps确认live，保留。931/982冻结106模块collect15207，旧14864全保留；真实MPI2/Metal状态相关Milstein+state-ratePoisson作业wait0/1360dispatch。311同版增量根session58196仍运行，未收到终态/未严格接受；首个harness选择器断言失败且pytest未启动，失败保留，仅harness修正后使用同一资产启动。完整106未执行/接受，无新完整队列/云调用。whole refractory/reset、更多尺寸/选择/动态delay等原始范围、最新CUDA及当前完整组合仍待补齐，goal active，cross-host后置。证据 mpi-evidence/training-neuron-whole-xi-rng-20261009/README.md。
