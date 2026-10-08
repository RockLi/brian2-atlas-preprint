# 资格补充阶段只读核验摘要

阶段已终结：qualification-followup-r1.status=finite_qualification_queue_exited。E4、E2和R45的子阶段正常退出；Metal资格命令退出1，其失败被保留。所有阶段performance_run=false，complete_evaluation=false。进程退出或初始准入不能替代数值资格、完整尺寸训练、吞吐成绩或应用准确率。

本摘要只复核小型JSON、日志、既有检查结果与结果hash；未读取大型数组、重新执行模型/oracle或修改冻结源。没有开展全raw数值复算，因此这是元数据一致性核验，不冒充独立重新验证全部数值。

Atlas E4 CPU FP64：四个小Q0案例全部qualified，分别覆盖零状态/边界状态与固定动力学/共同训练共享隐藏参数。每个案例保存36次公开调用记录，其中三次为Adam训练；覆盖二值输入、显式v/a/m状态、CE surrogate VJP、共享原始z梯度、初始状态VJP和三次Adam更新。

| E4案例 | 通过检查 | 状态 |
|---|---:|---|
| E4-Q0-zero-fixed_dynamics | 41/41 | qualified |
| E4-Q0-boundary-fixed_dynamics | 41/41 | qualified |
| E4-Q0-zero-weights_plus_shared_hidden_tau_v_tau_a_k | 54/54 | qualified |
| E4-Q0-boundary-weights_plus_shared_hidden_tau_v_tau_a_k | 54/54 | qualified |

E4只证明这些冻结Q0案例的CPU FP64资格；不扩张为竞品E4资格、GPU资格或大模型性能。

E2卷积Q0：seed11、B2、T16、输入2×5×5，卷积通道2→3、读出4→2。五个真实视图各完成zero_initial与nonzero_threshold_boundary两变体，两个变体的全部检查均通过，包括三次Adam。Atlas用共享OIHW图参数bank和有界prefix replay观测状态；竞品用真实框架卷积与神经元/状态变换。

| E2视图 | 实际engine | 编译标志 | zero检查 | boundary检查 | 状态 |
|---|---|---|---:|---:|---|
| atlas | atlas | false | 97/97 | 97/97 | qualified |
| snntorch | snntorch_fp64 | false | 96/96 | 96/96 | qualified |
| spikingjelly | spikingjelly | true | 96/96 | 96/96 | qualified |
| spyx | spyx | false | 96/96 | 96/96 | qualified |
| brainstate | brainx_state | false | 96/96 | 96/96 | qualified |

这里compile标志是脚本参数：Spyx/Brainstate内部仍使用JAX JIT。Atlas的局部surrogate可观测性是CE诱导的全网络VJP，不宣称额外任意cotangent原生API。上述资格及冷调用耗时不是性能分数；JAX线程资源资格也不能由数值Q0推出。

E2 full初始准入：B16、T100、输入1×28×28、卷积16→32、读出128→10；五种子全部admitted_initial_only且native_executed=false。每个初始请求的原生tape估算280,836,352 bytes；只有初始零Adam状态请求已获准，不能推断后续Adam状态请求也获准。

| seed | 完整初始JSON bytes | 状态 |
|---|---:|---|
| 11 | 15,272,690 | admitted_initial_only |
| 23 | 15,272,748 | admitted_initial_only |
| 37 | 15,273,441 | admitted_initial_only |
| 51 | 15,272,318 | admitted_initial_only |
| 71 | 15,272,603 | admitted_initial_only |

H0R Metal两案例均未取得Atlas原生执行资格。失败发生在stage=metal_forward_backward的第一条gradients公开请求：软件准入已通过；构建阶段已返回并记录library_sha256，但b2-train报“cannot load native Metal training library”。没有成功返回backend/numeric_profile/gpu_dispatches证明，也没有Atlas梯度可用于数值比较。这是原生库加载/执行失败，不是观察到数值不一致；现有错误没有给出底层dlopen原因，不能进一步断言具体依赖缺失。

| H0R案例 | Atlas状态 | Torch MPS检查 | 整体资格 |
|---|---|---:|---|
| base | 首次gradients执行失败 | 22/22通过 | unqualified |
| initial_threshold_boundary | 首次gradients执行失败 | 9/9通过 | unqualified |

依赖盘点已找到Xcode15.4、macOS14.5 SDK及Metal/Foundation framework；这只证明盘点成功。Torch MPS实际检查通过，且fallback被设为0，但strict_same_numeric_profile=false；不能把MPS自身诊断通过称为Atlas/MPS相同数值配置等价。report.execution_status=completed表示诊断脚本已走完两案例；minimal_dense_checks_passed=false、boundary_execution_complete=false才是整体资格边界。大模型、性能、分布式更新和checkpoint restore仍未获此阶段资格。

R45初始准入已终态：9个case×5个固定种子，全部Atlas视图，35 admitted_initial_only、10 budget_rejected。35个获准案例没有运行native训练；10个拒绝均为rejected_step=1、passed_native_steps=0。以下byte范围均为五个种子的实际记录min–max，不把软件估算称为采样RSS。

| R case | 获准/5 | 请求JSON bytes范围 | 原生tape bytes范围 | 初始拒绝原因 |
|---|---:|---:|---:|---|
| R-N128-T512-p0.1 | 5/5 | 3,014,178–3,017,050 | 49,137,408–49,141,872 | 无 |
| R-N512-T512-p0.1 | 5/5 | 6,714,819–6,735,784 | 155,843,208–155,876,112 | 无 |
| R-N2048-T512-p0.1 | 5/5 | 36,446,318–36,528,514 | 603,901,728–604,023,768 | 无 |
| R-N512-T512-p0.01 | 5/5 | 5,646,803–5,651,095 | 154,159,776–154,166,544 | 无 |
| R-N512-T512-p1 | 5/5 | 17,852,694–17,853,468 | 172,850,688–172,850,688 | 无 |
| R-N512-T128-p0.1 | 5/5 | 5,129,667–5,150,632 | 44,661,384–44,694,288 | 无 |
| R-N512-T2048-p0.1 | 5/5 | 13,055,427–13,076,392 | 600,570,504–600,603,408 | 无 |
| R-N512-T8192-p0.1 | 0/5 | 38,417,859–38,438,824 | 2,379,479,688–2,379,512,592 | tape>1GiB |
| R-N8192-T512-p0.1 | 0/5 | 398,703,242–398,905,215 | 2,736,285,480–2,736,563,256 | JSON>64MiB且tape>1GiB |

N512/T8192的五种子仅tape超1 GiB；N8192/T512的五种子同时超过67,108,864-byte请求上限与1,073,741,824-byte tape上限。它们是初始软件预算拒绝，不是OOM或后续Adam拒绝。R45不是R225多引擎资格/性能完成；其他35案例后续Adam、数值资格与性能仍需各自正式阶段。

元数据交叉检查：外层七条终态与各子job终态相符、无记录残留owned进程；E2两阶段各5行及R45的45个唯一case/seed分母完整；所有result.json hash与终态引用一致；E4每项check和E2两变体check为passed；Metal真实error位置与总体资格标志一致；R拒绝原因与记录的JSON/tape上限算术一致。没有据此声称排除了所有瞬态主机干扰。

主要证据SHA256（路径相对于评估根目录）：

- evidence/qualification-followup-r1/terminal.json：8b6d38cf37de9527173e965a13cc8c4fe77525ae4746c1dcac8951fff25580df
- evidence/qualification-followup-r1/freeze.json：d2b59de4f3e7be9899f125ccb0913755a7a47dae30eb0f83dc8e85904e7384ee
- evidence/e4/runs/r1/summary.json：f3fc1dae88e8deb09fa2e6ce65cc086326f914885126e351f51672e94993759c
- evidence/e2-q0-v1/terminal.json：1fbaec555a0baa5024507c852537ba43fd0dbc6ea83c3b694551627f8851c865
- evidence/e2-admission-v1/terminal.json：0391e27e16fd1b955faa4e11012687eccd9c0f513bf54069da22f4a6c89b1138
- evidence/h0r/runs/q0-r1/report.json：642e843b9338efa2e0ab86e5fe2095c56dbf6ba61082ab13e44937ecc42e975d
- evidence/h0r/runs/q0-r1/base.json：3c4dc5e890e2fb881b4dc0f5c8d05d065d30dd5c66bb3331cec0f92b7668b066
- evidence/h0r/runs/q0-r1/initial_threshold_boundary.json：3bf320a1f7b0f9fc282a18762351d31c3fa911304604650fafb6d1ef38775f1d
- evidence/recurrent-admission-v1/terminal.json：6378b044355f422de4b969acba7e08ebdef5aff9419b22ebe639cd334082e49c

此次核验仅读取71份小型JSON，未输出大型数组；所有原始证据留在100.90.28.27。
