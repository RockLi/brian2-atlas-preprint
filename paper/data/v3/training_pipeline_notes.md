# Metal 管线缓存完整回归

原随机方程、动态突触完整目标保持 active。

当前完整53模块4375精确身份已从本次控制器自身收集并保存 planned-tests.json。
实际 exec session68504，初始 chunk244d57，运行中；尚未验收。
源码744输入、1654复制资产；隔离Rust50 runner；CPU、Metal、本地MPI2/8。
不运行云或NVIDIA作业，跨机器按用户要求暂缓。

当前独立快照：/atlas-storage/0002/b2-metal-pipeline-full-validation-sxfsoa5_
自有APFS临时目录：/private/tmp/b2-metal-pipeline-validation-tmp-8y9mzky1

先前四模块59851实际exit1 chunk2a20da：129通过、42明确CUDA硬件跳过、
1个Metal/MPI8训练前缀请求超时。原日志、XML、快照和终态全部保留。
测试给初始和恢复请求显式设置300秒后，原失败身份99649实际exit0
chunk8c79cc，1通过17.70秒；生产源码、数学容差和跳过规则未改。
不单独重跑四模块：本次完整53模块包含全部172身份。
restore-timeout-runtime-proof.json保存修正前后生产输入一致性。

更早缓存诊断67312实际exit1 chunk4ecec2，39通过1失败；Metal/MPI8
全weak/pathwise/all-initial差分通过，失败为超时fixture未就绪。
修正fixture后79040实际exit0 chunk0e1f8c，1通过38未选择。
所有历史失败保留，不能记作完整回归通过。

本次验收需要实际exec终态、完整XML、精确身份、源/资产/控制器/runtime哈希，
并通过audit_frozen.py。冻结回归之外的新语义开发须独立验证。
默认runner与Git状态不改动；CUDA最新实机与剩余语义仍未全部验收。

最终实际终态68504 exit0 chunk487309：3194p1181明确skip/4375，53完整模块，
约3小时20分。audit_frozen.py实际exit0 chunk5a1356：744源/1654资产、全部精确
身份/runtime/controller/原始Rust50与CUDA host语法来源一致。该范围仅代表原
Metal cache/weak/SDE版本，后续端点/外部前端另行冻结运行；原目标仍active。
