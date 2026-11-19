# 原始记录索引

这些文件由真实运行目录逐字节复制，JSONL无损gzip。完整校验表见上级ARTIFACT_MANIFEST.json；解压后的SHA单独记录。

- BC：各D/V3运行目录的config/run/history/metrics/summary。
- PPO：PPO_V3_mixed、anchor、legacy、legacy_extended；最后一组3072局，重复前缀单独核验。
- 主要最终评估：arena_final，13候选×500墙×5对手×4座位=130000局。
- 补充独立确认：arena_confirmation，2候选×新500墙×5对手×4座位=20000局。
- selection.json、supplementary_confirmation_plan.json分别在对应评估前锁定候选。
- confirmatory_statistics.json是三项原预注册比较；supplementary_confirmation.json是新的单项确认。
- *_smoke*、*_test.json、*_regression.log属于流程/单元测试证据，不是实力实验。特别是confirmatory_statistics_test.json使用已知常数效果的合成fixture验证统计计算，不能当作模型成绩。
- development_partial等文件是开发过程中保留的阶段记录，最终完整合并见development_complete.json。
- 初始缺失依赖的测试错误与一次stdin多进程启动错误均保留，恢复过程见execution_recovery.json。

没有归档原始专家语料、巨大特征缓存和所有未入选epoch权重；按复现说明获取或重新生成。
