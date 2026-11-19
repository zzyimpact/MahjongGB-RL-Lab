# 外部参考访问范围（2026-11-19）

本轮尝试通过网页工具访问用户提供的 LegendXct 仓库，但工具未返回可核验的页面正文；终端访问 raw.githubusercontent.com 和 api.github.com 也曾连接失败。因此，本轮未将上述项目的具体源码、架构或训练结果当作已经审查的证据，也未复制其代码。

用户提供的参考地址：

- https://github.com/LegendXct/Chinese-Standard-Mahjong-AI
- https://github.com/SuuTTT/IJCAI-mahjong
- https://github.com/PiggyQwQ/chinese-standard-mahjong-ai

V3 基于本仓库的已验证 D 模型独立实现：34 平面 + 分类 token + 候选弃牌向量。国标番种包含与特定花色有关的条件；本轮不启用未经逐状态算番验证的花色置换。性能结论只使用本地实际训练与对战数据。
