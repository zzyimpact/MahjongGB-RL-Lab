# 外部参考访问范围（2026-11-19）

本轮通过网页工具实际阅读了 LegendXct 项目的 FEATURE.md：其采用 102 个平面，包含最近弃牌、向听与有效进张，并明确提示花色置换对绿一色、推不倒等番种存在例外。本项目没有复制其代码或训练结果，采用独立实现的 34 平面 + 分类 token + 候选弃牌向量。为保证国标语义，本轮不启用未经逐状态算番验证的花色置换。

来源：https://github.com/LegendXct/Chinese-Standard-Mahjong-AI/blob/main/FEATURE.md

PiggyQwQ 仓库首页可以访问，但 docs 目录和 LegendXct 的 src 目录在网页工具中获取失败，未声称已审核其完整源码。SuuTTT/IJCAI-mahjong 的公开地址返回 404；这不证明私有仓库不存在。终端访问 raw.githubusercontent.com 曾连接失败。参考设计只作背景，本轮性能结论全部来自本地实验。
