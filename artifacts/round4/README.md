# 第四轮可提交候选与模型归档

推荐 **V3 BC，seed20261119，第1epoch**。这是完成12epoch充分性研究后保留的较早泛化快照，属于纯BC权重。PPO备选确实经过本轮RL更新，不能把两者混称。长程3072局PPO另用全新500墙确认，仍略跨零，因此未替换主推荐。最终对战与不确定性见 [最终报告](../../reports/round4/FINAL_REPORT.md)。

## 只允许上传一个代码文件时

1. 打开下面表格中的单个 `mahjong_bot.py`，作为Bot代码上传。无需上传多个源码文件，也无需把ZIP内容逐个上传。
2. 将同一目录的 `mahjong_model.pt` **单独**上传到用户存储，文件名保持 `mahjong_model.pt`。代码默认读取 `data/mahjong_model.pt`。
3. 选择 Chinese-Standard-Mahjong（136张无花牌）及简单交互。
4. 同一代码绑定同一权重SHA。切换BC/PPO/D时必须同时换配套代码和权重；不能复用错误版本。各版本默认使用相同存储文件名，比较多个Bot时应安排各自匹配的存储路径。

| 版本 | 用途 | 单个代码文件 | 独立权重 | 备用ZIP |
|---|---|---|---|---|
| BC | 推荐；V3纯BC | [mahjong_bot.py](releases/BC/mahjong_bot.py) | [mahjong_model.pt](releases/BC/mahjong_model.pt) | [mahjong_bot.zip](releases/BC/mahjong_bot.zip) |
| PPO | 课程如要求提交RL更新权重时的备选；iteration48/768训练局 | [mahjong_bot.py](releases/PPO/mahjong_bot.py) | [mahjong_model.pt](releases/PPO/mahjong_model.pt) | [mahjong_bot.zip](releases/PPO/mahjong_bot.zip) |
| PPO_long | 完成3072局的RL备选；独立确认趋势为正但未达采用门槛 | [mahjong_bot.py](releases/PPO_long/mahjong_bot.py) | [mahjong_model.pt](releases/PPO_long/mahjong_model.pt) | [mahjong_bot.zip](releases/PPO_long/mahjong_bot.zip) |
| BC_seed3 | 第一批最终池点估计最高的BC seed；探索性备选 | [mahjong_bot.py](releases/BC_seed3/mahjong_bot.py) | [mahjong_model.pt](releases/BC_seed3/mahjong_model.pt) | [mahjong_bot.zip](releases/BC_seed3/mahjong_bot.zip) |
| D_fallback | 较快的D第6epoch；平台缺少V3原生向听API时的兼容备选 | [mahjong_bot.py](releases/D_fallback/mahjong_bot.py) | [mahjong_model.pt](releases/D_fallback/mahjong_model.pt) | [mahjong_bot.zip](releases/D_fallback/mahjong_bot.zip) |

V3本机测试依赖Python3.10.22、PyTorch2.3.1、NumPy1.26.4、PyMahjongGB1.4.0；不在代码中执行pip安装。Python3.6只通过语法检查。**Botzone线上依赖、内存/时间额度和账号上传尚未验证**。若平台报缺少五类向听API，D_fallback不使用这些V3候选特征；不要静默删特征或换权重。

## 文件身份与预算

| 版本 | 模型bytes | ZIP bytes | 解压bytes | 单py bytes | 部署权重SHA-256 |
|---|---:|---:|---:|---:|---|
| BC | 4805859 | 12320 | 35174 | 38181 | `e5c679163a80d02b691c74effaaa9987d9b92b526d079794cfbb14cab6de8ab6` |
| PPO | 4805859 | 12319 | 35174 | 38181 | `0e8212c1dbb9b893bdc38542e7cdef4c71272059a4fa897ee0fbeeb8432d5306` |
| D_fallback | 3870059 | 12283 | 35104 | 38111 | `7e6ce89acecae536f148b9e3b25cfbb1fecd742e9cc8415dd83efd12c3ecdea2` |
| PPO_long | 4805859 | 12319 | 35174 | 38181 | `271bee1f42d6f46832a8dbe2e104e7debd8f901f8be6ed5036418cdb758c9ec9` |
| BC_seed3 | 4805859 | 12319 | 35174 | 38181 | `3b01bcd35d40d3c900c0051a4140959c9ed71c1b65e79c441fd64bafb5bd73b5` |

模型额度≤268435456bytes；ZIP、解压代码/资源合计及单py各项<4000000bytes。每个release目录都保存完整清单、SHA和本地运行检查。模型没有嵌入代码。

## 继续训练与复现

上述部署权重只含推理参数。各入选模型的 `checkpoint.pt` 由原始运行逐字节归档，保留训练配置和优化器状态，具体来源/大小/SHA见 [ARTIFACT_MANIFEST.json](../../reports/round4/ARTIFACT_MANIFEST.json)。历史C/D仍指向round3原文件。

本机完整未入选快照位于 `runs/final_v3/`；原始数据和缓存位于 `data/`，不进入Git。获取数据、生成缓存、18组BC、PPO、评估、导出和核验命令见 [REPRODUCIBILITY.md](../../reports/round4/REPRODUCIBILITY.md)。

```bash
.venv/bin/python -m mahjong.verify_submission artifacts/round4/releases/BC
.venv/bin/python -m mahjong.verify_round4
```

不要把带 `policy/config/optimizer` 的训练checkpoint直接当作上传用的 `mahjong_model.pt`。
