# 国标麻将 RL 大作业 —— 第一轮本地开发与训练验证报告

日期：2026-11-18 · 机器：Apple M5 Pro（15 核 CPU / 16 核 GPU / 24GB 内存 / macOS 26.5.2）

## 1. 实现方案与理由

**选定路线：复用课程示例框架 + 新增 BC 管线。**

- 复用 `mahjong/gamecore/`（136 张无花复式裁判，8 番起胡）、`observation.py`（SeatView 私有状态重建 + 235 维合法动作 mask）、`botzone/`（简单交互协议、CHI/PENG 复合响应、冷启动历史重放）、`rl_framework/`（多进程同步 PPO，含参数更新校验）。
- 算番用 `PyMahjongGB 1.4.0`（源码本地编译成功），与 Botzone 平台预装库同源，不重写规则/算番。
- 新增：`teacher.py`（向听数启发式教师）、`bc_data.py`（教师自对局数据）、`replay.py`（IJCAI-2026 真实对局回放提取）、`train_bc.py`（mask 交叉熵 BC，checkpoint 与 PPO resume 格式兼容）、`tests/test_export.py`（本地部署测试）；扩展 `evaluate.py`（teacher/random 对手与策略对照）。
- 理由：示例的规则、协议、PPO 已经过测试且质量可靠，重写风险高收益低；课程要求的"可解释改进"由 **BC 预训练 + BC 初始化 PPO** 满足（原示例从随机权重开始 RL）。特征增强列为第二轮改进。
- 环境：`~/.local/bin/python3.10`（3.10.21）+ `.venv`；依赖见 `requirements-lock.txt`（torch 2.3.1 / tensordict 0.1.0 / numpy 1.26.4 / PyMahjongGB 1.4.0）。pip 走清华镜像（PyPI 直连慢，huggingface.co 被墙需 hf-mirror.com）。

## 2. 完成情况与验证证据

| 模块 | 状态 | 证据 |
| --- | --- | --- |
| 规则/协议一致性 | ✅ | `mahjong/tests/test_protocol.py` 2 项通过（40 局完整对局，Session 与 SeatView 逐事件比对手牌/副露/牌墙/特征） |
| RL 框架核心 | ✅ | `rl_framework/tests` 31 通过 1 跳过；4 个失败均为可选 `fdl` 包缺失（与麻将无关） |
| 完整对局 + 合法动作 | ✅ | env.py 对非法动作直接抛错；教师 1000 局、PPO 2400+ 局、评估 700 局均无非法动作 |
| BC 数据管线（教师） | ✅ | 1000 局 251,700 样本，87% 有胡牌，29.6s（8 进程） |
| BC 数据管线（真实回放） | ✅ | IJCAI-2026 前 3000 局 → 563,257 样本，**0 局被过滤**（每条响应都被 Session 精确重建且标签合法） |
| BC 训练 | ✅ | 8 epoch × 716 更新 = 5728 次梯度更新，loss/acc 真实记录，checkpoint 保存/加载验证 |
| BC→RL 衔接 | ✅ | `train.py --resume runs/bc_v1/checkpoint.pt` 直接续训 |
| PPO 训练 | ✅ | 300 迭代 × 8 actor，10,401 次梯度更新，`parameters_changed=true` 逐迭代校验 |
| 评估 | ✅ | 7 组 × 100 局，固定种子 + 轮换座位 + 同墙配对比较 |
| Botzone 导出与协议 | ✅ | `tests/test_export.py` 3 项通过：py3.6 语法、真实子进程 stdin/stdout 对局、冷启动重放一致性 |
| 特征增强 / 大规模训练 / 官方 Baseline 对比 | ⏳ 第二轮 | — |

## 3. 数据源

| 来源 | 许可 | 规模 | 处理 |
| --- | --- | --- | --- |
| IJCAI-2026 MCR 决赛回放（HF `Dannibal/ijcai-mahjong-corpora-2026`，经 hf-mirror 下载 `final2_all.jsonl.gz` 46MB） | MIT | 12,288 局全部 finished、全部无花；本轮用前 3000 局 | 逐座位走部署 Session 重放；响应→动作标签必须落在合法 mask 内；重建响应必须与记录一致；按整局拆分 train/val（game_id%10==0→val） |
| 启发式教师自对局（`mahjong/teacher.py`） | 本项目代码 | 1000 局（seed 1000–1999） | 向听数贪心 + 牌效 tie-break；**局限：无防守、无番种规划**，质量中等，仅作补充数据与诊断基线 |
| IJCAI-2020 人类数据（百度网盘，提取码 rm79） | 待确认 | 用户下载中 | 接口已就绪（replay.py 的协议解析可直接适配），等文件到位后接入 |

特征与训练/部署共用 `features.encode`（6×4×9：座风、圈风、4 层手牌计数），决策仅依赖 SeatView 可见信息，无隐藏手牌/牌墙泄露。混合数据集 train 732,211 / val 82,746 样本。

## 4. 训练记录（全部真实运行）

**BC v1**（`runs/bc_v1/`）：MPS，batch 1024，Adam lr 1e-3，8 epoch，49.8s。
val loss 0.533，总准确率 80.7%（pass 99.3% / hu 100% / **出牌 32.1%** / 鸣牌 39.6%）。
出牌准确率随机水平约 10%，32% 说明学到了有效信号；鸣牌偏低与极简特征有关（第二轮特征增强目标）。

**RL v1**（`runs/rl_v1/`，从 BC v1 resume）：CPU，8 actor × 1 局/迭代，PPO（γ0.99, λ0.95, lr 3e-4, clip 0.2, 3 epoch, mb 256），300 迭代，127.6s。
2400 局、846,109 决策步、10,401 次梯度更新。非荒庄终局比例 12.8%→23.3%，value loss 0.0087→0.0046，approx KL ~0.002 稳定，无非法动作、无退化报错。

Checkpoint：`runs/bc_v1/checkpoint.pt`（240KB）、`runs/rl_v1/checkpoint.pt`（691KB，含优化器/RNG/episode_cursor，可继续 resume）。

## 5. 评估结果（100 局/组，seed 100000 起，轮换座位，奖励=终局分/32）

| 策略 | 对手 | 平均奖励 | 胜/荒/负 |
| --- | --- | --- | --- |
| random | random | -0.011 | 0/99/1 |
| 未训练网络 | random | -0.015 | 0/98/2 |
| **BC** | random | **+0.048** | 4/95/1 |
| **BC+RL** | random | **+0.186** | 16/82/2 |
| teacher（参考） | random | +0.477 | 37/63/0 |
| BC | teacher | -0.258 | 3/25/72 |
| BC+RL | teacher | -0.206 | 9/18/73 |

同墙配对：RL 相对 BC 平均差 +0.138（vs random）、+0.053（vs teacher）。
**结论分级：模型可运行 ✅；真实训练 ✅；测试中改善 ✅（BC>随机，RL>BC）；超过课程官方 Baseline ❓（官方 Baseline 未知，teacher 只是自建参考，不能冒充官方基线）。**

## 6. Botzone 可部署性

- 导出：`submissions/mahjong_v1/`（`mahjong_bot.zip` 10KB + `mahjong_model.pt` 223KB，legacy torch 序列化 + pickle protocol 2，兼容 Torch 1.8）。
- 本地实测（真实子进程 stdin/stdout，简单交互 + 长时运行标记）：稳态推理 **平均 0.05ms / 最大 0.44ms**；首包（含模型加载）~0.42–0.52s；**173 轮冷启动历史重放 0.43s**，响应与在线决策完全一致。平台时限 1s/步，余量充足。
- 部署代码通过 Python 3.6 语法检查（ast feature_version）；ZIP 仅含推理必需文件。
- 提交方式（本轮未实际提交）：权重放个人存储根目录，ZIP 作 Python 3.6 源码上传，选 Chinese-Standard-Mahjong 136 张无花复式、简单交互。

## 7. 本机性能与远程 GPU 评估

实测吞吐：

| 项目 | 速度 |
| --- | --- |
| 教师数据生成 | ~34 局/s（8 进程） |
| 回放提取 | ~900 局/s（8 进程） |
| BC 训练（MPS, batch 1024） | ~118k 样本/s（732k 训练样本 × 8 epoch 共 50s） |
| BC 训练（CPU, batch 256） | ~40k 样本/s（小 batch 时 CPU 快于 MPS） |
| PPO（8 actor, CPU） | 0.41s/迭代 ≈ 20 局/s ≈ 6600 决策/s |

成本估算：一轮"BC 8 epoch + RL 2000 迭代 + 评估"≈ 50s + 14min + 3min ≈ **20 分钟**；RL 10000 迭代（8 万局）≈ 70 分钟。**瓶颈在 CPU 对局采样而非 GPU**（模型极小，MPS 仅在大 batch BC 时占优）。

**结论：暂不需要远程 GPU。** 本机可在数小时内完成多轮百万局级训练与消融。若第二轮要做更大模型/更密集采样（如 12+ actor 持续数天），再考虑课程服务器；届时建议配置：8 核以上 CPU（采样主导）、16GB 内存、任意入门 GPU 即可，需要 SSH 访问与 Python 3.10 环境。

## 8. 问题与风险

1. **荒庄率高**（弱策略下 80%+）：8 番起胡 + 极简特征导致奖励稀疏；RL 已缓解（95→82）。第二轮特征增强（公开discard历史、副露、剩余可见张数）与更长时间 RL 是主要提升手段。
2. **鸣牌准确率低**（39.6%）：当前特征看不到对手discard序列，难以判断鸣牌时机。
3. 教师无防守/番种规划，作为数据源有上限；IJCAI-2026 强 agent 数据是更优质的 BC 来源（已验证 0 过滤率），可扩大到全部 12288 局。
4. 官方 Baseline 未明确，当前所有对比均为自建参考。
5. `rl_framework` 的 FDL 相关测试因缺可选 `fdl` 包失败（不影响麻将）；`test_remote.py`/`test_cuda_execution.py` 未运行（无远程/CUDA）。
6. IJCAI-2020 人类数据尚未到位；其协议若与 2026 回放不同需适配（replay.py 已模块化）。

## 9. 第二轮行动建议

1. **特征增强**（课程推荐改进方向）：在 `features.py` 增加各座位discard计数/序列、副露编码、剩余可见张数通道；保持训练/部署同一 encode；重跑协议测试。
2. **扩大 BC**：IJCAI 全部 12288 局（~230 万样本，本机几分钟）；对比纯 IJCAI vs 混合教师数据；尝试类别加权缓解 pass 主导。
3. **放量 RL**：从新 BC 出发 2000–10000 迭代（本机 15–70 分钟/轮），监控荒庄率与对 teacher 胜率；保留 BC 权重防退化。
4. **评估升级**：复式评估（同墙四座位轮换取平均分差）、模型间循环赛；若课程公布官方 Baseline，第一时间对接。
5. 需要用户提供的资料：百度网盘 IJCAI-2020 数据（可选）、课程官方 Baseline 信息、（若需要）远程服务器访问方式。

## 10. 复现命令

```bash
# 环境（已建好 .venv；重建时）
~/.local/bin/python3.10 -m venv .venv
.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-lock.txt
.venv/bin/python -m pip install -e ./rl_framework

# 测试
.venv/bin/python -m unittest mahjong.tests.test_protocol mahjong.tests.test_export -v
PYTHONPATH=rl_framework/tests .venv/bin/python -m pytest rl_framework/tests -q \
  --ignore=rl_framework/tests/test_remote.py --ignore=rl_framework/tests/test_cuda_execution.py

# 数据
.venv/bin/python -m mahjong.bc_data --games 1000 --seed 1000 --output data/bc_teacher_v1 --workers 8 --shard-games 50
.venv/bin/python -m mahjong.replay --input data/ijcai2026/final2_all.jsonl.gz --output data/bc_ijcai_v1 --limit 3000 --workers 8 --shard-games 200

# 训练
.venv/bin/python -m mahjong.train_bc --data data/bc_ijcai_v1 data/bc_teacher_v1 --output runs/bc_v1 --epochs 8 --batch-size 1024 --device mps
.venv/bin/python -m mahjong.train --resume runs/bc_v1/checkpoint.pt --output runs/rl_v1 --iterations 300 --actors 8

# 评估（示例）
.venv/bin/python -m mahjong.evaluate --checkpoint runs/rl_v1/checkpoint.pt --output runs/eval_v1/rl_vs_teacher.json --episodes 100 --opponent teacher

# 导出
.venv/bin/python -m mahjong.export_bot --checkpoint runs/rl_v1/checkpoint.pt --output submissions/mahjong_v1
```

种子：BC seed 0；RL 配置 seed 2026（configs/default.json）；评估 seed 100000 起；教师数据 seed 1000 起。
产物目录：`runs/`（训练/评估）、`data/`（数据集）、`submissions/`（导出），均不入 git（见 .gitignore）；代码与报告入 git。