# AUDIT_EVIDENCE — 审计证据索引

审计日期：2026-11-19 · 审计基准 commit：`1227e4a8ad9144976f4537176a793f01db06d747`
 permalink 前缀：`https://github.com/zzyimpact/MahjongGB-RL-Lab/blob/1227e4a8ad9144976f4537176a793f01db06d747/`
所有审计脚本在 `audit/`，输出 JSON 同目录；**未修改任何项目源码、checkpoint、数据集或已有日志**（本轮 git 工作区新增仅 audit/）。

## 1. 版本与产物核对（audit/inventory.py → inventory.json）

执行命令：`PYTHONPATH=. .venv/bin/python audit/inventory.py`

| 项 | 结果 |
|----|------|
| 本地路径 / 分支 | /Users/zzy/Desktop/Coding/rl-lab / main |
| HEAD | 1227e4a8ad9144976f4537176a793f01db06d747 |
| origin | git@github.com:zzyimpact/MahjongGB-RL-Lab.git（私有，用户确认） |
| 本地 vs origin/main | 一致（git rev-parse origin/main 相同；git status 干净，审计前） |
| ROUND1_REPORT.md 对应提交 | 1227e4a（即 HEAD） |
| Python / torch / tensordict / numpy | 3.10.21 / 2.3.1 / 0.1.0 / 1.26.4 |
| PyMahjongGB | 1.4.0（本地编译；包无 `__version__` 属性，版本取自 dist-info） |
| MPS | 可用 |

产物存在性与可读性（全部实测加载/计数）：

| 产物 | 大小 | 来源 | 可读 |
|------|------|------|------|
| runs/bc_v1/checkpoint.pt | 239,970 B | 第一轮 BC（8 epoch，MPS） | ✅ 55,468 参数 FP32 |
| runs/rl_v1/checkpoint.pt | 691,298 B | 第一轮 PPO 300 迭代 | ✅ counters 与报告一致 |
| runs/eval_v1/*.json（7 个） | 各 ~22 KB | 第一轮评估 7×100 局 | ✅ 含逐局 records |
| runs/bc_v1/bc_history.json | — | BC 逐 epoch 指标 | ✅ |
| runs/rl_v1/metrics.jsonl | — | PPO 逐迭代指标（300 行） | ✅ |
| data/bc_ijcai_v1（15 shards） | 6.5 MB | IJCAI-2026 前 3000 局 | ✅ 重计数 563,257 = meta |
| data/bc_teacher_v1（20 shards） | 3.3 MB | teacher 自对局 1000 局 | ✅ 重计数 251,700 = meta |
| data/ijcai2026/final2_all.jsonl.gz | 45,777,973 B | hf-mirror 下载（MIT） | ✅ 12,288 行 |
| data/ijcai2026/final2_games_summary.jsonl.gz | 200,512 B | 同上 | ✅ |
| submissions/mahjong_v1/ | 236 KB | 第一轮导出 | ✅ 权重==rl_v1 |
| data/mjdata.sfs | 164 MB | 用户提供（IJCAI-2020 数据包） | ⚠️ SquashFS，本机无 unsquashfs/7z，**内容未直接读取** |
| data/output2017/ | 2.0 GB | 用户提供（疑似 sfs 解压内容） | ✅ 文本可读；PLAY 385,324 / MO 132,994 / LIU 12,140 文件 |
| runs/（其余冒烟/基准目录） | — | 第一轮 | ✅ 9 个 checkpoint 全部可加载 |

权重身份核验（inventory.json `export_identity` / `bc_critic_vs_fresh_init`）：
- 导出权重与 rl_v1 module 权重**逐张量 torch.equal 相等**；与 bc_v1 不等。
- rl_v1 相对 bc_v1：actor/critic/conv 均变化。
- bc_v1 的 conv.0/conv.2/hidden.0/critic 权重 == `torch.manual_seed(0)` 后 `Model(64)` 初始化；actor 不等。
- bc checkpoint 优化器：1 param group，lr 3e-4，**state 条目 0**（空 Adam，供 RL resume）；rl checkpoint：state 条目 10。

## 2. 模型结构与基准（audit/model_bench.py → model_bench.json / model_structure.txt）

命令：`PYTHONPATH=. .venv/bin/python audit/model_bench.py`
- 参数量：总 55,468 = conv 3,200 + hidden 36,928 + actor 15,275 + critic 65；FP32 理论 221,872 B。
- 逐层 shape 由 forward hook 实测（见 MODEL_SPEC §2）。
- 前向基准（预热 10 次，50 次中位数）：CPU b1 0.0406 ms / b1024 6.861 ms；MPS b1 0.474 ms / b1024 0.519 ms。
- 决策管线（encode+forward+argmax）CPU 中位 0.0396 ms；模型加载 1.82 ms。
- env vs forward：167 联合步，env 中位 0.0322 ms，forward 合计 0.103 ms。
- 峰值 RSS 237.3 MB（进程生命周期口径，macOS ru_maxrss）。

## 3. 规则边界检查（audit/rule_checks.py → rule_checks.json）

命令：`PYTHONPATH=. .venv/bin/python audit/rule_checks.py` · **12/12 通过**

| 检查 | 结果 |
|------|------|
| 碰碰和样例算番 = 26 | ✅ |
| 七对（含自摸/五门齐）= 31 | ✅ |
| 杠上开花（有 GANG 副露 + isAboutKong + 自摸） | ✅ 8 番计入 |
| 妙手回春（isWallLast + 自摸）/ 海底捞月（isWallLast + 点和） | ✅ |
| 低番点和（6 番）SeatView 不提供 Hu | ✅ valid=[0] |
| 抢杠和（两面听 + isAboutKong）SeatView 提供 Hu，13 番 | ✅ valid=[1,0] |
| 裁判拒绝 <8 番胡牌：reward [-30,10,10,10] | ✅ |
| 自摸第 4 张 is4thTile=True（shownTiles=3+1） | ✅ |
| 荒庄 reward [0,0,0,0] | ✅ |

算番库语义核对（C++ 源码，pip sdist 解包于 /tmp/pmg_src，**未改动项目**）：
- `fan_calculator.cpp:2367-2380`：杠上开花要求副露含 GANG；抢杠和要求 `fixed_table[win]==0 && standing_table[win]==1`（和牌张含入后恰 1 张 → 单钓抢杠在物理上不可能，库语义正确）。
- `fan_calculator.cpp:2348`：`++standing_table[win_tile]`。
- `fan_calculator.cpp:1566`：杠上开花不计自摸；妙手回春不计自摸（1562-1564）。
- 初版审计测试 3 处 FAIL 均为**测试构造错误**（缺杠副露 / 物理不可能的单钓抢杠 / 手牌实际含花龙 8 番），修正后全过；**未发现算番库或裁判规则错误**。

课程自带测试：`mahjong/tests/test_protocol.py` 2/2 通过（40 局完整对局，Session vs SeatView 逐事件比对）；其中 `test_self_draw_fourth_tile_counted_once` 覆盖上游已知的"和绝张重复计数"bug（LegendXct README 报告其 baseline 环境存在此 bug；本课程示例已含回归测试，engine.py:103 注释明确）。
框架测试：`PYTHONPATH=rl_framework/tests .venv/bin/python -m pytest rl_framework/tests -q --ignore=test_remote.py --ignore=test_cuda_execution.py` → 31 通过 1 跳过 4 失败；4 个失败均为 `ModuleNotFoundError: No module named 'fdl'`（可选 FDL 桥接，麻将路径不使用）。

## 4. 数据统计与泄漏（audit/data_stats.py → data_stats.json；audit/clean_val.py → clean_val.json）

命令：`PYTHONPATH=. .venv/bin/python audit/data_stats.py`、`PYTHONPATH=. .venv/bin/python audit/clean_val.py`

动作分布（IJCAI v1，563,257 样本）：pass 72.03% / play 24.95% / chi 1.45% / peng 0.92% / hu 0.53% / bugang 0.05% / gang 0.04% / angang 0.02%。平均合法动作 3.352；**单合法动作样本 68.2%**。teacher v1 类似（pass 70.63%，单合法 69.0%）。

泄漏专项：
- 从原始回放 INIT display 提取 srand：前 3000 局 = **126 个独立牌墙**；组大小分布 {24: 124 组, 23: 1, 1: 1}（4 名选手 × 座位排列的复式重复）。
- `game_id % 10 == 0` 切分下：**125 个牌墙组跨 train/val；100% 的 val 局与 train 共享牌墙**。
- 精确 (obs float32 + mask bool + action int16) blake2b 哈希：**92.78% 的 IJCAI val 样本（52,385/56,459）在 train 中有完全相同样本**。
- 无泄漏对照：取第 3000-3200 局中 srand 不重叠的 176 局（10 个新墙，37,456 样本），BC 前向评估：总 acc 81.27% / play 32.78% / nonpass 34.55% —— 与泄漏 val（81.15% / 32.41% / 34.68%）**几乎相同** → 记忆效应有限，小模型以泛化为主；但切分方法学无效。
- RL v1 在同批新墙样本上与专家动作一致率：play 18.16%（BC 32.78%）——RL 偏离模仿目标（一致率非棋力指标）。

BC val 重算（保存的 checkpoint，前向 only）：
- IJCAI val：pass acc 99.17%（n=40,684）/ hu 100%（301）/ play 32.41%（14,093）/ chi 26.42%（810）/ peng 65.37%（514）；单合法样本 acc 100%（n=38,520）；nonpass acc 34.68%（n=15,775）。
- teacher val（无墙重复，干净）：总 79.76% / play 31.30% / chi 16.7% / peng 50%。

数据源 A 抽查（500 个 PLAY 文件，seed 0）：455/500（91%）含花牌（H1-8）或补花动作；格式为中文文本（摸牌/打牌/碰/补花），番表含"花牌"。**144 张含花，与课程 136 无花复式不兼容**，需过滤/转换（本轮未做）。

## 5. BC 训练审计（代码 + runs/bc_v1/bc_history.json）

- `train_bc.py:122`：`optimizer = Adam(model.actor.parameters(), lr=1e-3)` → **仅 actor 头更新**；conv/hidden 梯度被计算但不应用（权重比对证实保持 seed-0 初始化）。
- `train_bc.py:135-137`：mask 以 -1e9 进入 CE；无类别加权；无 LR schedule；batch 1024；8 epoch；seed 0。
- `train_bc.py:31-35`：多源 game_id 偏移 10^7 后 `%10` 切分（泄漏见 §4）。
- `train_bc.py:156-176`：按 val loss 最优保存；checkpoint 含空 Adam(lr 3e-4) 供 RL resume。
- 曲线（bc_history.json）：train loss 0.5670→0.5384，val loss 0.5497→0.5331，play acc 0.3057→0.3206；8 epoch 仍在下降，**无过拟合迹象，训练不充分**；val loss 低于 train loss 与 val 集泄漏/简单样本占比差异一致。
- 梯度更新：716/epoch × 8 = 5,728（日志）。

## 6. PPO 实证检验（audit/ppo_checks.py → ppo_checks.json；audit/ppo_timing.py → ppo_timing.json）

命令：`PYTHONPATH=. .venv/bin/python audit/ppo_checks.py`、`... audit/ppo_timing.py`

| 检验 | 结果 |
|------|------|
| γ=λ=1 望远镜恒等式 advantage_t = R_terminal − V(s_t)（非荒庄局 seed 797，BC 策略） | 最大误差 3.6e-7 ✅ |
| 奖励路由：seat 3 自摸 12 番 → +1.875，其余 −0.625；零和 | ✅ |
| 奖励仅落在每座位最后一个决策；terminated/discount/trace 标志 | ✅ |
| 独立 numpy GAE（γ0.99/λ0.95）对照框架实现 | 最大误差 2.5e-7 ✅ |
| 更新前 ratio = exp(new_logp − old_logp) ≡ 1 | 偏差 9.5e-7 ✅ |
| 每局 4 条独立轨迹（每座位一条），key 含 agent_id | ✅ |
| gradient_updates = 3 × ceil(agent_steps/256)，300/300 迭代吻合 | ✅ |
| 日志总和 = 423,189 env / 846,109 agent / 10,401 updates / 2,400 episodes | ✅ 与报告一致 |
| 非荒庄终局比例（非零奖励轨迹占比）first50 12.75% → last50 23.25% | ✅ |
| 座位级 vs 局级口径：300/300 迭代非零计数均为 4 的倍数；全局 401/2400 局=16.71% | ✅ 两者相等（零和） |
| 迭代耗时分解（8 actor）：collect 0.18 s / GAE 0.03 s / update 0.25 s | update 占 53% |
| 子进程 CPU 占用 | **未测**：macOS RUSAGE_CHILDREN 只统计已退出子进程 |

代码要点：
- BC→PPO：`train.py:57-71` resume 加载完整 policy（含 critic）+ 优化器 + RNG + episode_cursor；hidden/capacity 不匹配则拒绝。
- 参数更新校验：`train.py:105-114` 每迭代比对前后权重，未变化即抛错；300 迭代全部 `parameters_changed=true`（metrics.jsonl）。
- 自对弈：`policy.py:23` 四座位路由到同一 "shared" 策略；无固定对手/历史快照/KL 锚定。
- 奖励：`env.py:39` 终局分/32；`collectors/rewards.py:22-28` 每 tick 读共享 rewards 向量；`collectors/sparse.py:169-175` 累加到 pending 决策；终局所有座位关闭 pending（`sparse.py:176-206`）。
- 非法动作：`env.py:50-51` mask 外动作抛错；`env.py:55-69` 裁判拒绝即 RuntimeError（训练中未发生）。
- GAE：`processing/gae.py:26-77` 逐轨迹、终局 bootstrap 0、episode_start 重置 carry；决策时钟折扣（`sparse.py:232-234` discount_clock="decision"）。
- PPO 目标：`algorithm/ppo.py:221-257` clip 0.2 / value 0.5 / entropy 0.01 / grad clip 0.5（PPOConfig 默认）/ advantage 归一化；approx_kl = ((ratio−1)−log_ratio).mean()。

## 7. 评估统计（audit/eval_stats.py → eval_stats.json）

命令：`PYTHONPATH=. .venv/bin/python audit/eval_stats.py`（逐局 records，n=100/组，seed 100000-100099，座位各 25）

| 评估 | mean | 95% CI | W/D/L |
|------|------|--------|-------|
| random vs random | −0.0109 | [−0.0324, 0.0105] | 0/99/1 |
| untrained vs random | −0.0150 | [−0.0363, 0.0063] | 0/98/2 |
| BC vs random | +0.0481 | **[-0.0056, 0.1018]（含 0）** | 4/95/1 |
| RL vs random | +0.1856 | [0.0910, 0.2803] | 16/82/2 |
| teacher vs random | +0.4766 | [0.3432, 0.6099] | 37/63/0 |
| BC vs teacher | −0.2584 | [−0.3331, −0.1837] | 3/25/72 |
| RL vs teacher | −0.2056 | [−0.3092, −0.1020] | 9/18/73 |

配对（同 seed = 同牌墙同座位；对手动作交互分歧）：
- RL−BC vs random：+0.1375，CI [0.0296, 0.2454]，16 胜 / 6 负 / 78 平 → **显著**。
- RL−BC vs teacher：+0.0528，CI [−0.0670, 0.1726]，29/34/37 → **不显著**。
- BC−untrained vs random：+0.0631，CI [0.0058, 0.1204] → 显著。
- 奖励均为 1/32 整数倍、每局零和：7/7 文件通过。
- 评估推理 greedy（evaluate.py:65）；对手 random 用 RandomState(seed)，teacher 确定性。
- **非复式冻结对手**：同墙但对手动作随受控策略变化而分歧；真正的复式配对 arena 未实现。

## 8. Botzone 部署（mahjong/tests/test_export.py，本轮重跑）

命令：`BOT_TEST_CHECKPOINT=runs/rl_v1/checkpoint.pt .venv/bin/python -m unittest mahjong.tests.test_export -v` → **3/3 OK**
- py3.6 语法（ast feature_version）✅
- 真实子进程 stdin/stdout：517 请求，稳态均值 0.06 ms / 最大 0.6 ms，首包 469.75 ms ✅
- 冷启动重放 173 轮 502.99 ms，响应与在线完全一致 ✅
- manifest.json：iteration 300，counters 与 rl_v1 一致；权重身份见 §1。
- 无静默回退：`inference.py:17` `load_state_dict(strict=True)`，失败即异常；`inference.py:28-29` 空 mask 抛错。
- **未验证**：真实 Botzone 平台（无登录/提交，本轮边界禁止）；平台 torch 版本（README 1.8 vs LegendXct 报告 1.10.2）；平台内存上限（LegendXct 称用户存储 256MB）。

## 9. 参考仓库调研（网络实测）

| 仓库 | 可访问性 | 获取内容 |
|------|----------|----------|
| LegendXct/Chinese-Standard-Mahjong-AI | ✅ raw.githubusercontent.com/main/README.md（24,714 B，存 /tmp/legend_readme.md） | IJCAI 2026 算法赛道积分第 3 / 决赛第 11；M5 纯 SL 14.5M，val 89.55%；102×4×9 + 107 vec；PreAct ResNet w192 d20 GN+GELU；98,209 人类局 → 1847 万决策点，按局 95/5 切分；花色置换增广；AdamW bf16 batch1024 20epoch 标签平滑 0.05；arena 复式配对 500+ 组；RL 一代弱基座 +1.41±0.32（2425 更新/275 万局/41.8h），强基座边际≈0；KL 锚定 + 对手池有效；同步 PPO 终局奖励 −0.4~−0.8；fanback +0.22 / potential +0.30（<2SE）；完美信息 critic R² 仅 +0.010；部署 py3.6.5/torch1.10.2/fp16 常驻 182.6MB |
| PiggyQwQ/chinese-standard-mahjong-ai | ✅ README（3,223 B，/tmp/piggy_readme.md） | 北大 AI 基础 2026 春同课程；SE-Res2Net w480 14 块 19.26M；IJCAI 95,621 专家局（官方 SL data.txt）；val 弃牌 86.83%，200 局 +2.47/局；**9 次 RL 全失败（P1-P4 病理，含奖励黑客：指标健康但对打 −30/局）**；val acc≠棋力案例；评测噪声 ±6.5/局；推理时增强无可测增益 |
| SuuTTT/IJCAI-mahjong | ❌ main/master raw 均 404，api.github.com 404 | **未验证**（HF 数据集 README 引用该仓库，但当前不可公开访问） |

候选架构参数量（torch 模块实际构造计数，含 107 vec 双头，4×9 网格）：
w64×4块/30ch = 1,319,692；w64×6 = 1,467,404；w96×6 = 2,032,492；w128×8/102ch = 3,492,044；w192×20/102ch = 14,474,380（与 LegendXct 报告 ~14.5M 吻合）。

## 10. 无法验证 / 未测事项清单

1. 真实 Botzone 平台行为（时限/内存/torch 版本/在线对手）——本轮禁止提交。
2. mjdata.sfs 内部内容（无 squashfs 工具）；output2017 与其是否同一内容仅凭 inode 数推断（530,462 vs 530,458 文件）。
3. IJCAI-2026 全 12,288 局的牌墙总数（仅核实前 3000 局 = 126 墙；按 24 局/墙外推 ≈ 512 墙，**外推未验证**）。
4. 子进程 CPU 利用率（macOS rusage 限制）。
5. 纯推理峰值内存（ru_maxrss 为进程生命周期口径）。
6. SuuTTT/IJCAI-mahjong 仓库内容（404）。
7. 官方课程 Baseline 强度（未知，所有对比为自建参考）。
8. 容量 vs 信息瓶颈的分离（需下一轮消融，本轮静态审计无法判定）。
9. PyMahjongGB 1.4.0 的 shanten API 是否仍有 1.3.0 报告的状态污染（本项目未使用该 API，无影响；未测）。
10. 训练后期 RL 是否退化（300 迭代内指标健康；无更长程日志）。

## 11. 本轮执行的全部命令（可复现）

```bash
git rev-parse HEAD && git status --short && git rev-parse origin/main
PYTHONPATH=. .venv/bin/python audit/inventory.py
PYTHONPATH=. .venv/bin/python audit/model_bench.py
PYTHONPATH=. .venv/bin/python audit/rule_checks.py
PYTHONPATH=. .venv/bin/python audit/data_stats.py
PYTHONPATH=. .venv/bin/python audit/clean_val.py
PYTHONPATH=. .venv/bin/python audit/ppo_checks.py
PYTHONPATH=. .venv/bin/python audit/ppo_timing.py
PYTHONPATH=. .venv/bin/python audit/eval_stats.py
BOT_TEST_CHECKPOINT=runs/rl_v1/checkpoint.pt .venv/bin/python -m unittest mahjong.tests.test_export -v
.venv/bin/python -m unittest mahjong.tests.test_protocol -v
PYTHONPATH=rl_framework/tests .venv/bin/python -m pytest rl_framework/tests -q \
  --ignore=rl_framework/tests/test_remote.py --ignore=rl_framework/tests/test_cuda_execution.py
.venv/bin/python -m pip download --no-binary :all: --no-deps -d /tmp/pmg_src PyMahjongGB==1.4.0  # 算番源码核对
```

审计脚本不写任何项目已有文件；ppo_timing.py 做 4 次一次性 PPO 更新用于计时，**不保存 checkpoint**。