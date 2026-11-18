# MODEL_SPEC — 当前模型精确规格（审计实测）

审计基准 commit：`1227e4a8ad9144976f4537176a793f01db06d747`（main，与 origin 一致）
实测环境：Python 3.10.21 / torch 2.3.1 / macOS 26.5.2 arm64（Apple M5 Pro，15 核 CPU / 16 核 GPU / 24GB）
证据：`audit/model_bench.json`、`audit/model_structure.txt`、`audit/inventory.json`（脚本 `audit/model_bench.py`、`audit/inventory.py`）

## 1. 模型类与定义位置

- 类名：`Model`，定义于 `mahjong/model.py:7-21`（课程示例原样，未修改）。
- 训练包装：`NamedCategoricalPolicy`（`rl_framework/rlframework/algorithm/named_policy.py:10-65`），`input_keys=("obs",)`，`mask_key="action_mask"`。
- 部署加载：`mahjong/botzone/inference.py:11-32`（不依赖 RL 框架/TensorDict）。
- GitHub permalink：https://github.com/zzyimpact/MahjongGB-RL-Lab/blob/1227e4a8ad9144976f4537176a793f01db06d747/mahjong/model.py#L7-L21

## 2. 层级结构表（forward hook 实测 shape）

输入：`obs` float32 `[B, 6, 4, 9]`；`action_mask` bool `[B, 235]`（不进网络，仅在输出端屏蔽）。

| # | 层 | 类型 | 配置 | 输入 shape | 输出 shape | 参数量 |
|---|----|------|------|-----------|-----------|--------|
| 1 | conv.0 | Conv2d | 6→16, kernel 3×3, stride 1, padding 1, bias | [B,6,4,9] | [B,16,4,9] | 880 |
| 2 | conv.1 | ReLU | — | [B,16,4,9] | [B,16,4,9] | 0 |
| 3 | conv.2 | Conv2d | 16→16, kernel 3×3, stride 1, padding 1, bias | [B,16,4,9] | [B,16,4,9] | 2,320 |
| 4 | conv.3 | ReLU | — | [B,16,4,9] | [B,16,4,9] | 0 |
| 5 | reshape | Flatten | `reshape(B, -1)` | [B,16,4,9] | [B,576] | 0 |
| 6 | hidden.0 | Linear | 576→64, bias | [B,576] | [B,64] | 36,928 |
| 7 | hidden.1 | ReLU | — | [B,64] | [B,64] | 0 |
| 8 | actor | Linear | 64→235, bias | [B,64] | [B,235] | 15,275 |
| 9 | critic | Linear | 64→1, bias（squeeze 末维） | [B,64] | [B] | 65 |

- **无**归一化层（无 BN/GN/LN）、**无**残差连接、**无**注意力、**无**RNN、**无**池化、**无**额外规则分支、**无** Dropout。
- Actor 与 Critic **共享全部 backbone**（conv + hidden），仅输出头分离。
- 感受野：两层 3×3 卷积 → 5×5（在 4×9 网格上），即每个位置最多看到跨 2 行/2 列邻域；行内对应同花色数字邻接（顺子归纳偏置），跨行邻接（如 W9 与 T1）无语义。
- 信息压缩点：576 维卷积特征 → **64 维隐藏层**（压缩 9×），再到 235 维 logits。

## 3. 参数量（实际计算，非文件大小反推）

`sum(p.numel() for p in model.parameters())`：

| 模块 | 参数量 |
|------|--------|
| conv（conv.0 + conv.2） | 3,200 |
| hidden（576→64） | 36,928 |
| actor head（64→235） | 15,275 |
| critic head（64→1） | 65 |
| **总计 = 可训练参数** | **55,468** |

- 权重 dtype：全部 `torch.float32`（checkpoint 实测）。
- FP32 理论存储：55,468 × 4 = **221,872 字节**（≈216.7 KB）。
- 实际文件：部署权重 `submissions/mahjong_v1/mahjong_model.pt` = 223,221 B（legacy 序列化 + pickle protocol 2 开销）；BC checkpoint 239,970 B（含 config/RNG/空优化器）；RL checkpoint 691,298 B（含 Adam 动量 10 个张量状态、RNG、计数器）。

## 4. 输入特征（6 通道 × 4 × 9）

`mahjong/features.py:9-17`（训练/评估/部署共用同一 `encode`）：

| 通道 | 含义 | 编码 | 取值 |
|------|------|------|------|
| 0 | 座风 | F1-F4 之一 one-hot（列 27-30） | {0,1} |
| 1 | 圈风 | 同上 | {0,1} |
| 2-5 | 手牌计数 | 温度计：某牌有 c 张则前 c 层置 1 | {0,1} |

- 34 种牌按 `TILE_LIST`（W1-9, T1-9, B1-9, F1-4, J1-3）排成 4×9 网格：第 0/1/2 行 = 万/条/筒，第 3 行 = 字牌（列 0-6），**列 7-8 恒 0**（padding）。
- 无归一化（二值输入）；无弃牌历史、无副露、无牌墙计数、无向听/进张（详见主报告 §5 信息可见性表）。
- 合法动作 mask：235 维 bool，来自 `SeatView.valid`（`mahjong/observation.py:71-263`），含 8 番起胡的算番校验。

## 5. 输出与动作选择

- 输出：235 维 logits（Pass 1 / Hu 1 / Play 34 / Chi 63 / Peng 34 / Gang 34 / AnGang 34 / BuGang 34，`observation.py:19-28`）+ 标量 V。
- Mask 应用位置：
  - 训练采样/评估：`named_policy.py:33` `masked_fill(~mask, -1e9)`；PPO learner 端 `ppo.py:271` 用 `-inf`。
  - 部署：`inference.py:30` `masked_fill(~mask, -1e9).argmax`（**greedy**，无采样）。
  - 训练采样为 Categorical 随机采样（`named_policy.py:39`）；评估 `deterministic=True`（`evaluate.py:65`）。
- 非法动作概率在 softmax 后下溢为 0；全 mask 为空时抛错（`named_policy.py:31-32`），不会静默选非法动作。

## 6. Checkpoint 清单（全部可加载，审计实测）

| 文件 | 大小 | 内容 | 关键验证 |
|------|------|------|----------|
| runs/bc_v1/checkpoint.pt | 239,970 B | policy 55,468 参数 FP32 + 空 Adam(lr 3e-4) + config + RNG | **conv/hidden/critic == seed-0 随机初始化**（BC 只训 actor 头，见主报告 P1-1） |
| runs/rl_v1/checkpoint.pt | 691,298 B | policy + Adam 10 状态 + iteration 300 + counters | counters: 423,189 env steps / 846,109 agent steps / 10,401 updates / 2,400 episodes |
| submissions/mahjong_v1/mahjong_model.pt | 223,221 B | 纯 module state_dict（legacy 格式） | **与 rl_v1 的 module 权重逐张量相等**；与 bc_v1 不等 |

另有 7 个冒烟/基准 checkpoint（runs/bc_smoke*、runs/ppo_bench8 等），均可加载，见 `audit/inventory.json`。

## 7. 计算成本（微基准，预热后 50 次取中位数）

前向（含 mask 外的纯网络）：

| batch | CPU 1 线程 | states/s | MPS | states/s |
|-------|-----------|----------|-----|----------|
| 1 | 0.0406 ms | 24,628 | 0.474 ms | 2,111 |
| 64 | 0.435 ms | 147,176 | 0.405 ms | 158,220 |
| 256 | 1.830 ms | 139,918 | 0.427 ms | 599,795 |
| 1024 | 6.861 ms | 149,246 | 0.519 ms | 1,973,419 |

- 模型加载（权重文件缓存热）：1.82 ms。
- 完整决策管线（encode + from_numpy + forward + masked argmax，CPU 1 线程）：中位 **0.0396 ms**。
- 环境侧（事件重放 + 算番合法性校验 + encode，每联合步）：中位 0.0322 ms；同批座位模型决策合计 0.103 ms。
- 进程生命周期峰值 RSS：237.3 MB（含 torch 导入与基准，**非**纯推理峰值；纯推理峰值未单独可靠测量）。
- MPS batch=1 比 CPU 慢 ~12×（传输开销）；batch≥256 后 MPS 吞吐为 CPU 4-13×。
- PPO 迭代分解（8 actor，实测 `audit/ppo_timing.json`）：采样 0.18 s（39%）+ GAE 0.03 s（6%）+ learner 更新 0.25 s（53%）≈ 0.47 s/迭代 ≈ 17 局/s。

## 8. 容量局限（代码证据 vs 推测）

**代码证据：**
- 输入仅 6 通道二值特征；SeatView 已维护的弃牌历史/副露/牌墙/公开牌计数（`observation.py:38-47`）全部未进网络。
- BC 阶段 backbone 冻结为随机权重（`train_bc.py:122` 优化器只含 `model.actor.parameters()`），当前 BC 模型实质是"随机 CNN 特征 + 训练过的线性读出"。
- 576→64 的压缩与 2 层 16 通道卷积是仅有的表征瓶颈点；无残差/归一化，深度无法直接堆叠。

**合理推测（未实验证明）：**
- 55k 参数相对参考实现（14.5M / 19.26M，见主报告 §13）小 260-350 倍，几乎必然不足以表达丰富特征下的麻将策略；但**当前证据无法区分"容量不足"与"信息不足"各自的贡献**——需要主报告 §13.4 的 2×2 消融。
- 参考消融（LegendXct）显示：特征侧 vec +2.32pp、GELU +6.22pp，而 2.3× 参数的双分支结构无收益——提示先补信息与训练配方，再扩容量。

## 9. 部署兼容

- 导出 ZIP 仅含推理必需 14 个文件；`ast.parse(feature_version=(3,6))` 通过（语法级）。
- legacy torch 序列化 + pickle protocol 2（`export_bot.py:29-34`），兼容旧版 torch 加载。
- 本地子进程实测：稳态 0.06 ms/请求，最大 0.6 ms；首包（含加载）~0.47 s；173 轮冷启动重放 0.50 s，响应与在线一致。平台时限 1 s/步。
- **未验证**：真实 Botzone 运行环境（课程 README 称 Python 3.6 / Torch 1.8；LegendXct 实测报告为 Python 3.6.5 / torch 1.10.2 / 用户存储 256MB——两者不一致，以平台实际为准）。当前 55k FP32 模型（~220KB）在任何一种环境下余量都极大。