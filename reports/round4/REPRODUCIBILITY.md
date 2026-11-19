# 第四轮复现说明

## 仓库与实际环境

原目录 `/Users/zzy/Desktop/Coding/rl-lab` 在本次工具环境中返回 `Operation not permitted`。没有删除或覆盖其中内容。实际从 origin 的 `d150d3d7e9345642e76f62571cad21dc48b87294` 恢复到 `/Users/zzy/Projects/MahjongGB-RL-Lab-v3-work`，工作分支 `codex/mahjong-v3-final`。另保留完整恢复克隆 `/Users/zzy/Projects/MahjongGB-RL-Lab-v3`。原目录可能存在的未提交文件未能核验，不声称已迁移。

环境实测：Apple M5 Pro、15 CPU、24 GiB 内存，macOS 26.5.2 arm64，Python 3.10.22、PyTorch 2.3.1、NumPy 1.26.4、TensorDict 0.1.0、PyMahjongGB 1.4.0。MPS 可用。BC 使用 MPS，Arena/部署使用 CPU，初始受控 PPO 使用 CPU。完整依赖记录见 raw 中的环境文件。MPS/CPU 跨硬件不承诺逐 bit 相同；分组、episode seed、配置、权重身份均保存。

推荐新环境：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install torch==2.3.1 tensordict==0.1.0 numpy==1.26.4 PyMahjongGB==1.4.0 pytest==9.2.0 matplotlib==3.10.9
.venv/bin/python -m pip install -e ./rl_framework
```

本机由 uv 建立环境，`.venv` 是指向 `/Users/zzy/Projects/mahjong-v3-env310` 的本地软链接，不提交 Git。不要复制环境中指向作者绝对路径的 editable 安装记录。

## 数据来源与分组

继续使用第三轮同一文件，未增加 BC 专家对局：

```bash
mkdir -p data/ijcai2026
curl -L --fail 'https://huggingface.co/datasets/Dannibal/ijcai-mahjong-corpora-2026/resolve/main/final2_all.jsonl.gz' -o data/ijcai2026/final2_all.jsonl.gz
```

如网络需要镜像，必须仍核对原文件 SHA。`data_v3` 会与 `reports/round3/raw/split_manifest.json` 中 SHA-256 严格匹配，然后逐局核对完整原始牌墙哈希。12,288 局、512 墙，完全保留 410/51/51 分组。原始数据与大缓存不进入 Git，可从上游取得或自行按 SHA 恢复。

在新输出目录准备数据，避免覆盖已有缓存：

```bash
.venv/bin/python -m mahjong.data_v3 --input data/ijcai2026/final2_all.jsonl.gz --manifest reports/round3/raw/split_manifest.json --output data/reproduce_v3 --workers 6
.venv/bin/python -m mahjong.audit_v3_data --data data/reproduce_v3 --source data/ijcai2026/final2_all.jsonl.gz --output data/reproduce_v3/audit.json
```

只存储非强制决策，强制决策的真实类别计数保留在 manifest；其 masked CE 精确为 0、准确率为 1。V3 有效样本 train/val/test = 576,202 / 75,685 / 73,294。总样本仍为 1,815,376 / 239,322 / 229,832。特征源代码与 PyMahjongGB 版本指纹写入 `feature_provenance.json`，来源变化时旧缓存拒绝加载。中断的数据构建可用同一输入、分组及 shard 设置加 `--resume`。

## BC 实验

主配方：非强制样本全量洗牌，AdamW，batch 512，学习率 5e-4，1 epoch warmup，20 epoch cosine 规划至 2.5e-5，weight decay .01、合法动作标签平滑 .02、梯度裁剪 5。至少观察 12 epoch；验证非强制 CE 超过 `.001` 才更新最佳状态，耐心 6。该规则会保留与精确最小值相差小于 .001 的较早快照，报告不会将其冒充精确 argmin。

```bash
.venv/bin/python -m mahjong.experiment_v3 --data data/reproduce_v3 --root runs/reproduce_v3 --configs \
 mahjong/configs/v3/D_retrained.json mahjong/configs/v3/V3.json \
 mahjong/configs/v3/V3_no_quality.json mahjong/configs/v3/V3_no_context.json \
 mahjong/configs/v3/V3_relu.json mahjong/configs/v3/D_constant.json
```

更保守配方同时降低学习率至 2e-4、batch 256、合法标签平滑 .1，在 12 epoch 完成 cosine 衰减后可继续低学习率训练至最多 20。它是**联合配方实验**，不将差值分别归因于某一个超参数。D 与 V3 采用同样配方：

```bash
.venv/bin/python -m mahjong.experiment_v3 --data data/reproduce_v3 --root runs/reproduce_v3 --configs \
 mahjong/configs/v3/D_regularized.json mahjong/configs/v3/V3_regularized.json
```

主配方另外使用 seed 20261120、20261121（首个为 20261119）：

```bash
.venv/bin/python -m mahjong.experiment_v3 --data data/reproduce_v3 --root runs/reproduce_v3 --configs \
 mahjong/configs/v3/D_retrained_seed2.json mahjong/configs/v3/V3_seed2.json \
 mahjong/configs/v3/D_retrained_seed3.json mahjong/configs/v3/V3_seed3.json
```

单实验中断恢复：

```bash
.venv/bin/python -m mahjong.train_v3 --config mahjong/configs/v3/V3.json --data data/reproduce_v3 --output runs/reproduce_v3/V3 --resume
```

保存 `last.pt`、验证最佳 `checkpoint.pt`、阶段快照、完整优化器和 RNG，以及每 epoch JSON/JSONL。CPU 单元测试验证中断恢复后的权重与不间断训练逐 bit 一致。不要编辑旧配置后直接续跑：配置与 manifest 不一致会拒绝。

## Arena 与统计

新开发种子：110000000 起 32 墙筛选；111000000 起 128 墙比较；PPO 周期开发评估 110200000 起 64 墙。另有 110100000 起 2 墙仅做流程冒烟，不能用于实力结论。最终集合使用 112000000 起 500 墙，在候选封存后才打开。

```bash
.venv/bin/python -m mahjong.arena_v3 \
 --candidate D_original=artifacts/round3/D/checkpoint.pt \
 --candidate V3=runs/reproduce_v3/V3/checkpoint.pt \
 --opponent random --opponent teacher \
 --opponent rl_v1=artifacts/round3/references/rl_v1.pt \
 --opponent C=artifacts/round3/C/checkpoint.pt --opponent D=artifacts/round3/D/checkpoint.pt \
 --seed 111000000 --walls 128 --workers 4 --registry-root runs/reproduce_v3 --purpose development --output runs/reproduce_v3/arena_dev
```

每副墙四个候选座位，各场对手都是其余三席同一冻结策略。统计以墙聚类，五对手等权综合分也在同一墙上先聚合，然后用 5,000 次 bootstrap、seed 20261119 计算区间。相同墙的四场绝不视为四个独立样本。跨训练 seed 结果另用 seed 对和共同墙双层 bootstrap（仅三个训练 seed，仍有统计局限）。

`arena_v3` 默认对 `runs/final_v3` 下注册集合执行交叉排除；使用其他根目录时须提供 `--registry-root`，如上例。PPO 和 study 队列自动使用对应实验根目录；所有新开发/训练集合应注册在同一根目录。`wall_audit_v3 --root ...` 可检查指定根目录。相同开发集合重复评估允许共享墙，最终集合不能与开发/训练共享墙。

候选封存与 test：

```bash
.venv/bin/python -m mahjong.finalize_v3 seal --candidate V3=runs/reproduce_v3/V3/checkpoint.pt --output runs/reproduce_v3/selection.json --reason 'Selected using development arena only'
.venv/bin/python -m mahjong.finalize_v3 test --seal runs/reproduce_v3/selection.json --data data/reproduce_v3 --output runs/reproduce_v3/test_metrics.json --device mps
```

test 沿用第三轮 51 墙；它不是项目历史上从未公开过的测试集，但本轮新模型/超参选择不使用 test 指标。最终新 Arena 另行独立。若看过 final 后继续调参，必须将其改标探索性并建立新确认集合。

## 受控 PPO

使用现有完整对局 collector、GAE 和 PPO objective。两学习席轮转对两个确定性冻结对手，BC/C/D 按 seed 循环。Critic 梯度在共享状态处截断；前 8 iteration 只训练 Critic，共 128 局，Actor 参数逐 tensor 不变。随后 lr=2e-5、clip=.1、entropy=.001、2 个 PPO epoch、batch 512。两个初始实验只改变 BC forward KL 系数 .2 与 1.0，均从同一 BC 开始，不从退化策略继续。

```bash
.venv/bin/python -m mahjong.ppo_v3 --config mahjong/configs/v3/PPO_V3_mixed.json --checkpoint runs/reproduce_v3/V3/checkpoint.pt --output runs/reproduce_v3/PPO_V3_mixed
.venv/bin/python -m mahjong.ppo_v3 --config mahjong/configs/v3/PPO_V3_anchor.json --checkpoint runs/reproduce_v3/V3/checkpoint.pt --output runs/reproduce_v3/PPO_V3_anchor
```

训练种子 130000000 起，两个控制实验故意复用训练墙；实际执行对局数和独立训练墙数分别报告。初始纯流程 smoke 使用 135000000 起 4 局，不能当作实力实验。周期评估选 `ppo_best.pt`；`eval_best.pt` 同时保护 BC，`last.pt` 保留最后状态，原 BC 永不覆盖。

完整策略、优化器、RNG、episode cursor 及轨迹计数支持 `--resume`。初始版本没有保存各 actor 子进程的 RNG 状态，因此跨中断恢复不承诺与不间断 PPO 逐 bit 一致；从起点在同样环境运行的 seed、对手和采样计划固定。所有恢复后的局号仍递增，不复用先前 episode seed。

## 导出、文件限制与运行验证

```bash
.venv/bin/python -m mahjong.export_bot --checkpoint runs/reproduce_v3/V3/checkpoint.pt --output submissions/reproduce_v3
.venv/bin/python -m mahjong.verify_submission submissions/reproduce_v3
.venv/bin/python -m mahjong.deployment_benchmark submissions/reproduce_v3 --games 8
```

生成单独的 `mahjong_model.pt`、`mahjong_bot.py`、`mahjong_bot.zip`。单文件代码使用内存模块加载器，只打包项目推理源码；没有嵌入权重，没有 TensorDict/训练框架依赖。用户只允许上传一个代码文件时，上传 `.py`；权重单独上传到用户存储，默认读取 `data/mahjong_model.pt`。本地可通过 `BOT_MODEL_PATH` 指定绝对路径。

硬门槛：模型 ≤268435456 bytes；ZIP <4000000 bytes；解压后的代码/资源总和 <4000000 bytes；另检查单个 `.py` <4000000 bytes。禁止意外混入 `.pt`、数据、日志、测试。导出 manifest 和静态验证写出真实字节数与 SHA-256。部署测试使用临时干净目录、真实子进程、完整对局、四席轮换和冷历史重放。

本机运行版本不等同于 Botzone 线上可用版本。V3 要求运行时的 MahjongGB 提供五种结构向听及有效进张 API；只支持算番的旧包不能静默降级成其他特征。线上 Python/PyTorch/原生包可用性、执行时间与内存限额仍须用户上传验证。

## 测试边界与原始记录

```bash
.venv/bin/python -m pytest mahjong/tests -q
BOT_TEST_CHECKPOINT=runs/reproduce_v3/V3/checkpoint.pt .venv/bin/python -m pytest mahjong/tests/test_export.py -q
PYTHONPATH="$PWD/rl_framework/tests:$PWD" .venv/bin/python -m pytest rl_framework/tests -q
```

完整框架测试中的四个 FDL 集成测试依赖仓库未包含的 `fdl`/生成样例，当前会因缺失模块失败；CUDA 测试在此 MPS 机器上跳过。不能把这些记录写成“整个仓库全绿”。Mahjong 新功能、PPO 数学/轨迹及已具备依赖的框架核心另有通过记录。全部首轮错误和后续针对性测试日志保留，最终精确结果见 FINAL_REPORT。

原始训练指标、配置、源代码 SHA、缓存身份、Arena 逐局/逐墙记录、性能基准与最终 checkpoint manifest 均随最终发布保存到 `reports/round4/raw`、`artifacts/round4`。完整本机训练优化器快照仍留在 `runs/final_v3`，大缓存留在 `data/final_v3`。最终文件索引与 Git 发布身份以 `ARTIFACT_MANIFEST.json` 和 `FINAL_REPORT.md` 为准。
