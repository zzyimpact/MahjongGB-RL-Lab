# 复现说明

## 环境和依赖

本轮：macOS 26.5.2 arm64、Apple M5 Pro、15 CPU、24 GiB、Python 3.10.21、PyTorch 2.3.1、TensorDict 0.1.0、NumPy 1.26.4、PyMahjongGB 1.4.0，MPS 可用。硬件、历史 checkpoint SHA 和环境检查见 `raw/environment.json`。不要直接安装旧 requirements-lock.txt 中指向作者绝对路径的 editable 行。

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install torch==2.3.1 tensordict==0.1.0 numpy==1.26.4 PyMahjongGB==1.4.0 pytest==9.2.0
.venv/bin/python -m pip install -e ./rl_framework
```

需要可构建 PyMahjongGB 的 C++ 环境。CPU 可运行所有流程，MPS 命令可改 `--device cpu`；跨硬件和 PyTorch 后端不承诺逐 bit 相同，分组/种子/协议相同。

## 原始数据获取

数据集 [Dannibal/ijcai-mahjong-corpora-2026](https://huggingface.co/datasets/Dannibal/ijcai-mahjong-corpora-2026)，上游 README 声明 MIT（本轮保存到 `raw/upstream_README.txt`），原始文件 45,777,973 bytes。本轮重用现有文件，未下载未知含花数据。

```bash
mkdir -p data/ijcai2026
curl -L --fail 'https://huggingface.co/datasets/Dannibal/ijcai-mahjong-corpora-2026/resolve/main/final2_all.jsonl.gz' -o data/ijcai2026/final2_all.jsonl.gz
# 网络受限时可将 huggingface.co 替换为 hf-mirror.com。
.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('data/ijcai2026/final2_all.jsonl.gz')
expected=json.load(open('reports/round3/raw/split_manifest.json'))['source_sha256']
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
print('raw source identity verified')
PY
```

原始数据和内存映射缓存保留本地、不提交 Git。不要覆盖现有 `data/codex_v2` 或 `runs/codex_v2`。下列命令使用全新 `reproduce_v2` 目录。

## 一条命令复现

最终训练/评估配置集中在 `mahjong/configs/round3.json`。以下命令顺序执行全量重建、A→增强特征数据→B/C/D/A0、开发选择、封存测试、最终 Arena、导出、40 iteration PPO 与独立 PPO Arena；每步日志和命令写入新目录。此主流程包含所有核心实验，不包含后加的第二 seed 稳定性扩展。

```bash
.venv/bin/python -m mahjong.round3 \
  --input data/ijcai2026/final2_all.jsonl.gz \
  --data-root data/reproduce_v2 --output runs/reproduce_v2 \
  --device mps --workers 8 --ppo-iterations 40 \
  --old-bc artifacts/round3/references/bc_v1.pt \
  --old-rl artifacts/round3/references/rl_v1.pt
```

该编排入口通过 CLI 解析检查，各个实际步骤已在本轮逐项执行；未再次完整跑一遍一键入口来消耗重复训练预算。PPO 运行期间选择仅使用独立 development 墙，`eval_best.pt` 包括 BC 基线，`ppo_best.pt` 仅比较 PPO 后迭代。

## 单独数据和四组训练

```bash
.venv/bin/python -m mahjong.data_v2 --input data/ijcai2026/final2_all.jsonl.gz --output data/reproduce_v2/basic --workers 6 --seed 20261119
.venv/bin/python -m mahjong.train_bc --data data/reproduce_v2/basic --output runs/reproduce_v2/A --epochs 3 --batch-size 1024 --lr .001 --seed 20261119 --device mps
.venv/bin/python -m mahjong.data_v2 --input data/ijcai2026/final2_all.jsonl.gz --output data/reproduce_v2/enhanced_v2 --feature-set enhanced --workers 6 --seed 20261119
.venv/bin/python -m mahjong.train_bc --data data/reproduce_v2/enhanced_v2 --output runs/reproduce_v2/B --feature-set enhanced --epochs 3 --batch-size 1024 --lr .001 --seed 20261119 --device mps
.venv/bin/python -m mahjong.train_bc --data data/reproduce_v2/basic --output runs/reproduce_v2/C --architecture small_resnet --epochs 3 --batch-size 1024 --lr .001 --seed 20261119 --device mps
.venv/bin/python -m mahjong.train_bc --data data/reproduce_v2/enhanced_v2 --output runs/reproduce_v2/D --feature-set enhanced --architecture small_resnet --epochs 3 --batch-size 1024 --lr .001 --seed 20261119 --device mps
.venv/bin/python -m mahjong.train_bc --data data/reproduce_v2/basic --output runs/reproduce_v2/A0 --actor-only-control --epochs 3 --device mps
.venv/bin/python -m mahjong.data_audit --basic data/reproduce_v2/basic --enhanced data/reproduce_v2/enhanced_v2 --output runs/reproduce_v2/data_validation.json
```

不使用旧 NPZ 缓存；生成过程验证 ordered wall SHA、srand 一致性、完整标签回放，保存全 manifest。Teacher 如需单独实验：`python -m mahjong.bc_data --games 1000 --seed 0 --workers 6 --output data/new_teacher`，其新输出同样包含三分组 manifest。主实验不使用 Teacher 数据混合。

## Arena 与封存 test

对已版本管理的本轮权重，可直接重评，不必重训：

```bash
.venv/bin/python -m mahjong.arena \
 --candidate A=artifacts/round3/A/checkpoint.pt --candidate B=artifacts/round3/B/checkpoint.pt \
 --candidate C=artifacts/round3/C/checkpoint.pt --candidate D=artifacts/round3/D/checkpoint.pt \
 --candidate A0=artifacts/round3/A0/checkpoint.pt --candidate legacy_bc=artifacts/round3/references/bc_v1.pt \
 --opponent random --opponent teacher --opponent old_rl=artifacts/round3/references/rl_v1.pt \
 --walls 300 --seed 9000000 --workers 8 --manifest reports/round3/raw/split_manifest.json \
 --output runs/new_final_arena --purpose final
```

开发选择使用 8000000 起 100 墙，候选只有 A/B/C/D。每个候选使用四个座位，每对手都是其他三家同一冻结策略。`purpose` 是明确记录用途的标签；跨实验种子和墙集合仍需遵守本文件。实际本轮集合间 hash 核查在 `raw/wall_independence.json`。

冻结 checkpoint SHA 后 test 命令如下；本轮所有主模型封存在 `raw/selection.json`，因此可直接用于 versioned checkpoint。新训练模型要用一键入口产生自己的选择封存，不能拿旧 seal 混用。

```bash
.venv/bin/python -m mahjong.round3_results test --checkpoint artifacts/round3/C/checkpoint.pt \
 --data data/reproduce_v2/enhanced_v2 --selection reports/round3/raw/selection.json --output runs/C_test_recheck.json
.venv/bin/python -m mahjong.round3_results benchmark --checkpoint artifacts/round3/C/checkpoint.pt --output runs/C_perf_recheck.json
```

旧模型测试只在其未见的35墙子集作公平历史对照；A0是全新、同协议的冻结骨干对照。普通 test 指标包含 51 墙，不能误称旧 BC 在这 51 墙全部未见过。

## PPO

本轮完整配置在 `raw/ppo_config.json`，训练与参考 KL 配置在 `raw/PPO_C/experiment.json`。

```bash
.venv/bin/python -m mahjong.train --config reports/round3/raw/ppo_config.json \
 --resume artifacts/round3/C/checkpoint.pt --output runs/new_PPO_C \
 --bc-reference artifacts/round3/C/checkpoint.pt --bc-kl .02 \
 --eval-every 10 --eval-walls 50 --eval-seed 8100000 \
 --split-manifest reports/round3/raw/split_manifest.json
.venv/bin/python -m mahjong.arena \
 --candidate BC=artifacts/round3/C/checkpoint.pt \
 --candidate PPO_best=runs/new_PPO_C/ppo_best.pt --candidate PPO_last=runs/new_PPO_C/checkpoint.pt \
 --opponent random --opponent teacher --opponent old_rl=artifacts/round3/references/rl_v1.pt \
 --walls 300 --seed 9100000 --workers 8 --manifest reports/round3/raw/split_manifest.json \
 --output runs/new_PPO_arena --purpose final
```

本轮 `eval_best.pt` 与 C checkpoint SHA 完全一致，说明保留了 BC。已发布的 `artifacts/round3/PPO_C/ppo_best.pt` 是 iteration 10 的最佳**后 PPO**模型，仅作负结果复现，不推荐部署。末轮和各中间 checkpoint 本地保留；均可按命令重新生成。恢复 PPO 时配置必须和 checkpoint 的 feature/architecture 严格匹配；训练脚本不会静默把新模型当旧模型。

## 第二 seed 扩展

```bash
.venv/bin/python -m mahjong.train_bc --data data/reproduce_v2/basic --output runs/reproduce_v2/C_seed2 --architecture small_resnet --epochs 3 --device mps --seed 20261120
.venv/bin/python -m mahjong.train_bc --data data/reproduce_v2/enhanced_v2 --output runs/reproduce_v2/D_seed2 --architecture small_resnet --feature-set enhanced --epochs 3 --device mps --seed 20261120
```

以相同 Arena 参数比较 C1/D1/C2/D2，但 seed=9200000、walls=300、三个冻结对手。第二 seed checkpoint 未加入 Git，已有实际训练 config/history/test/arena 记录与 SHA，重新训练可再生成。结果不用于替换主实验选择。

## 导出与测试

```bash
.venv/bin/python -m mahjong.export_bot --checkpoint artifacts/round3/C/checkpoint.pt --output submissions/new_mahjong_v2_C
BOT_TEST_CHECKPOINT=artifacts/round3/C/checkpoint.pt .venv/bin/python -m pytest mahjong/tests/test_export.py -q -s
BOT_TEST_CHECKPOINT=artifacts/round3/D/checkpoint.pt .venv/bin/python -m pytest mahjong/tests/test_export.py -q -s
.venv/bin/python -m pytest mahjong/tests -q
PYTHONPATH=.:rl_framework/tests .venv/bin/python -m pytest mahjong/tests rl_framework/tests -q \
 -k 'not test_fdl_compiles_and_encodes_without_rl_stack and not test_generated_runtime_does_not_import_parser and not test_loader_rejects_missing_and_modified_artifacts and not test_multiple_fdl_rules_become_separate_tensor_dict_fields'
```

末条排除4项依赖本仓库不存在的 FDL 源码及示例 generated 产物的测试；完整尝试及失败原因存于 `raw/all_tests_correct_path.txt`。实际最终回归为49通过、3跳过、4明确排除。没有假装完整框架全绿。

ZIP 内 source 和独立 CPU 权重需按 Botzone 平台方式放置，默认权重路径 `data/mahjong_model.pt`。支持 `BOT_MODEL_PATH` 本地覆盖。未实际上传 Botzone；Python 3.6 仅静态语法检查，运行验证是在当前 Python/PyTorch 环境。

## 种子、身份和产物

- split/主训练/bootstrap：20261119；补充训练：20261120；bootstrap=5000次。
- 冒烟：7000000–7000003；开发：8000000–8000099；最终BC：9000000–9000299。
- PPO采样：42000000–42000319；PPO开发：8100000–8100049；PPO最终：9100000–9100299。
- seed重复Arena：9200000–9200299。
- 全部上述牌墙与全部BC墙均实测无重叠，见 `raw/wall_independence.json`。
- 源代码里程碑 commit：见 `SOURCE_COMMIT.txt`；完整最终交付以当前 Git commit 为准（避免报告内自引用 SHA）。
- 所有版本管理权重及本地 checkpoint 的路径、bytes、SHA-256：`ARTIFACT_MANIFEST.json`。
- 原始逐局 Arena 以 `raw/*/games.jsonl.gz` 存储；解压为JSONL后可用 `mahjong.arena.summarize` 重算统计。
- 最终发布权重约45MB合计，单文件不到12MB，无需 Git LFS；原始数据、缓存、临时中间权重留在 ignored 目录。
- 本地根路径 `/Users/zzy/Desktop/Coding/rl-lab`，实际运行产物在 `runs/codex_v2/`。

`collect_artifacts.py` 从本轮真实运行目录收集报告证据和合理大小的权重，`generate_reports.py` 从机器记录生成主要报告。重新训练的结果应保存在新报告目录，避免覆盖这次实验。历史 V1 产物 SHA 核验通过，未被本轮覆盖。
