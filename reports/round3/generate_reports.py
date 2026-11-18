"""Render round-3 Markdown directly from the archived machine-readable evidence."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent; RAW=R/'raw'
def read(path):return json.loads((RAW/path).read_text())
def pct(x):return '—' if x is None else '%.2f%%'%(100*x)
def ci(x):return '%+.4f [%+.4f, %+.4f]'%(x['mean'],*x['ci95'])
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
models={m:read('test_metrics/'+m+'.json') for m in ['A','B','C','D','A0','legacy_bc']}
hist={m:read(m+'/bc_history.json') for m in ['A','B','C','D','A0']}
sums={m:read(m+'/summary.json') for m in hist}
bench={m:read('benchmarks/'+m+'.json') for m in models}
arena=read('arena_final/results.json');ppo=read('arena_ppo_final/results.json');pa=read('PPO_C/analysis.json')
manifest=read('split_manifest.json');effects=read('classification_effects.json');interaction=read('arena_interaction.json')
feature={'A':'Basic','B':'Enhanced','C':'Basic','D':'Enhanced','A0':'Basic'}
main_table=table(['模型','特征/网络','参数量','Test Play Acc','Test Nonpass Acc','Teacher 奖励 [95% CI]','训练秒'],[
 [m,feature[m]+'/'+('ResNet' if m in 'CD' else 'CNN'),sums[m]['parameters'],pct(models[m]['metrics']['play']['accuracy']),pct(models[m]['metrics']['nonpass']['accuracy']),ci(arena['summary'][m+' vs teacher']['reward']),'%.2f'%sums[m]['seconds']] for m in 'ABCD'])
metrics_table=table(['模型','最佳 epoch','Train CE（该 epoch）','Val CE','Test CE','Overall Acc','Nonforced Acc','总更新数'],[
 [m,min(hist[m],key=lambda x:x['val']['loss'])['epoch'],'%.5f'%min(hist[m],key=lambda x:x['val']['loss'])['train_loss'],'%.5f'%sums[m]['best_val_loss'],'%.5f'%models[m]['loss'],pct(models[m]['accuracy']),pct(models[m]['metrics']['nonforced']['accuracy']),sums[m]['gradient_updates']] for m in 'ABCD'])
category_table=table(['动作','测试样本数','A','B','C','D'],[[k,models['A']['metrics'][k]['samples']]+[pct(models[m]['metrics'][k]['accuracy']) for m in 'ABCD'] for k in ['pass','hu','play','chi','peng','gang','angang','bugang','nonpass','nonforced']])
arena_table=table(['模型','Random','Teacher','历史 RL v1'],[[m]+[ci(arena['summary'][m+' vs '+o]['reward']) for o in ['random','teacher','old_rl']] for m in ['A0','legacy_bc','A','B','C','D']])
perf_table=table(['模型','CPU forward ms','编码 ms','独立进程峰值 MiB','Checkpoint bytes','训练样本/秒（epoch 平均）'],[[m,'%.3f'%bench[m]['cpu_forward_mean_ms'],'%.3f'%bench[m]['encoding_mean_ms'],'%.1f'%(bench[m]['process_peak_rss_bytes']/1024**2),sums[m]['checkpoint_bytes'],'%.0f'%(sum(x['samples_per_second'] for x in hist[m])/len(hist[m]))] for m in 'ABCD'])
diff_table=table(['对比 / Teacher','配对奖励差 [95% CI]'],[[x,ci(arena['paired_differences'][x+' vs teacher'])] for x in ['B - A','C - A','D - B','D - C']]+[['(D-B)-(C-A)',ci(interaction['teacher'])]])
class_ci=table(['模型','出牌准确率','按墙 bootstrap 95% CI','非 PASS 准确率','按墙 bootstrap 95% CI'],[[m,pct(models[m]['metrics']['play']['accuracy']),' – '.join(map(pct,models[m]['wall_cluster_ci95']['play'])),pct(models[m]['metrics']['nonpass']['accuracy']),' – '.join(map(pct,models[m]['wall_cluster_ci95']['nonpass']))] for m in 'ABCD'])
seed_section='第二训练 seed 的补充结果见后续生成的 `raw/arena_seed2/`；它不改变主实验预先选定的 C。\n'
if (RAW/'arena_seed2/results.json').exists():
 sr=read('arena_seed2/results.json')
 seed_section='补充试验固定相同超参数，仅将训练 seed 改为 20261120；C/D 各有两个 seed。该试验是在看到主实验结果后追加，使用新的 300 墙（9200000–9200299），用于检验稳定性，不重新选择主实验模型。第二 seed 测试 Play：C2=55.95%、D2=65.77%。新 Arena 中 D2 在三类对手下均分都高于 C2，说明对战排名随初始化变化，不能认定 C 普遍更强；这不改变主实验预先选择。两个 seed 仍不足以精确估计初始化方差。\n\n'+table(['模型/训练 seed','Random','Teacher','历史 RL'],[[m]+[ci(sr['summary'][m+' vs '+o]['reward']) for o in ['random','teacher','old_rl']] for m in ['C1','D1','C2','D2']])+'\n'
abl='''# 第三轮：特征 × 容量消融

所有数字来自 `raw/`，不是历史报告估算。主训练 seed=20261119，牌墙 split seed=20261119。数据仅为全量 IJCAI 2026 final2，不混入 Teacher，不加候选动作提示或奖励标签。

## 受控协议

训练 1,815,376 条、验证 239,322 条、测试 229,832 条；410/51/51 个独立牌墙。每模型从随机初始化完整训练 3 epochs，batch=1024、Adam lr=0.001、梯度裁剪上限 10，同一随机样本顺序，5,319 更新，MPS、float32 计算。强制决策保留，交叉熵为零；不存在通过强制 PASS 构造的额外梯度。Critic 无价值监督，BC 不更新其参数。

最小验证 CE 选择 checkpoint；dev Arena 的 Teacher 均分选择 C；然后封存 SHA 并一次性读取 test/final Arena。测试集不用于早停、调参或 checkpoint 选择。四模型固定训练预算，不声称都训练至收敛。

## 四模型主结果

'''+main_table+'\n\n'+metrics_table+'\n\n'+class_ci+'''

分类 CI 按 51 个测试牌墙重采样，5000 次，以重采样后的正确数/样本数计算比例；不是把重复回放当独立样本。对局 CI 按 300 墙聚类。均为未校正多重比较的边际 95% CI。

## 分动作指标与真实样本数

'''+category_table+'''

CHI/PENG/HU 等均按标签类别统计；非强制=合法动作数>1。HU/部分小类别可能全对；bootstrap 退化为 [1,1] 不证明总体成功概率为 100%。ANGANG 等样本很少且集中在少数牌墙，不能据此宣称强能力。逐类正确数、样本数、CI 均在 `raw/test_metrics/*.json`。

## 真实对战与配对效应

'''+arena_table+'\n\n'+diff_table+'''

每模型每对手 300 墙 × 4 座 = 1200 场；6 个候选（包括 A0、历史 BC）×3 对手，共 21600 场。奖励=原始国标分/32；原始分差、胜负荒庄率、标准误和逐墙记录均在 `raw/arena_final/results.json`，逐局在 `games.jsonl.gz`。Random 在每个墙/座位/玩家使用固定独立 RNG 流；同墙仍会因模型动作不同而产生不同轨迹。

- **修复 BC**：A 的出牌准确率 52.48%，同数据同 seed 同预算冻结 Backbone 的 A0 为 30.35%，差 +22.13 个百分点。A0 是显式 bug 对照，绝非新训练默认。A 的骨干、隐藏层、Actor 全部有实测权重差；Critic 差值为零。
- **历史 BC v1**：它训练过前 3000 局，和本轮 test 的 16 墙重叠。公平历史分类比较仅取其未见的 35 墙：旧 BC Play 30.26%，A 52.44%，差 +22.18 个百分点；仍存在旧/新数据量、混合来源、预算差异。因此因果归因以 A0 对照为主。独立 Arena 中 A 对 Teacher 比旧 BC 提升约 +0.1482；比 A0 提升 +0.1595。
- **特征**：B−A 的 Play 提升约 5.75 个百分点；D−C 约 7.61 个百分点。对战收益依赖对手；B 对 Random 反而较弱，B−A 对 Teacher 的 CI 跨零。
- **容量**：C−A、D−B 的 Play 分别约 +5.00、+6.86 个百分点；两者对 Teacher 的配对奖励 CI 均高于零。此预算下 950k 量级有收益，但不意味着更大一定更好。
- **交互**：分类差中差约 +1.86 个百分点（详见 `raw/classification_effects.json`）；Teacher 奖励差中差 +0.0060，CI 跨零，没有可靠的对战协同证据。
- **谁最好**：C 在 Random、历史 RL 下均分最高；D 在 Teacher 下均分最高，但 D−C 对 Teacher 的 CI 跨零。保留开发集预先选定的 C 为正式候选，D 为有竞争力的备选，不依据最终测试更换 checkpoint。

## 训练时间、推理和产物

'''+perf_table+'''

CPU 测量为 1 线程、预热后 1000 次单状态前向；RSS 包含 Python/PyTorch 运行时，不是裸权重大小。部分基准与后台训练/评估并行，因此不是设备理论极限。所有主 checkpoint 位于 `artifacts/round3/{A,B,C,D}/checkpoint.pt`（仓库根目录）；原始本机路径为 `runs/codex_v2/{A,B,C,D}/checkpoint.pt`。参数量精确计数，文件大小含优化器和 RNG 状态。SHA-256 见 `ARTIFACT_MANIFEST.json`。

## 随机种子重复

'''+seed_section
(R/'ABLATION_RESULTS.md').write_text(abl)
# Main report
split_table=table(['集合','牌墙','局数','决策样本','非强制比例'],[[s,manifest['summary'][s]['walls'],manifest['summary'][s]['games'],manifest['summary'][s]['samples'],pct(manifest['summary'][s]['nonforced_fraction'])] for s in ['train','val','test']])
ppo_table=table(['模型','Random','Teacher','历史 RL'],[[m]+[ci(ppo['summary'][m+' vs '+o]['reward']) for o in ['random','teacher','old_rl']] for m in ['BC','PPO_best','PPO_last']])
report='''# ROUND3_REPORT — 国标麻将训练修复、消融与可靠评估

## 完成摘要与证据等级

本轮实际修复 BC 优化器与按局切分泄漏，重建全量数据，训练 A/B/C/D 与冻结骨干 A0 对照，执行复式 Arena、受控 PPO、部署测试，并保存原始结果、权重和复现入口。**事实来自代码/测试，性能结论来自本轮真实训练和对战；没有声称超过未提供的课程官方 Baseline。**

接管时本地 main=`a4aaa23`，比 origin/main=`1227e4a8ad9144976f4537176a793f01db06d747` 超前一个审计提交，工作区干净。创建 `codex/mahjong-v2-ablation`，完整保留审计历史。初始麻将测试 5 passed。V1 权重 SHA 在 `raw/environment.json`，原 checkpoint、历史日志与旧提交目录保持不变。最终交付 commit 请以 Git 历史和最终回复为准；源代码里程碑 SHA 见 `SOURCE_COMMIT.txt`。

## BC 修复

原实现只把 `model.actor.parameters()` 交给 Adam，15,275 个参数；现在默认优化全部非 Critic 参数。测试证明骨干/隐藏层有非零梯度，优化步骤后骨干和 Actor 实际变化，Critic 保持不变；全模型 strict 恢复并可进入真实 PPO 价值更新。Masked CE 与仅截取合法 logits 的 CE 一致；单合法动作 CE 和 logits 梯度严格为零。每 epoch 保存真实梯度范数、loss、更新数、验证分类计数，原子保存最佳 checkpoint。测试读取是单独封存后的命令。

## 数据、牌墙与缓存

原始数据：`data/ijcai2026/final2_all.jsonl.gz`，来源为 Dannibal/ijcai-mahjong-corpora-2026（本轮核对上游 README 的 license: mit 声明并保存原文；使用本地现存原始数据，未新增外部训练数据）。12,288 局全部成功重建，2,284,530 决策样本，0 非法标签、0 被丢弃回放。每座位通过部署 Session 重建，复合响应必须逐字匹配回放。

`initdata.srand` 与 INIT display.srand 全量一致；每个 srand 对应唯一完整有序 136 张牌墙，512 个 srand 与 512 个 SHA-256 牌墙指纹一一对应，每墙 24 局。实际以完整牌墙 hash 分组，固定 seed=20261119，排序后洗牌，410/51/51（约 80/10/10）。即使将来同墙出现不同 srand，仍会合到同一组；srand 对应不同墙则直接报错。四家座位排列均随整墙进入同一集合。

'''+split_table+'''

完整分组、每局来源、SHA、有效样本和动作频数见 `raw/split_manifest.json`。schema-2 缓存必须完成生成，加载时核对每个 game_id 的分组归属，旧 NPZ/game_id%10 缓存不被训练器接受。`mahjong.replay` CLI 已转向新版准备流程。Teacher 生成器也提供按实际墙指纹的三分法；历史 Teacher seeds 0–999 对应 1000 独立墙，与 IJCAI 零重叠，本轮四组不使用 Teacher 训练数据。

跨 split 相同 `(obs, mask, action)` 的**唯一字节样本**数：basic train/val=22、train/test=12、val/test=3；enhanced 三对均为 0。字节相同不是牌墙泄漏的判据。全缓存验证 A/B/C/D 的基础前六通道、mask、label、game_id 完全一致；9 个覆盖三 split 的回放重新重建逐样本比对。float16 缓存无量化损失，因为全部增强特征使用二进制分数；训练和部署输入均为 float32。未使用早期试编码缓存 `data/codex_v2/enhanced/`，正式增强缓存为 `enhanced_v2/`。

## 版本化状态与网络

v1/basic 为 6×4×9，历史 checkpoint 不带配置时仍按此解释。v2/enhanced 为 34×4×9，所有信息来自当前 SeatView。下表所有相对座位顺序均为自己、下家、对家、上家；34 种牌以外 padding 保持零。

| 通道 | 定义 |
| --- | --- |
| 0–5 | 座风、圈风、四层手牌温度计，完全保留 v1 |
| 6–9 | 各家完整弃牌历史计数 /4（被鸣走的历史弃牌仍属于历史） |
| 10–13 | 各家已知副露牌计数 /4；自己暗杠牌合法可见，对手暗杠身份隐藏 |
| 14–17 | 当前待响应弃牌/补杠牌及相对来源 one-hot；自摸/吃碰后弃牌时清空 |
| 18 | SeatView 已公开牌计数 /4，避免弃牌和副露重复计数 |
| 19 | 未见牌估计=(4−公开−自己手牌−自己暗杠)/4，不等同真实牌墙 |
| 20–23 | 四家剩余墙数 /32（初始 21/32，精确二进制表示） |
| 24–27 | 各家暗杠个数 /4，仅计数不暴露对手暗杠身份 |
| 28–31 | 自摸决策、响应弃牌、吃碰后的弃牌、抢补杠响应阶段 |
| 32–33 | 最后一张墙标志、杠后状态/本次杠后自摸标志 |

未加入重型向听、有效进张、未来信息或 Teacher 提示。抢杠通道直接读取当前补杠事件牌，不误用旧 curTile；公开计数直接使用维护好的 shownTiles，自己暗杠单独扣除。训练、replay、Arena、PPO、Botzone 共用同一 encoder。多座位完整对局、吃碰复合动作、冷启动、最后一张、暗杠/补杠、隐藏手牌与未来墙扰动均有测试。旧 response2action 的 CHI 历史接口未用于本轮路径；标签转换使用已验证的 replay.response_to_action，没有为了清理该未使用接口改动裁判。

Tiny CNN 保留 16/16 两层卷积、576→64、235/1 双头；A=55,468，B=59,500 参数。ResNet 为 64 通道 stem、4 个 PreAct 残差块（GroupNorm 8 组、ReLU、每块两层 3×3 卷积）、末端 GroupNorm/ReLU、2304→256、235/1 双头；C=950,316，D=966,444 参数。加载统一检查 feature version、architecture，并 strict=True；没有裁剪输入、补零或 strict=False。

## 消融主结果

'''+main_table+'''

完整 loss、各类样本数、置信区间、吞吐、参数及效应分析见 [ABLATION_RESULTS.md](ABLATION_RESULTS.md)。总体准确率不是核心指标。仅训练 Actor 对照 A0 的独立出牌准确率 30.35%，完整 A 为 52.48%；训练修复效果清晰。

## 冻结复式评估

开发集 100 墙（8000000–8000099）用于模型选择；最终集 300 墙（9000000–9000299）用于一次确认；两者不共享牌墙且均与所有 BC split 比对无重叠。每墙轮换四个候选座位，对面三个是同一冻结策略：Random、Teacher、历史 RL v1。独立 RNG 流、checkpoint SHA/config、逐局分数和墙 hash 全保存。6 个模型/对照共 21,600 最终局，另有 4,800 开发局和 48 冒烟局。

统计单位是整副墙的四座平均；均分标准误用墙均值，CI 用固定 seed 的 5000 次 cluster bootstrap。配对差使用相同墙/座位配置，原始得分差为候选得分减其他三家平均；零和时该差与候选原始分线性相关，不能作为额外独立证据。CI 是边际区间，未做多重比较校正。300 墙仍无法排除几百分之一奖励的小差异；训练初始化不确定性需另看重复 seed。

'''+arena_table+'\n\n'+diff_table+'''

**正式 BC 候选为 C**：它由开发集选出，在最终 Random 和旧 RL 下最高。D 在最终 Teacher 下点估计更高，但两者差异不确定，不把最终集变为新的选择集。C、D 都未证明胜过 Teacher；更未验证课程 Baseline。

## 受控 PPO：真实退化，保护生效

C → PPO，40 iteration、4 actors×2 episodes/iteration、320 局、51,460 环境步、102,868 决策步、1,263 optimizer updates，162.46 秒（含四次周期评估，不含最初基线评估）。PPO 使用原框架 GAE/分座轨迹/old-logp 逻辑，仅新增可选辅助损失 hook。KL 为 `KL(BC || current)`，相同合法 mask，系数 0.02；BC reference 冻结且 detach。符号、方向、非法动作零梯度、强制动作零 KL、极端 logits 稳定性和真实 PPO 梯度更新均通过测试。

rollout seed=42000000 起，初始 policy 权重 SHA 与 C 完全一致。Adam lr=0.0003、3 PPO epochs、minibatch=256、clip=0.2、value系数=0.5、entropy系数=0.01。每 10 iteration 用 50 个固定 development 墙（8100000 起）对 Teacher 评估。把 BC 作为 iteration 0 纳入保护；`eval_best.pt` 始终保留 BC。PPO 后的最佳为 iteration 10，另存 `ppo_best.pt`；iteration 40 为末轮。全部中间权重本地保留。

训练非荒庄率 46.25%；全体座位平均终局奖励=0 是共享自对弈零和恒等式，不能说明牌力。日志每次都有原始终局奖励。按 iteration 简单平均：Actor loss=-0.01126、Value loss=0.03683、reference KL=0.13559；末轮 reference KL=0.16615，old-policy approx KL=0.01211。完整粒度在 `raw/PPO_C/metrics.jsonl`。

独立 PPO 最终集（9100000–9100299）重新对 BC、最佳后 PPO、末轮 PPO 进行相同三对手/四座评估，共 10,800 局：

'''+ppo_table+'''

末轮−BC 对 Teacher 的配对差 -0.05372，95% CI [-0.10188,-0.00401]；对 Random -0.07576、对旧 RL -0.11188，区间也均低于零。最佳后 PPO 对 Random/旧 RL 也显著退化，对 Teacher 的差异不确定。**不部署这轮 PPO；最终候选保持 BC C。** KL 锚定 0.02 并未阻止退化，不能据此声称 PPO 改善。可能与稀疏奖励、共享自博弈分布、随机 Critic 对共享骨干的影响有关，这是待检验的解释，不是已证实的因果结论。

## 第二训练 seed

'''+seed_section+'''

## 部署与测试

C 和增强 D 均已导出至新的 `submissions/mahjong_v2_C` / `mahjong_v2_D`，并纳入 `artifacts/round3/`。每个导出各通过 3 项真实子进程测试：Python 3.6 **语法**检查、完整三局协议与合法性、冷启动历史重建。C：515 请求、冷启动历史 174 轮；D：503 请求、历史 169 轮。详细延迟见 `raw/deploy_C_tests.txt` / `raw/deploy_D_tests.txt`，本机最大请求耗时均低于 1 秒。CPU 单状态 ResNet 前向约 0.7 ms，独立进程峰值约 200 MiB（包含运行时）。

这不是实际 Botzone 平台测试，也未验证真实 Python 3.6/旧 PyTorch 二进制环境；平台实际库版本、内存/冷启动限制仍需用户提供目标环境或登录后测试。本轮未登录账号、未提交比赛。

麻将最终测试与核心框架回归记录在 `raw/`。旧规则审计重跑 12/12 通过，PPO GAE/奖励时序/old-logp 检查写入新文件，未覆盖历史审计结果。全框架尝试存在 4 项外部 FDL 相关失败：该仓库未含 `fdl` 包和 `rl_examples/.../generated`，并非麻将变更引入；排除这四个明确不具备前提的测试后 49 passed、3 skipped。没有修改断言来制造全绿，完整失败输出保留。

## 算力与研究问题 Q1–Q10

1. **Q1**：优化器修复的同协议出牌准确率提升 +22.13 pp，Teacher 配对奖励 +0.1595 [0.1184,0.2009]；历史 BC 的干净 35 墙比较提升 +22.18 pp，但其旧数据/预算不同。
2. **Q2**：增强特征对分类明确有益；小 CNN 的 Teacher 对战提升未达明确证据，Random 下还有下降，不能用分类改善替代牌力结论。
3. **Q3**：此预算下约 950k ResNet 比 tiny 在两种输入下均有分类及 Teacher 对战收益；这只是两个容量点。
4. **Q4**：分类有小的正交互趋势，但其配对区间也跨零；实际对战差中差区间跨零，没有可靠协同结论。
5. **Q5**：C 在 Random/旧 RL 最好，D 在 Teacher 点估计最好，D−C 对 Teacher 不确定；正式预选候选保留 C。
6. **Q6**：冻结骨干的确定性错误已消除；现在证据指向训练目标/分布和有限独立牌墙（512 墙，而非 12288 独立局）的限制。分类最高 D 并非所有对手下最强，纯自博弈 PPO 退化。数据多样性、决策目标与算法稳定性更值得优先研究；没有单一瓶颈的因果证明。
7. **Q7**：不建议直接跳到 10M。可以在更多独立墙、更多种子和稳定训练下研究 3M，但本轮没有证明必要；先解决泛化与 PPO 退化。
8. **Q8**：本机 Apple M5 Pro、15 CPU/24 GiB、MPS 足够本轮全量训练和数万场评估。四组主 BC 累计约 512 秒，最终 BC Arena 174 秒，PPO 162 秒；无需以模型参数量为由申请远程 GPU。更大规模/更多 seed 时再按实测吞吐决定。
9. **Q9**：PPO 没有改善，末轮在三个冻结对手下均有退化证据；最佳后 PPO 也不如原 BC，保护保留原 BC。
10. **Q10**：还需拿到课程官方 Baseline 代码/权重、确切对战与计分协议；锁定新墙做公平比较；确认 Botzone Python/PyTorch/资源条件并实际上传验证。不能把自建 Teacher 或本地 ZIP 测试当作上述证据。

## 剩余风险与下一轮

主消融每格只有一个训练 seed；C/D 的补充重复仍仅两个。测试独立墙只有 51；重复局在墙内强相关。只训练 3 epochs，不能区分更长预算下容量的收益或过拟合。自博弈 PPO 未引入固定对手池；没有做 KL 系数、value warm-up 等额外调参。下一轮优先增加独立牌墙/对手分布，建立稳定的 BC anchor/价值学习或对手池对照，再考虑增大模型。所有未完成/未验证项都不标记为已达成。
'''
(R/'ROUND3_REPORT.md').write_text(report)
print('Generated reports')
