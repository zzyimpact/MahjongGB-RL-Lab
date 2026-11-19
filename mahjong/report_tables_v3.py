"""Build report tables from completed, sealed experiment files only."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def pct(x):return '—' if x is None else '%.2f%%'%(100*x)
def ci(x):return '%+.4f [%+.4f, %+.4f]'%(x['mean'],*x['ci95'])
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def generate(root,report,release):
    root=Path(root);report=Path(report);release=Path(release)
    selection=read(root/'selection.json');final=read(root/'arena_final/results.json');test=read(root/'test_metrics.json')
    dev=read(root/'development_complete.json');seeds=read(root/'seed_stability.json');roles=selection['roles'];bc=roles['selected_bc'];ppo=roles['selected_ppo']
    alias={'D_retrained':'D_e02','V3':'V3_e03'}
    training=[]
    for p in sorted(root.glob('*/summary.json')):
        summary=read(p)
        if 'best_epoch' not in summary:continue
        name=p.parent.name;history=read(p.parent/'history.json');selected=history[summary['best_epoch']-1];cfg=read(p.parent/'config.json');arena=dev['aggregate']['scores'].get(alias.get(name,name))
        training.append(dict(name=name,summary=summary,config=cfg,best=selected,last=history[-1],development=arena))
    tr_rows=[[r['name'],r['config']['seed'],r['summary']['parameters'],r['summary']['epochs'],r['summary']['best_epoch'],r['summary']['updates'],
              '%.4f'%r['best']['train_ce'],'%.4f'%r['best']['val']['nonforced_loss'],pct(r['best']['val']['metrics']['play']['accuracy']),
              '%.1f'%r['summary']['seconds'],ci(r['development']) if r['development'] else '—'] for r in training]
    tr_table=table(['Run','Seed','参数','实际 epoch','保留 epoch','累计更新','保留 epoch Train NF CE','Val NF CE','Val Play','秒','开发综合奖励 [95% CI]'],tr_rows)
    epochs=[]
    for name in ['D_retrained','V3','D_constant','D_regularized','V3_regularized','D_legacy','V3_legacy']:
        history=read(root/name/'history.json')
        for r in history:
            if r['epoch'] in (3,6,12,20,30):epochs.append([name,r['epoch'],'%.4f'%r['train_ce'],'%.4f'%r['val']['nonforced_loss'],pct(r['val']['metrics']['play']['accuracy']),'%.7f'%r['lr'],'%.3f'%r['mean_grad_norm'],r['gradient_updates']])
    epoch_table=table(['Run','Epoch','Train NF CE','Val NF CE','Val Play','LR','梯度范数（裁剪前均值）','更新'],epochs)
    seed_table=table(['模型族','三 seed 开发均值','seed 标准差','双层 bootstrap 95% CI'],[[n,', '.join('%+.4f'%x for x in r['seed_means']),'%.4f'%r['seed_std'],'[%+.4f, %+.4f]'%tuple(r['hierarchical_ci95'])] for n,r in seeds['families'].items()])
    text='# BC 训练充分性研究\n\n所有表由 `mahjong.report_tables_v3` 读取原始记录生成。CE 默认指非强制决策；全样本 CE 另保存在 JSON。测试指标仅在候选 SHA 封存后读取。\n\n## 配方与选择\n\n'
    text+='主配方使用 AdamW、非强制 batch512、5e-4 学习率、1 epoch warmup、20 epoch cosine 规划、weight decay .01、合法标签平滑 .02、clip5。至少观察12 epoch，验证 CE 改善需超过 .001，patience6。更保守配方联合改变 batch256/lr2e-4/smoothing.1，并在12 epoch衰减至floor；不是单变量因果试验。\n\n'
    text+='D_constant 只去掉主配方 cosine。D_legacy/V3_legacy 使用旧 Adam .001、完整序列1024分批、无平滑/WD、clip10；从随机初始化重新训练。强制行没有有效梯度，因此只对同批非强制行前向，仍除原始完整批大小。全部训练集原始位置、mask、action经重新回放核验；GroupNorm逐样本梯度等价测试通过。不同批 shape/数值内核不承诺与历史运行逐bit相同。\n\n'
    text+='## 全部实际 BC 运行\n\n'+tr_table+'\n累计更新包含保留 checkpoint 之后的观察训练。`checkpoint.pt` 是验证规则选出的较早状态，`last.pt` 和阶段快照另存。BC 不优化无价值监督的 Critic。\n\n## 3/6/12 等阶段\n\n'+epoch_table
    text+='\n![BC learning curves](figures/bc_learning_curves.png)\n\n![Ablations](figures/bc_ablations.png)\n\n## 多 seed\n\n'+seed_table
    text+='\n'+table(['差值','均值','双层95% CI'],[[n,'%+.4f'%r['mean'],'[%+.4f, %+.4f]'%tuple(r['hierarchical_ci95'])] for n,r in seeds['paired'].items()])
    text+='\n仅三个训练 seed；双层bootstrap同时重采样配对训练 seed 和共同墙，不能精确代表所有初始化的分布。没有将三次训练当作新增专家数据。训练与验证差距、早停和开发 Arena 的综合解释见 FINAL_REPORT。未运行的20/30 epoch不填入表中。\n'
    (report/'TRAINING_STUDY.md').write_text(text)
    # PPO totals derive from actual counters, not the planned iteration count.
    rl=[]
    for directory in sorted(root.glob('PPO_V3*')):
        if not (directory/'summary.json').exists():continue
        summary=read(directory/'summary.json');h=[json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()];cfg=read(directory/'config.json')
        post=[r for r in h if not r['warmup']]
        rl.append(dict(name=directory.name,summary=summary,config=cfg,learning_decisions=int(sum(r['metrics']['shared/valid_samples'] for r in h)),
            non_draw_rate=float(np.mean([r['non_draw_rate'] for r in h])),mean_learner_terminal_reward=float(np.mean([x for r in h for x in r['learner_terminal_rewards']])),
            postwarmup_mean={k:float(np.mean([r['metrics'][k] for r in post])) for k in ['shared/ppo/policy_loss','shared/ppo/value_loss','shared/ppo/entropy','shared/ppo/bc_reference_kl_nonforced']},last=h[-1]['metrics']))
    rl_table=table(['实验','KL系数','实际局','全部决策','学习席决策','更新','warmup局','秒','非荒庄率','训练学习席平均reward'],[[r['name'],r['config']['rl']['bc_kl'],r['summary']['completed_episodes'],r['summary']['agent_steps'],r['learning_decisions'],r['summary']['gradient_updates'],r['summary']['warmup_episodes'],'%.1f'%r['summary']['seconds'],pct(r['non_draw_rate']),'%+.6f'%r['mean_learner_terminal_reward']] for r in rl])
    losses=table(['实验','Actor loss','Value loss','Entropy','NF BC-reference KL'],[[r['name']]+['%.6f'%v for v in r['postwarmup_mean'].values()] for r in rl])
    text='# 受控强化学习研究\n\n本轮从新的 V3 BC 出发，未加载第三轮退化 PPO。保留原框架的 sparse collector、分玩家轨迹、GAE、固定 old-logp 与 clipped PPO objective。\n\n'
    text+='每局两学习席轮转，其余两席由冻结 BC/C/D 按 seed 循环；不是四席实时共享策略。Critic 在共享状态输入处 detach，前8 iteration只训练价值头，Actor逐tensor保持不变。Actor更新学习率2e-5，clip .1，entropy .001，2 PPO epoch，batch512，每iteration16局。KL方向为 KL(BC || current)，只在合法动作上归一化，reference冻结。\n\n'
    text+='初始两个控制实验只改变 KL 系数 .2→1.0。它们复用同一组1024训练墙，warmup权重逐tensor相同。两次执行的2048局不能称为2048副独立牌墙。统计学习席reward，不用零和四席均值证明牌力。\n\n## 实际规模\n\n'+rl_table
    text+='\n耗时含周期开发评估、不含最初 BC 基线评估；非荒庄率由每局两学习席终局reward重建。下面是warmup后每iteration指标的简单均值，不是按全体样本重加权的估计。\n\n'+losses
    text+='\n![PPO learning curves](figures/ppo_learning_curves.png)\n\n每16 iteration以64开发墙、五冻结对手选 `ppo_best.pt`，另外保留 `last.pt`、周期快照、原始BC和包含BC的 `eval_best.pt`。128墙作开发比较，最终选择规则与候选SHA在新的500墙之前封存。\n\n'
    text+='## 独立最终配对结果\n\n'+table(['候选','五对手综合reward [95% CI]'],[[n,ci(final['aggregate']['scores'][n])] for n in [bc,ppo]])
    confirm=read(root/'confirmatory_statistics.json')
    text+='\n'+table(['预注册差值','均值 / 95% CI','三主要比较 Bonferroni 区间'],[[n,ci(r),'[%+.4f, %+.4f]'%tuple(r['bonferroni_family_ci95'])] for n,r in confirm['contrasts'].items() if n.startswith(ppo+' - ')])
    text+='\nPPO结果解释和是否采用见 FINAL_REPORT。未针对所有潜在退化原因做独立因果实验：价值隔离、对手混合、更小更新强度是联合防护，不能把与第三轮差异归功于其中一个组件。当前主pilot的子进程RNG未在中断恢复点保存，故不承诺中断前后逐bit一致；可选episode独立seed模式有单元测试，但未冒充这些pilot使用的配置。\n'
    (report/'RL_STUDY.md').write_text(text)
    final_table=table(['模型','Random','Teacher','RL v1','C','D','综合reward [95% CI]'],[[n]+['%+.4f'%final['summary'][n+' vs '+o]['reward']['mean'] for o in ['random','teacher','rl_v1','C','D']]+[ci(r)] for n,r in final['aggregate']['scores'].items()])
    test_table=table(['模型','Test全样本CE','Test非强制CE','All Acc','Play Acc','Nonpass Acc','Nonforced Acc'],[[n,'%.5f'%r['metrics']['loss'],'%.5f'%r['metrics']['nonforced_loss']]+[pct(r['metrics']['metrics'][k]['accuracy']) for k in ['all','play','nonpass','nonforced']] for n,r in test['models'].items()])
    classes=[]
    for n,r in test['models'].items():
        for cat,m in r['metrics']['metrics'].items():classes.append([n,cat,m['correct'],m['samples'],pct(m['accuracy'])])
    paired_table=table(['预注册差值','均值 [95% CI]','三主要比较 Bonferroni 区间'],[[n,ci(r),'[%+.4f, %+.4f]'%tuple(r['bonferroni_family_ci95'])] for n,r in confirm['contrasts'].items()])
    (report/'RESULT_TABLES.md').write_text('# 最终机器记录汇总\n\n## 五冻结对手 / 500独立墙\n\n'+final_table+'\n每模型每对手2000局，四座位先按墙聚合；区间不是把座位当独立样本计算。\n\n'+paired_table+'\n## 封存后 Test\n\n'+test_table+'\n总体准确率含大量强制PASS；只看总体准确率会夸大模型质量。Test仅51副独立墙。少样本 GANG/ANGANG 指标不能代表稳定水平。\n\n'+table(['模型','类别','正确','真实样本数','准确率'],classes))
    deploy=[];runtime=[]
    for directory in sorted(release.iterdir()):
        if not directory.is_dir() or not (directory/'submission_check.json').exists():continue
        check=read(directory/'submission_check.json');run=read(directory/'runtime_check.json');deploy.append([directory.name,check['model']['bytes'],check['code_zip']['bytes'],check['code_zip']['expanded_bytes'],check['single_file']['bytes'],'PASS' if check['passed'] else 'FAIL',check['sha256']['mahjong_model.pt']])
        for kind,r in run['results'].items():runtime.append([directory.name,kind,r['games'],r['requests'],r['illegal_actions'],'%.2f'%r['cold_start_ms'],'%.3f'%r['steady_mean_ms'],'%.3f'%r['steady_max_ms'],r['peak_rss_bytes'],', '.join('%.2f ms'%(x['seconds']*1000) for x in r['cold_replays'])])
    text='# Botzone 发布核验\n\n模型上限 **268435456 bytes**（允许等于）；代码 ZIP、解压后的代码与资源合计、单 `.py` 均必须 **严格小于4000000 bytes**。权重独立，不嵌入代码。\n\n'
    text+=table(['发布目录','模型bytes','ZIP bytes','解压bytes','单py bytes','静态检查','模型SHA-256'],deploy)
    text+='\n静态检查同时核对代码资源清单、源码语法、manifest SHA与运行时绑定的权重SHA。完整资源清单见每个发布目录的 `submission_check.json`。\n\n## 本机完整子进程验证\n\n'
    text+=table(['发布','格式','局数','请求数','非法动作','冷启动ms','稳态均值ms','稳态最大ms','峰值RSS bytes','冷历史重放'],runtime)
    text+='\nZIP和单文件运行在干净临时目录，四席轮换，逐响应一致，冷启动重放一致。稳态均值含PASS/初始化等便宜请求，不能替代非强制完整决策基准；后者见 ARCHITECTURE_AND_FEATURES 与 raw/model_benchmark_final.json。\n\n'
    text+='**真实 Botzone 线上未验证。** Python3.6只是语法解析检查，实际运行是本机Python3.10/PyTorch2.3.1/PyMahjongGB1.4.0。V3需要五类native shanten/有效牌API和GELU，平台旧库缺失时明确报错。模型文件额度不代表运行内存额度。\n\n上传时只选该目录的 `mahjong_bot.py` 作为代码文件；`mahjong_model.pt` 单独放用户存储，默认路径 `data/mahjong_model.pt`。同目录代码绑定其权重SHA，不混用BC/PPO/其他版本。ZIP只作为备用打包格式，不要求用户同时上传多个代码文件。\n'
    (report/'SUBMISSION_CHECK.md').write_text(text)
    result=dict(training=training,ppo=rl,selected=roles,final_scores=final['aggregate']['scores'],confirmatory=confirm,seed_stability=seeds)
    (root/'report_summary.json').write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/final_v3');p.add_argument('--reports',default='reports/round4');p.add_argument('--release',default='artifacts/round4/releases');a=p.parse_args()
    result=generate(a.root,a.reports,a.release);print(json.dumps(dict(training_runs=len(result['training']),selected=result['selected'])))


if __name__=='__main__':main()
