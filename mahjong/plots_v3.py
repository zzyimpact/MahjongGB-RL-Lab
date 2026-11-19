"""Create publication-ready curves directly from saved machine records."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plots(root,output,final_arena=None):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    main=['D_retrained','V3','D_regularized','V3_regularized','D_legacy','V3_legacy'];fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name in main:
        p=root/name/'history.json'
        if not p.exists():continue
        h=json.loads(p.read_text());ep=[r['epoch'] for r in h]
        ax[0,0].plot(ep,[r['train_ce'] for r in h],label=name)
        ax[0,1].plot(ep,[r['val']['nonforced_loss'] for r in h],label=name)
        ax[1,0].plot(ep,[100*r['val']['metrics']['play']['accuracy'] for r in h],label=name)
        ax[1,1].plot(ep,[r['lr'] for r in h],label=name)
    for a,title,y in zip(ax.flat,['Training decisions','Held-out validation walls','Validation discard prediction','Learning-rate schedule'],['Nonforced CE','Nonforced CE','Play accuracy (%)','Learning rate']):
        a.set(title=title,xlabel='Epoch',ylabel=y);a.grid(alpha=.2);a.legend(fontsize=8)
    fig.savefig(output/'bc_learning_curves.png');fig.savefig(output/'bc_learning_curves.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name in ['V3','V3_no_quality','V3_no_context','V3_relu','D_retrained','D_constant']:
        p=root/name/'history.json'
        if not p.exists():continue
        h=json.loads(p.read_text());ep=[r['epoch'] for r in h]
        ax[0].plot(ep,[r['val']['nonforced_loss'] for r in h],label=name)
        ax[1].plot(ep,[100*r['val']['metrics']['play']['accuracy'] for r in h],label=name)
    for a,y in zip(ax,['Nonforced validation CE','Validation play accuracy (%)']):a.set(xlabel='Epoch',ylabel=y);a.grid(alpha=.2);a.legend(fontsize=8)
    fig.savefig(output/'bc_ablations.png');plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');found=False
    for directory in sorted(root.glob('PPO*')):
        if 'smoke' in directory.name.lower() or not (directory/'metrics.jsonl').exists():continue
        h=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        if not h:continue
        found=True;steps=[r['counters']['completed_episodes'] for r in h]
        for a,key in [(ax[0,0],'shared/ppo/bc_reference_kl_nonforced'),(ax[0,1],'shared/ppo/value_loss'),(ax[1,0],'shared/ppo/policy_loss')]:a.plot(steps,[r['metrics'][key] for r in h],label=directory.name)
        if (directory/'evaluations.json').exists():
            es=json.loads((directory/'evaluations.json').read_text());cfg=json.loads((directory/'config.json').read_text());scale=cfg['actors']*cfg['episodes_per_actor']
            ax[1,1].plot([e['iteration']*scale for e in es],[e['score']['mean'] for e in es],marker='o',label=directory.name)
    if found:
        for a,y in zip(ax.flat,['KL(BC || current), nonforced','Value loss','PPO actor loss','Development 5-opponent mean reward']):
            a.set(xlabel='Completed training games',ylabel=y);a.grid(alpha=.2);a.legend(fontsize=8)
        fig.savefig(output/'ppo_learning_curves.png');fig.savefig(output/'ppo_learning_curves.pdf')
    plt.close(fig)
    if final_arena:
        r=json.loads(Path(final_arena).read_text());scores=r['aggregate']['scores'];names=list(scores);mean=np.array([scores[n]['mean'] for n in names]);ci=np.array([scores[n]['ci95'] for n in names])
        fig,a=plt.subplots(figsize=(9,max(4,len(names)*.5)),layout='constrained');yy=np.arange(len(names))
        a.errorbar(mean,yy,xerr=[mean-ci[:,0],ci[:,1]-mean],fmt='o',capsize=3);a.set_yticks(yy,labels=names);a.axvline(0,color='gray',lw=.7)
        a.set(xlabel='Mean normalized reward (score / 32), five frozen opponents equally weighted',title='Independent final arena: wall-cluster bootstrap 95% CI');a.grid(axis='x',alpha=.2)
        fig.savefig(output/'final_arena.png');fig.savefig(output/'final_arena.pdf');plt.close(fig)
        followup=root/'supplementary_confirmation.json'
        if followup.exists():
            extra=json.loads(followup.read_text());old=r['aggregate']['paired_differences']['V3_legacy_e01 - PPO_extended_last']
            values=[dict(mean=-old['mean'],ci95=[-old['ci95'][1],-old['ci95'][0]]),extra['statistics']]
            means=np.array([v['mean'] for v in values]);bounds=np.array([v['ci95'] for v in values])
            fig,a=plt.subplots(figsize=(9,3.2),layout='constrained');a.errorbar(means,[1,0],xerr=[means-bounds[:,0],bounds[:,1]-means],fmt='o',capsize=4)
            a.set_yticks([1,0],labels=['Exploratory late snapshot: first 500 walls','Independent confirmation: new 500 walls'])
            a.axvline(0,color='gray',lw=1);a.grid(axis='x',alpha=.2);a.set(xlabel='Paired change in five-opponent reward (PPO last minus BC), 95% CI',title='Late PPO signal did not clear the independent adoption threshold')
            fig.savefig(output/'ppo_confirmation.png');fig.savefig(output/'ppo_confirmation.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/final_v3');p.add_argument('--output',default='reports/round4/figures');p.add_argument('--final-arena');a=p.parse_args();plots(a.root,a.output,a.final_arena)

if __name__=='__main__':main()
