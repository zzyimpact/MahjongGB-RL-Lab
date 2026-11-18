"""One-command reproduction of the primary round-3 protocol in new directories."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .arena import identity


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path('data/ijcai2026/final2_all.jsonl.gz'))
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='mps');p.add_argument('--workers',type=int,default=8)
    p.add_argument('--ppo-iterations',type=int,default=40)
    p.add_argument('--old-bc',type=Path,default=Path('runs/bc_v1/checkpoint.pt'))
    p.add_argument('--old-rl',type=Path,default=Path('runs/rl_v1/checkpoint.pt'))
    a=p.parse_args()
    if a.output.exists() or a.data_root.exists():p.error('Use new data and output directories')
    if not all(p.exists() for p in (a.input,a.old_bc,a.old_rl)):p.error('Raw corpus and historical checkpoint paths must exist')
    a.output.mkdir(parents=True);a.data_root.mkdir(parents=True)
    commands=[]
    def run(module,args,log):
        cmd=[sys.executable,'-m',module]+list(map(str,args));commands.append(cmd)
        (a.output/'commands.json').write_text(json.dumps(commands,indent=2))
        print('Running',module,log,flush=True)
        with (a.output/log).open('w') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
    spec=json.loads((Path(__file__).parent/'configs/round3.json').read_text())
    basic=a.data_root/'basic';enhanced=a.data_root/'enhanced_v2'
    prepare=['--input',a.input,'--workers',a.workers]
    run('mahjong.data_v2',prepare+['--output',basic],'prepare_basic.log')
    for n,cfg in spec['experiments'].items():
        if n=='B':run('mahjong.data_v2',prepare+['--output',enhanced,'--feature-set','enhanced'],'prepare_enhanced_v2.log')
        data=basic if cfg['feature_set']=='basic' else enhanced
        run('mahjong.train_bc',['--data',data,'--output',a.output/n,'--feature-set',cfg['feature_set'],
            '--architecture',cfg['architecture'],'--epochs',spec['epochs'],'--batch-size',spec['batch_size'],
            '--lr',spec['learning_rate'],'--seed',spec['training_seed'],'--device',a.device],n+'.log')
    run('mahjong.train_bc',['--data',basic,'--output',a.output/'A0','--actor-only-control','--epochs',3,'--device',a.device],'A0.log')
    run('mahjong.data_audit',['--basic',basic,'--enhanced',enhanced,'--output',a.output/'data_validation.json'],'data_validation.log')
    paths={n:a.output/n/'checkpoint.pt' for n in ['A','B','C','D','A0']};paths['legacy_bc']=a.old_bc
    def arena(paths,directory,seed,walls,purpose):
        args=[]
        for n,path in paths.items():args+=['--candidate',n+'='+str(path)]
        args+=['--opponent','random','--opponent','teacher','--opponent','old_rl='+str(a.old_rl),
               '--walls',walls,'--seed',seed,'--workers',a.workers,'--manifest',basic/'manifest.json',
               '--output',a.output/directory,'--purpose',purpose]
        run('mahjong.arena',args,directory+'.log')
    arena({n:paths[n] for n in 'ABCD'},'arena_dev',8000000,100,'development')
    dev=json.loads((a.output/'arena_dev/results.json').read_text())
    best=max('ABCD',key=lambda n:dev['summary'][n+' vs teacher']['reward']['mean'])
    seal=dict(selected=best,rule=spec['model_selection'],frozen_checkpoint_sha256={n:identity(path)['sha256'] for n,path in paths.items()},test_accessed=False,final_arena_accessed=False)
    (a.output/'selection.json').write_text(json.dumps(seal,indent=2));(a.output/'test_metrics').mkdir()
    for n,path in paths.items():
        run('mahjong.round3_results',['test','--checkpoint',path,'--data',enhanced,'--selection',a.output/'selection.json','--output',a.output/'test_metrics'/(n+'.json')],n+'_test.log')
        run('mahjong.round3_results',['benchmark','--checkpoint',path,'--output',a.output/(n+'_benchmark.json')],n+'_benchmark.log')
    arena(paths,'arena_final',9000000,300,'final')
    run('mahjong.export_bot',['--checkpoint',paths[best],'--output',a.output/'deployment'],'export.log')
    if a.ppo_iterations:
        cfg=json.loads((a.output/best/'config.json').read_text());cfg.update(seed=42000000,actors=4,episodes_per_actor=2,iterations=a.ppo_iterations,device=a.device)
        config=a.output/'ppo_config.json';config.write_text(json.dumps(cfg,indent=2));ppo=a.output/('PPO_'+best)
        run('mahjong.train',['--config',config,'--resume',paths[best],'--output',ppo,'--bc-reference',paths[best],
            '--bc-kl',.02,'--eval-every',10,'--eval-walls',50,'--eval-seed',8100000,'--split-manifest',basic/'manifest.json'],'PPO.log')
        arena({'BC':paths[best],'PPO_last':ppo/'checkpoint.pt','PPO_best':ppo/'ppo_best.pt'},'arena_ppo_final',9100000,300,'final')

if __name__=='__main__':main()
