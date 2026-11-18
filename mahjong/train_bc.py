"""Full-backbone masked BC on versioned wall-disjoint arrays.

Critic is excluded: there is no value target. PPO gets a fresh optimizer.
Forced decisions have exactly zero CE/gradient and are reported separately.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .policy import make_policy

CATEGORIES=[('pass',0,1),('hu',1,2),('play',2,36),('chi',36,99),('peng',99,133),('gang',133,167),('angang',167,201),('bugang',201,235)]

class ShardDataset:
    def __init__(self,sources,split,feature_set='basic'):
        if len(sources)!=1:raise ValueError('Use one explicitly prepared source; implicit source mixing is forbidden')
        self.directory=Path(sources[0])
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        if self.manifest.get('schema')!=2 or not self.manifest.get('complete'):
            raise ValueError('Requires complete schema-2 wall split; regenerate old caches')
        assignments=self.manifest['groups']
        game_map={g['game_id']:g for g in self.manifest['games']}
        if any(assignments.get(g['wall_id'])!=g['split'] for g in game_map.values()):
            raise ValueError('manifest wall group leakage')
        gids=np.load(self.directory/(split+'_game_id.npy'),mmap_mode='r')
        if any(int(g) not in game_map or game_map[int(g)]['split']!=split for g in np.unique(gids)):
            raise ValueError('cached samples disagree with wall split manifest')
        self.feature_set=feature_set
        stored=self.manifest['feature_set']
        if stored!=feature_set and not (stored=='enhanced' and feature_set=='basic'):
            raise ValueError('feature mismatch')
        self.arrays={k:np.load(self.directory/(split+'_'+k+'.npy'),mmap_mode='r') for k in ('obs','mask','action')}
    def __len__(self):return len(self.arrays['action'])
    def batches(self,batch_size,device,shuffle=False,seed=0):
        order=np.random.RandomState(seed).permutation(len(self)) if shuffle else None
        for start in range(0,len(self),batch_size):
            idx=order[start:start+batch_size] if shuffle else slice(start,start+batch_size)
            obs=self.arrays['obs'][idx]
            if self.feature_set=='basic':obs=obs[:,:6]
            yield (torch.tensor(np.array(obs),dtype=torch.float32,device=device),
                   torch.tensor(np.array(self.arrays['mask'][idx]),dtype=torch.bool,device=device),
                   torch.tensor(np.array(self.arrays['action'][idx]),dtype=torch.long,device=device))

def bc_parameters(model):
    params=[p for name,p in model.named_parameters() if not name.startswith('critic.')]
    assert len(params)==len({id(p) for p in params})
    return params

def masked_ce(logits,mask,action):
    if not mask.any(-1).all() or not mask.gather(1,action[:,None]).all():
        raise ValueError('empty mask or illegal BC label')
    return torch.nn.functional.cross_entropy(logits.masked_fill(~mask,-1e9),action)

def evaluate(model,loader,device):
    previous=model.training;model.eval()
    counts={k:[0,0] for k in ['all','nonpass','nonforced']+[c[0] for c in CATEGORIES]}
    loss_sum=0.;nonforced_loss=0.
    with torch.no_grad():
        for obs,mask,action in loader:
            obs,mask,action=obs.to(device),mask.to(device),action.to(device)
            logits,_=model(obs);logits=logits.masked_fill(~mask,-1e9)
            losses=torch.nn.functional.cross_entropy(logits,action,reduction='none')
            hit=logits.argmax(-1)==action;nf=mask.sum(-1)>1
            loss_sum+=float(losses.sum());nonforced_loss+=float(losses[nf].sum())
            selections={'all':torch.ones_like(hit),'nonpass':action!=0,'nonforced':nf}
            selections.update({k:(action>=lo)&(action<hi) for k,lo,hi in CATEGORIES})
            for k,sel in selections.items():
                counts[k][0]+=int((hit&sel).sum());counts[k][1]+=int(sel.sum())
    model.train(previous)
    metrics={k:dict(correct=h,samples=n,accuracy=h/n if n else None) for k,(h,n) in counts.items()}
    return dict(loss=loss_sum/max(1,counts['all'][1]),nonforced_loss=nonforced_loss/max(1,counts['nonforced'][1]),
                accuracy=metrics['all']['accuracy'],samples=counts['all'][1],metrics=metrics,
                accuracy_by_verb={k:metrics[k]['accuracy'] for k,_,_ in CATEGORIES})

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/default.json')
    p.add_argument('--feature-set',choices=['basic','enhanced'],default='basic')
    p.add_argument('--architecture',choices=['tiny_cnn','small_resnet'],default='tiny_cnn')
    p.add_argument('--actor-only-control',action='store_true',help='Explicit historical-bug control, never a production default')
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=20261119);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.output.exists():p.error('output already exists')
    if a.epochs<1 or a.batch_size<1:p.error('epochs and batch size must be positive')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    cfg=json.loads(a.config.read_text());cfg.update(seed=a.seed,feature_set=a.feature_set,
        feature_version='v1' if a.feature_set=='basic' else 'v2',architecture=a.architecture)
    train=ShardDataset(a.data,'train',a.feature_set);val=ShardDataset(a.data,'val',a.feature_set)
    device=torch.device(a.device);policy=make_policy(cfg['hidden'],config=cfg).to(device);model=policy.module
    optimized=list(model.actor.parameters()) if a.actor_only_control else bc_parameters(model)
    optimizer=torch.optim.Adam(optimized,lr=a.lr)
    a.output.mkdir(parents=True)
    run={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};run['data']=list(map(str,a.data))
    run.update(config=cfg,parameters=sum(p.numel() for p in model.parameters()),
               optimized_parameters=sum(p.numel() for p in optimized),
               manifest_sha256=hashlib.sha256((a.data[0]/'manifest.json').read_bytes()).hexdigest())
    (a.output/'config.json').write_text(json.dumps(cfg,indent=2));(a.output/'run.json').write_text(json.dumps(run,indent=2))
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    start=time.monotonic();history=[];best=float('inf');updates=0
    for epoch in range(1,a.epochs+1):
        model.train();seen=0;loss_sum=0.;epoch_start=time.monotonic();grad_norm_sum=0.
        for obs,mask,action in train.batches(a.batch_size,device,True,a.seed+epoch):
            model.zero_grad(set_to_none=True)
            logits,_=model(obs);loss=masked_ce(logits,mask,action);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(optimized,10.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
            optimizer.step();updates+=1;seen+=len(action);loss_sum+=float(loss)*len(action);grad_norm_sum+=float(norm)
        training_seconds=time.monotonic()-epoch_start
        val_metrics=evaluate(model,val.batches(a.batch_size,device),device)
        row=dict(epoch=epoch,train_loss=loss_sum/seen,gradient_updates=updates,
                 mean_grad_norm=grad_norm_sum/((seen+a.batch_size-1)//a.batch_size),
                 train_seconds=training_seconds,samples_per_second=seen/training_seconds,
                 val=val_metrics,elapsed_seconds=time.monotonic()-start)
        history.append(row);print(json.dumps(row),flush=True)
        (a.output/'bc_history.json').write_text(json.dumps(history,indent=2))
        if val_metrics['loss']<best:
            best=val_metrics['loss']
            state=dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},
                optimizer=torch.optim.Adam(policy.parameters(),lr=cfg['ppo']['learning_rate']).state_dict(),
                bc_optimizer=optimizer.state_dict(),config=cfg,iteration=0,
                counters=dict(env_steps=0,agent_steps=0,gradient_updates=0,completed_episodes=0),episode_cursor=0,
                torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                bc=dict(epoch=epoch,gradient_updates=updates,train_samples=len(train),val=val_metrics,run=run))
            torch.save(state,a.output/'checkpoint.tmp');(a.output/'checkpoint.tmp').replace(a.output/'checkpoint.pt')
    changes={k:float((v.detach().cpu()-initial[k]).abs().max()) for k,v in model.state_dict().items()}
    (a.output/'parameter_changes.json').write_text(json.dumps(changes,indent=2))
    # Test is deliberately a separate command, after all checkpoint/model choices are frozen.
    summary=dict(seconds=time.monotonic()-start,best_val_loss=best,gradient_updates=updates,parameters=run['parameters'],
                 checkpoint_bytes=(a.output/'checkpoint.pt').stat().st_size)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
