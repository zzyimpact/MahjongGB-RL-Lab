"""Resumable controlled BC: nonforced sampling, AdamW, warmup/cosine, val-only selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from .model import input_keys
from .policy import make_policy
from .train_bc import CATEGORIES, bc_parameters
from .cache_identity import check as check_feature_cache


class Dataset:
    def __init__(self, root, split, config):
        self.root=Path(root);self.split=split;self.config=config
        check_feature_cache(self.root)
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest.get('schema')!=3 or not self.manifest.get('complete'):raise ValueError('requires complete schema-3 cache')
        self.keys=input_keys(config)
        self.arrays={k:np.load(self.root/(split+'_'+k+'.npy'),mmap_mode='r') for k in self.keys+('mask','action','game_id')}
        gm={g['game_id']:g for g in self.manifest['games']}
        for gid in np.unique(self.arrays['game_id']):
            g=gm[int(gid)]
            if g['split']!=split or self.manifest['groups'][g['wall_id']]!=split:raise ValueError('cache/split mismatch')
        self.counts=self.manifest['summary'][split]
        self.full_index=None
        if split=='train' and config.get('bc',{}).get('sampling')=='full_equivalent':
            path=self.root/'train_full_index.npy';record=json.loads((self.root/'full_sampling.json').read_text())
            if record['manifest_sha256']!=hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest() or record['index_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('full sampling index identity mismatch')
            self.full_index=np.load(path,mmap_mode='r')
            if len(self.full_index)!=self.counts['samples'] or not np.array_equal(self.full_index[self.full_index>=0],np.arange(len(self))):raise ValueError('full sampling index malformed')
    def __len__(self):return len(self.arrays['action'])
    @property
    def training_samples(self):return len(self.full_index) if self.full_index is not None else len(self)
    def training_batches(self,batch_size,device,seed):
        if self.full_index is None:
            for inputs,mask,action in self.batches(batch_size,device,seed):yield inputs,mask,action,len(action)
            return
        order=np.random.RandomState(seed).permutation(len(self.full_index))
        for start in range(0,len(order),batch_size):
            positions=order[start:start+batch_size];ix=self.full_index[positions];ix=ix[ix>=0]
            inputs={k:torch.tensor(np.asarray(self.arrays[k][ix]),device=device,dtype=torch.float32) for k in self.keys}
            if self.config.get('feature_set','basic')=='basic':inputs['obs']=inputs['obs'][:,:6]
            mask=torch.tensor(np.asarray(self.arrays['mask'][ix]),device=device,dtype=torch.bool)
            action=torch.tensor(np.asarray(self.arrays['action'][ix]),device=device,dtype=torch.long)
            yield inputs,mask,action,len(positions)
    def batches(self, batch_size, device, seed=None):
        order=np.random.RandomState(seed).permutation(len(self)) if seed is not None else None
        for start in range(0,len(self),batch_size):
            ix=order[start:start+batch_size] if order is not None else slice(start,start+batch_size)
            inputs={k:torch.tensor(np.asarray(self.arrays[k][ix]),device=device,dtype=torch.float32) for k in self.keys}
            if self.config.get('feature_set','basic')=='basic':inputs['obs']=inputs['obs'][:,:6]
            mask=torch.tensor(np.asarray(self.arrays['mask'][ix]),device=device,dtype=torch.bool)
            action=torch.tensor(np.asarray(self.arrays['action'][ix]),device=device,dtype=torch.long)
            yield inputs,mask,action


def masked_loss(logits, mask, action, smoothing=0):
    if not mask.any(-1).all() or not mask.gather(1,action[:,None]).all():raise ValueError('invalid mask/label')
    logp=torch.log_softmax(logits.masked_fill(~mask,-1e9),-1)
    ce=-logp.gather(1,action[:,None]).squeeze(1)
    smooth=-(logp*mask).sum(-1)/mask.sum(-1)
    return ((1-smoothing)*ce+smoothing*smooth).mean(),ce


def evaluate(model, ds, device, batch_size=1024):
    previous=model.training;model.eval();loss=0.
    counts={k:[0,0] for k in ['all','nonpass','nonforced']+[k for k,_,_ in CATEGORIES]}
    with torch.inference_mode():
        for inputs,mask,action in ds.batches(batch_size,device):
            logits,_=model(*(inputs[k] for k in ds.keys));_,ce=masked_loss(logits,mask,action)
            hit=logits.masked_fill(~mask,-1e9).argmax(-1)==action;loss+=float(ce.sum())
            selections={'all':torch.ones_like(hit),'nonpass':action!=0,'nonforced':mask.sum(-1)>1}
            selections.update({k:(action>=lo)&(action<hi) for k,lo,hi in CATEGORIES})
            for k,sel in selections.items():counts[k][0]+=int((hit&sel).sum());counts[k][1]+=int(sel.sum())
    # Forced masked decisions are exactly correct with zero CE for any model.
    for key,n in ds.counts['forced_action_counts'].items():
        a=int(key);counts['all'][0]+=n;counts['all'][1]+=n
        if a!=0:counts['nonpass'][0]+=n;counts['nonpass'][1]+=n
        for k,lo,hi in CATEGORIES:
            if lo<=a<hi:counts[k][0]+=n;counts[k][1]+=n
    model.train(previous)
    return dict(loss=loss/ds.counts['samples'],nonforced_loss=loss/len(ds),samples=ds.counts['samples'],
                metrics={k:dict(correct=h,samples=n,accuracy=h/n if n else None) for k,(h,n) in counts.items()})


def lr_at(step,total,warmup,base,minimum):
    if step<warmup:return base*(step+1)/max(1,warmup)
    progress=min(1.,(step-warmup)/max(1,total-warmup))
    return minimum+(base-minimum)*.5*(1+math.cos(math.pi*progress))


def atomic_save(state,path):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


def train(config_file,data,output,resume=False):
    cfg=json.loads(Path(config_file).read_text());out=Path(output)
    if out.exists() and not resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    seed=cfg['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    device=torch.device(cfg.get('device','mps'));policy=make_policy(config=cfg).to(device);model=policy.module
    params=bc_parameters(model);spec=cfg['bc']
    optimizer_class={'adamw':torch.optim.AdamW,'adam':torch.optim.Adam}[spec.get('optimizer','adamw')]
    optimizer=optimizer_class(params,lr=spec['lr'],weight_decay=spec['weight_decay'])
    train_ds=Dataset(data,'train',cfg);val_ds=Dataset(data,'val',cfg)
    run=dict(config=cfg,data=str(data),manifest_sha256=hashlib.sha256((Path(data)/'manifest.json').read_bytes()).hexdigest(),
             source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('mahjong').glob('*.py')},parameters=sum(p.numel() for p in model.parameters()))
    if train_ds.full_index is not None:run['full_sampling']=json.loads((Path(data)/'full_sampling.json').read_text())
    history=[];updates=0;best=float('inf');best_epoch=0;start_epoch=1;elapsed=0.
    if resume:
        old=json.loads((out/'run.json').read_text())
        if old['config']!=cfg or old['manifest_sha256']!=run['manifest_sha256']:raise ValueError('resume configuration or data mismatch')
        state=torch.load(out/'last.pt',map_location=device);policy.load_state_dict(state['policy'],strict=True);optimizer.load_state_dict(state['bc_optimizer'])
        history=state['bc']['history'];updates=state['bc']['updates'];best=state['bc']['best_val'];best_epoch=state['bc']['best_epoch'];start_epoch=state['bc']['epoch']+1;elapsed=state['bc']['elapsed_seconds']
        torch.set_rng_state(state['torch_rng'].cpu());random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        if 'mps_rng' in state and device.type=='mps':torch.mps.set_rng_state(state['mps_rng'].cpu())
        # Recover an interruption after last.pt was committed but before the
        # same newly-best epoch reached checkpoint.pt. Never silently select a
        # stale best file after resumption.
        best_path=out/'checkpoint.pt'
        saved_epoch=torch.load(best_path,map_location='cpu')['bc']['epoch'] if best_path.exists() else None
        if saved_epoch!=best_epoch:
            if state['bc']['epoch']!=best_epoch:raise ValueError('Best checkpoint missing or inconsistent with last.pt')
            atomic_save(state,best_path)
    else:
        (out/'run.json').write_text(json.dumps(run,indent=2));(out/'config.json').write_text(json.dumps(cfg,indent=2))
    steps=math.ceil(train_ds.training_samples/spec['batch_size']);total=steps*spec.get('schedule_epochs',spec['epochs']);warmup=steps*spec['warmup_epochs'];begin=time.monotonic()
    for epoch in range(start_epoch,spec['epochs']+1):
        model.train();seen=0;ce_sum=objective_sum=grad_sum=0.;epoch_begin=time.monotonic()
        for inputs,mask,action,denominator in train_ds.training_batches(spec['batch_size'],device,seed+epoch):
            lr=lr_at(updates,total,warmup,spec['lr'],spec['min_lr']) if spec.get('schedule','cosine')=='cosine' else spec['lr']*min(1.,(updates+1)/max(1,warmup))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            if len(action):
                logits,_=model(*(inputs[k] for k in train_ds.keys));loss,ce=masked_loss(logits,mask,action,spec['label_smoothing'])
                (loss*len(action)/denominator).backward()
            else:
                # A fully forced batch still advances Adam's momentum and LR.
                loss=sum(p.sum()*0. for p in params);ce=loss.reshape(1);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,spec['grad_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradients')
            optimizer.step();updates+=1;seen+=len(action);ce_sum+=float(ce.sum());objective_sum+=float(loss)*len(action);grad_sum+=float(norm)
        train_seconds=time.monotonic()-epoch_begin;val=evaluate(model,val_ds,device)
        improved=val['nonforced_loss']<best-spec['min_delta']
        if improved:best=val['nonforced_loss'];best_epoch=epoch
        row=dict(epoch=epoch,train_ce=ce_sum/seen,train_full_ce=ce_sum/train_ds.counts['samples'],objective=objective_sum/seen,val=val,
                 lr=lr,gradient_updates=updates,mean_grad_norm=grad_sum/steps,train_seconds=train_seconds,
                 elapsed_seconds=elapsed+time.monotonic()-begin,best_epoch=best_epoch)
        history.append(row)
        state=dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},config=cfg,
            optimizer=torch.optim.Adam(policy.parameters(),lr=cfg['ppo']['learning_rate']).state_dict(),bc_optimizer=optimizer.state_dict(),
            iteration=0,counters=dict(env_steps=0,agent_steps=0,gradient_updates=0,completed_episodes=0),episode_cursor=0,
            torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            bc=dict(epoch=epoch,history=history,updates=updates,best_val=best,best_epoch=best_epoch,elapsed_seconds=row['elapsed_seconds'],run=run))
        if device.type=='mps':state['mps_rng']=torch.mps.get_rng_state()
        atomic_save(state,out/'last.pt')
        if improved:atomic_save(state,out/'checkpoint.pt')
        if epoch in (3,6,12,20,30) or improved:atomic_save(state,out/('epoch_%03d.pt'%epoch))
        (out/'history.json').write_text(json.dumps(history,indent=2));(out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        print(json.dumps(row),flush=True)
        if epoch>=spec['min_epochs'] and epoch-best_epoch>=spec['patience']:break
    summary=dict(best_epoch=best_epoch,best_val_nonforced_ce=best,epochs=len(history),updates=updates,seconds=history[-1]['elapsed_seconds'],
                 parameters=run['parameters'],checkpoint_bytes=(out/'checkpoint.pt').stat().st_size,stopped_early=len(history)<spec['epochs'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    train(a.config,a.data,a.output,a.resume)

if __name__=='__main__':main()
