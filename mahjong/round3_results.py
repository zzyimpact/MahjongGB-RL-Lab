"""Sealed test evaluation and CPU deployment benchmark; no checkpoint selection."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from .arena import load_checkpoint,identity
from .train_bc import ShardDataset,evaluate,CATEGORIES
from .features import encode
from .gamecore.engine import GameCore
from .observation import SeatView


def ratio_ci(rows,metric,seed=20261119):
    values=np.asarray([r[metric] for r in rows],dtype=float)
    rng=np.random.RandomState(seed);idx=rng.randint(0,len(values),size=(5000,len(values)))
    sums=values[idx].sum(1);valid=sums[:,1]>0
    ratio=sums[valid,0]/sums[valid,1]
    return np.quantile(ratio,[.025,.975]).tolist() if len(ratio) else None

def test_metrics(path,directory,output,selection):
    seal=json.loads(Path(selection).read_text())
    sha=identity(path)['sha256']
    if sha not in seal['frozen_checkpoint_sha256'].values():raise ValueError('checkpoint was not frozen before test')
    model,cfg=load_checkpoint(path);feature=cfg.get('feature_set','basic')
    ds=ShardDataset([directory],'test',feature);device='mps' if torch.backends.mps.is_available() else 'cpu';model.to(device)
    result=evaluate(model,ds.batches(1024,device),device)
    gids=np.load(Path(directory)/'test_game_id.npy');manifest=ds.manifest
    game_map={g['game_id']:g for g in manifest['games']}
    original_walls={g['wall_id'] for g in manifest['games'] if g['game_id']<3000}
    metrics=['all','nonpass','nonforced']+[k for k,_,_ in CATEGORIES]
    wall_rows={};cursor=0
    with torch.inference_mode():
        for obs,mask,action in ds.batches(1024,device):
            logits,_=model(obs);hit=(logits.masked_fill(~mask,-1e9).argmax(-1)==action).cpu().numpy()
            act=action.cpu().numpy();nf=(mask.sum(-1)>1).cpu().numpy()
            selections={'all':np.ones(len(act),bool),'nonpass':act!=0,'nonforced':nf}
            selections.update({k:(act>=lo)&(act<hi) for k,lo,hi in CATEGORIES})
            group=np.asarray([game_map[int(g)]['wall_id'] for g in gids[cursor:cursor+len(act)]]);cursor+=len(act)
            for wall in set(group):
                row=wall_rows.setdefault(wall,{k:[0,0] for k in metrics});wsel=group==wall
                for k,sel in selections.items():row[k][0]+=int((hit&sel&wsel).sum());row[k][1]+=int((sel&wsel).sum())
    rows=list(wall_rows.values());clean=[r for w,r in wall_rows.items() if w not in original_walls]
    result['wall_cluster_ci95']={k:ratio_ci(rows,k) for k in metrics}
    result['walls']=len(rows)
    result['legacy_unseen']={'walls':len(clean),'metrics':{}}
    for k in metrics:
        h,n=np.asarray([r[k] for r in clean]).sum(0)
        result['legacy_unseen']['metrics'][k]=dict(correct=int(h),samples=int(n),accuracy=float(h/n) if n else None,ci95=ratio_ci(clean,k))
    result.update(identity=identity(path),selection_sha256=hashlib.sha256(Path(selection).read_bytes()).hexdigest(),wall_metrics=wall_rows)
    Path(output).write_text(json.dumps(result,indent=2))

def benchmark(path,output):
    torch.set_num_threads(1);model,cfg=load_checkpoint(path)
    core=GameCore();core.r.seed(87654321);core.reset();v=SeatView(0)
    for event in core.agents[0].events:v.request2obs(event)
    feature=cfg.get('feature_set','basic');a=encode(v,feature);obs=torch.from_numpy(a['obs']).unsqueeze(0)
    with torch.inference_mode():
        for _ in range(30):model(obs)
        times=[]
        for _ in range(1000):
            start=time.perf_counter();model(obs);times.append((time.perf_counter()-start)*1000)
    start=time.perf_counter()
    for _ in range(1000):encode(v,feature)
    result=dict(identity=identity(path),parameters=sum(p.numel() for p in model.parameters()),
                encoding_mean_ms=(time.perf_counter()-start), # 1000 repetitions, seconds -> ms/iteration cancel.
                cpu_forward_mean_ms=float(np.mean(times)),cpu_forward_p95_ms=float(np.quantile(times,.95)),
                cpu_forward_max_ms=max(times),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                repetitions=1000,threads=1)
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['test','benchmark'])
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--selection',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('output exists')
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    if a.mode=='test':
        if not a.data or not a.selection:p.error('test requires --data and --selection')
        test_metrics(a.checkpoint,a.data,a.output,a.selection)
    else:benchmark(a.checkpoint,a.output)

if __name__=='__main__':main()
