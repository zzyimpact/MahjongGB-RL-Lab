"""Measured local feature/inference costs and layer-level parameter/MAC counts."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .arena import load_checkpoint,identity
from .model import input_keys
from .features import encode
from .features_v3 import hand_shapes
from .env import Environment


def benchmark(path,n=300):
    torch.set_num_threads(1);model,cfg=load_checkpoint(path);env=Environment(feature_set=cfg.get('feature_set','basic'))
    env.reset({'seed':160000000});views=[];rng=np.random.RandomState(33)
    # Store real decision states across multiple hands, before acting.
    import copy
    seed=160000000
    while len(views)<n:
        if env.core.done:seed+=1;env.reset({'seed':seed})
        commands={}
        from tensordict import TensorDict
        for seat in env.core.obs:
            if len(env.views[seat].valid)>1:views.append(copy.deepcopy(env.views[seat]))
            commands[str(seat)]=TensorDict({'action':torch.tensor(int(rng.choice(env.views[seat].valid)))},[])
        env.step(commands)
    views=views[:n];macs={};handles=[]
    def count(name):
        def hook(layer,args,out):
            if isinstance(layer,nn.Conv2d):macs[name]=out.numel()*layer.in_channels//layer.groups*layer.kernel_size[0]*layer.kernel_size[1]
            elif isinstance(layer,nn.Linear):macs[name]=out.numel()*layer.in_features
        return hook
    for name,layer in model.named_modules():
        if isinstance(layer,(nn.Conv2d,nn.Linear)):handles.append(layer.register_forward_hook(count(name)))
    first=encode(views[0],cfg.get('feature_set','basic'));args=[torch.from_numpy(first[k]).float().unsqueeze(0) for k in input_keys(cfg)]
    with torch.inference_mode():model(*args)
    for h in handles:h.remove()
    hand_shapes.cache_clear();feature=[];forward=[];total=[]
    with torch.inference_mode():
        for v in views:
            start=time.perf_counter();a=encode(v,cfg.get('feature_set','basic'));f=time.perf_counter()
            args=[torch.from_numpy(a[k]).float().unsqueeze(0) for k in input_keys(cfg)]
            logits,_=model(*args);logits.masked_fill(~torch.from_numpy(a['action_mask']),-1e9).argmax().item();end=time.perf_counter()
            feature.append((f-start)*1000);forward.append((end-f)*1000);total.append((end-start)*1000)
    def stats(x):return dict(mean_ms=float(np.mean(x)),median_ms=float(np.median(x)),p95_ms=float(np.quantile(x,.95)),max_ms=float(max(x)))
    groups={name:sum(p.numel() for p in child.parameters()) for name,child in model.named_children()}
    return dict(identity=identity(path),parameters=sum(groups.values()),module_parameters=groups,layer_macs=macs,total_macs=sum(macs.values()),
        approximate_flops=2*sum(macs.values()),flops_scope='conv/linear multiply-adds only; excludes activations, normalization, embedding, feature encoding',
        samples=n,feature=stats(feature),forward_including_tensor_conversion_and_mask=stats(forward),full_decision=stats(total),shanten_cache=hand_shapes.cache_info()._asdict())


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result={name:benchmark(path) for name,path in (x.split('=',1) for x in a.checkpoint)}
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:{x:v[x] for x in ['parameters','total_macs','feature','full_decision']} for k,v in result.items()}))

if __name__=='__main__':main()
