"""Measure BC/PPO policy drift on validation observations, never test labels."""
import argparse
import json
from pathlib import Path
import torch
from .arena import load_checkpoint,identity
from .train_v3 import Dataset
from .ppo_anchor import masked_reference_kl


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',action='append',required=True);p.add_argument('--data',default='data/final_v3');p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1);reference,cfg=load_checkpoint(a.reference);models={n:load_checkpoint(path)[0] for n,path in (v.split('=',1) for v in a.candidate)}
    dataset=Dataset(a.data,'val',cfg);result={n:dict(samples=0,changed=0,kl_sum=0.) for n in models}
    with torch.inference_mode():
        for inputs,mask,unused_labels in dataset.batches(512,'cpu'):
            x=[inputs[k] for k in dataset.keys];logits,_=reference(*x);actions=logits.masked_fill(~mask,-1e9).argmax(-1)
            for name,model in models.items():
                current,_=model(*x);r=result[name];r['samples']+=len(actions);r['changed']+=int((current.masked_fill(~mask,-1e9).argmax(-1)!=actions).sum());r['kl_sum']+=float(masked_reference_kl(current,logits,mask).sum())
    ref_state=reference.state_dict()
    for name,path in (v.split('=',1) for v in a.candidate):
        r=result[name];r['action_change_fraction']=r['changed']/r['samples'];r['mean_forward_kl']=r['kl_sum']/r['samples'];r['identity']=identity(path)
        state=torch.load(path,map_location='cpu');r['actual_optimizer_lr']=[g['lr'] for g in state['optimizer']['param_groups']];r['parameter_delta']={}
        weights=models[name].state_dict()
        for group in ['conv','actor','critic','hidden','discard','context']:
            differences=[(weights[k]-v).float() for k,v in ref_state.items() if k.split('.')[0]==group]
            r['parameter_delta'][group]=dict(changed_values=sum(int((d!=0).sum()) for d in differences),l2=float(torch.sqrt(sum((d*d).sum() for d in differences))))
    Path(a.output).write_text(json.dumps(dict(reference=identity(a.reference),split='validation nonforced observations',candidates=result),indent=2))
    print(json.dumps({k:{x:v[x] for x in ['mean_forward_kl','action_change_fraction','actual_optimizer_lr']} for k,v in result.items()}))

if __name__=='__main__':main()
