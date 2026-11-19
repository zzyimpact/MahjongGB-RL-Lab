"""Optional forward KL(ref || current) on identical legal-action support."""
import torch
from .arena import load_checkpoint
from .model import model_spec, input_keys


def masked_reference_kl(logits, reference_logits, action_mask):
    if logits.shape != reference_logits.shape or logits.shape != action_mask.shape:
        raise ValueError('KL shapes disagree')
    if not action_mask.any(-1).all():raise ValueError('empty KL action mask')
    logp=torch.log_softmax(logits.float().masked_fill(~action_mask,-1e9),dim=-1)
    logref=torch.log_softmax(reference_logits.detach().float().masked_fill(~action_mask,-1e9),dim=-1)
    return (logref.exp()*(logref-logp)).sum(-1)


class BCReferenceLoss:
    def __init__(self,policy,path,config,device,coefficient):
        self.model,refconfig=load_checkpoint(path)
        if model_spec(refconfig)!=model_spec(config):raise ValueError('BC reference configuration mismatch')
        if coefficient<0:raise ValueError('negative KL coefficient')
        self.model.to(device).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.policy=policy;self.coefficient=coefficient;self.keys=input_keys(config)

    def __call__(self,items,valid):
        obs=items.get('obs');x=[obs.get(k).float() for k in self.keys]
        with torch.no_grad():reference,_=self.model(*x)
        current,_=self.policy.module(*x)
        all_kl=masked_reference_kl(current,reference,obs.get('action_mask').bool())
        kl=all_kl[valid].mean()
        nonforced=valid & (obs.get('action_mask').sum(-1)>1)
        nf=all_kl[nonforced].mean() if nonforced.any() else kl*0
        return self.coefficient*kl,{'bc_reference_kl':kl,'bc_reference_kl_nonforced':nf}
