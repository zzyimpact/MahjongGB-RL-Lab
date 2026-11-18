import copy
import numpy as np
import pytest
import torch
from mahjong.model import Model
from mahjong.policy import make_policy
from mahjong.train_bc import bc_parameters, masked_ce, ShardDataset
from mahjong.data_v2 import split_groups


def test_backbone_actor_update_critic_unchanged_and_restore(tmp_path):
    torch.manual_seed(7)
    policy=make_policy();m=policy.module
    before=copy.deepcopy(m.state_dict())
    opt=torch.optim.Adam(bc_parameters(m),lr=.01)
    obs=torch.randn(12,6,4,9);mask=torch.zeros(12,235,dtype=torch.bool);mask[:,2:10]=True
    actions=torch.arange(12)%8+2
    logits,_=m(obs);loss=masked_ce(logits,mask,actions)
    expected=torch.nn.functional.cross_entropy(logits[:,2:10],actions-2)
    torch.testing.assert_close(loss,expected)
    opt.zero_grad();loss.backward()
    assert m.conv[0].weight.grad.norm()>0
    assert m.hidden[0].weight.grad.norm()>0
    opt.step()
    for k in ['conv.0.weight','hidden.0.weight','actor.weight']:assert not torch.equal(before[k],m.state_dict()[k])
    for k in ['critic.weight','critic.bias']:assert torch.equal(before[k],m.state_dict()[k])
    torch.save(policy.state_dict(),tmp_path/'bc.pt')
    restored=make_policy();restored.load_state_dict(torch.load(tmp_path/'bc.pt'),strict=True)
    torch.testing.assert_close(restored.module(obs)[0],m(obs)[0])


def test_forced_zero_gradient():
    logits=torch.randn(4,235,requires_grad=True);mask=torch.zeros(4,235,dtype=torch.bool);mask[:,0]=True
    loss=masked_ce(logits,mask,torch.zeros(4,dtype=torch.long));loss.backward()
    assert loss.item()==0 and torch.count_nonzero(logits.grad)==0
    with pytest.raises(ValueError):masked_ce(logits,mask,torch.ones(4,dtype=torch.long))


def test_split_reproducibility_and_legacy_cache_rejection(tmp_path):
    groups=['wall%d'%i for i in range(100)]
    a=split_groups(groups);assert a==split_groups(list(reversed(groups)))
    sets=[{g for g,s in a.items() if s==k} for k in ['train','val','test']]
    assert [len(s) for s in sets]==[80,10,10]
    assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
    (tmp_path/'manifest.json').write_text('{"schema":1}')
    with pytest.raises(ValueError):ShardDataset([tmp_path],'train')
