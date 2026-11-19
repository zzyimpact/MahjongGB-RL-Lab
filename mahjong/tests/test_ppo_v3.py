import json
from pathlib import Path
import torch
import numpy as np
from tensordict import TensorDict
from mahjong.model import make_model,input_keys
from mahjong.features import encode
from mahjong.tests.test_v3 import view
from mahjong.ppo_v3 import isolate_critic,MixedCollector,FrozenOpponent
from mahjong.ppo_anchor import masked_reference_kl


def test_value_gradient_isolation_and_actor_kl():
    torch.set_num_threads(1);cfg=json.loads(Path('mahjong/configs/v3/V3.json').read_text());m=make_model(cfg);isolate_critic(m)
    v=view();v.request2obs('Draw B2');a=encode(v,'v3');args=[torch.tensor(a[k]).float().unsqueeze(0) for k in input_keys(cfg)]
    logits,value=m(*args);value.square().mean().backward()
    assert m.critic[0].weight.grad.norm()>0
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('critic.'))
    m.zero_grad();reference=logits.detach()+torch.randn_like(logits);mask=torch.tensor(a['action_mask']).unsqueeze(0)
    loss=masked_reference_kl(m(*args)[0],reference,mask).mean();assert torch.isfinite(loss) and loss>=0
    loss.backward();assert m.conv[0].weight.grad.norm()>0 and m.discard[0].weight.grad.norm()>0
    assert all(p.grad is None for p in m.critic.parameters())


def test_mixed_collection_filters_frozen_trajectories_and_rotates():
    torch.set_num_threads(1);cfg=json.loads(Path('mahjong/configs/v3/V3.json').read_text());cfg['seed']=130000000
    collector=MixedCollector(0,cfg,{'C':'artifacts/round3/C/checkpoint.pt','D':'artifacts/round3/D/checkpoint.pt'})
    frozen_before={n:p.clone() for n,p in collector.frozen['C'].named_parameters()}
    for offset in range(4):
        batch=collector.collect_episode(config={'seed':130000000+offset})
        assert batch.completed_episodes==1 and len(batch.trajectories)==2
        assert {int(t.key.agent_id) for t in batch.trajectories}=={offset,(offset+1)%4}
        assert all(t.policy_id=='shared' and t.policy_version==0 for t in batch.trajectories)
        for t in batch.trajectories:
            assert torch.isfinite(t.data.get(('policy','extras','logp'))).all()
            mask=t.data.get(('obs','action_mask'));action=t.data.get(('action','action'))
            assert mask.gather(1,action[:,None]).all()
    for n,p in collector.frozen['C'].named_parameters():torch.testing.assert_close(p,frozen_before[n],rtol=0,atol=0)


def test_historical_adapter_agrees_with_original_features():
    from mahjong.arena import load_checkpoint
    v=view();v.request2obs('Draw B2');a=encode(v,'v3');model,cfg=load_checkpoint('artifacts/round3/C/checkpoint.pt')
    opponent=FrozenOpponent('artifacts/round3/C/checkpoint.pt','v3')
    obs=TensorDict({k:torch.tensor(x) for k,x in a.items()},[])
    with torch.no_grad():expected=model(torch.tensor(encode(v,'basic')['obs']).unsqueeze(0))[0].masked_fill(~torch.tensor(a['action_mask']),-1e9).argmax().item()
    assert int(opponent.act(obs).action['action'])==expected
