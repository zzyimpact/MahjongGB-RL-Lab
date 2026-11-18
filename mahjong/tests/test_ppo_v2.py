import copy
import numpy as np
import pytest
import torch
from mahjong.ppo_anchor import masked_reference_kl,BCReferenceLoss
from mahjong.policy import make_policy,make_collector
from rlframework.algorithm.ppo import PPOLearner,PPOConfig
from rlframework.processing.gae import GAEProcessor


def test_kl_direction_mask_gradient_and_stability():
    current=torch.tensor([[1.,-1.,1e20],[1.,2.,3.]],requires_grad=True)
    reference=torch.tensor([[-.5,.5,-1e20],[3.,2.,1.]],requires_grad=True)
    mask=torch.tensor([[True,True,False],[False,True,False]])
    kl=masked_reference_kl(current,reference,mask)
    p=torch.softmax(reference[0,:2].detach(),-1)
    expected=(p*(torch.log(p)-torch.log_softmax(current[0,:2],-1))).sum()
    torch.testing.assert_close(kl[0],expected);assert kl[1]==0 and kl[0]>0
    kl.sum().backward();assert reference.grad is None and current.grad[0,2]==0
    assert current.grad[0,0]>0 and current.grad[0,1]<0
    assert torch.isfinite(current.grad).all()
    identical=masked_reference_kl(current.detach(),current.detach(),mask)
    torch.testing.assert_close(identical,torch.zeros(2))


@pytest.mark.parametrize("architecture", ["tiny_cnn", "small_resnet"])
def test_enhanced_bc_restores_for_real_ppo_value_update(tmp_path, architecture):
    torch.set_num_threads(1);torch.manual_seed(8)
    cfg=dict(feature_set='enhanced',feature_version='v2',architecture=architecture,hidden=64)
    original=make_policy(config=cfg);torch.save(original.state_dict(),tmp_path/'bc.pt')
    policy=make_policy(config=cfg);policy.load_state_dict(torch.load(tmp_path/'bc.pt'),strict=True)
    collector=make_collector(0,config=cfg)
    collector.router.policies['shared'].load_state_dict(policy.state_dict(),strict=True)
    collection=collector.collect_episode(config={'seed':654321})
    processor=GAEProcessor();batch=torch.cat([processor.process(t.data,policy) for t in collection.trajectories],dim=0)
    before=copy.deepcopy(policy.module.state_dict())
    reference=tmp_path/'reference.pt'
    torch.save({'policy':original.state_dict(),'config':cfg},reference)
    learner=PPOLearner(policy,PPOConfig(epochs=1,minibatch_size=256))
    learner.auxiliary_loss=BCReferenceLoss(policy,reference,cfg,'cpu',.02)
    metrics=learner.update(batch)
    assert 'ppo/bc_reference_kl' in metrics
    assert all(p.grad is None for p in learner.auxiliary_loss.model.parameters())
    assert np.isfinite(list(metrics.values())).all()
    assert not torch.equal(before['critic.weight'],policy.module.critic.weight)
    assert not torch.equal(before['conv.0.weight'],policy.module.conv[0].weight)
