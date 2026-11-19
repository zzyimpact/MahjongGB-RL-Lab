import torch
from mahjong.train_v3 import masked_loss,lr_at


def test_label_smoothing_respects_mask_and_forced_zero_signal():
    logits=torch.tensor([[2.,1.,100.],[0.,-1.,100.]],requires_grad=True)
    mask=torch.tensor([[True,True,False],[True,False,False]])
    labels=torch.tensor([1,0]);loss,ce=masked_loss(logits,mask,labels,.1)
    logp=torch.log_softmax(logits[0,:2],-1)
    expected=(.9*-logp[1]+.1*-logp.mean())/2
    torch.testing.assert_close(loss,expected);assert ce[1]==0
    loss.backward();assert logits.grad[:,2].abs().sum()==0 and logits.grad[1].abs().sum()==0


def test_lr_warmup_cosine_schedule():
    assert lr_at(0,100,10,.001,.00001)==.0001
    assert lr_at(9,100,10,.001,.00001)==.001
    assert lr_at(100,100,10,.001,.00001)==.00001
    assert lr_at(80,100,10,.001,.00001)<lr_at(40,100,10,.001,.00001)


def test_cosine_stays_at_floor_after_decay_horizon():
    assert lr_at(200,120,10,.0002,.000025)==.000025
