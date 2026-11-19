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


def test_forced_row_elision_preserves_groupnorm_gradients_and_adam_step():
    import copy
    from mahjong.model import SmallResNet
    from mahjong.train_bc import bc_parameters
    torch.manual_seed(23);torch.set_num_threads(2)
    full=SmallResNet(6).double();reduced=copy.deepcopy(full)
    obs=torch.randn(9,6,4,9,dtype=torch.double)
    mask=torch.zeros(9,235,dtype=torch.bool);mask[:,0]=True;mask[::3,2:9]=True
    action=torch.zeros(9,dtype=torch.long);action[::3]=3;nf=mask.sum(1)>1
    optimizers=[torch.optim.Adam(bc_parameters(m),lr=.001) for m in [full,reduced]]
    for _ in range(2):
        for model,opt,sel,denominator in [(full,optimizers[0],slice(None),9),(reduced,optimizers[1],nf,9)]:
            opt.zero_grad(set_to_none=True);logits,_=model(obs[sel]);loss,ce=masked_loss(logits,mask[sel],action[sel])
            (loss*len(ce)/denominator).backward();torch.nn.utils.clip_grad_norm_(bc_parameters(model),10.)
        for p,q in zip(bc_parameters(full),bc_parameters(reduced)):torch.testing.assert_close(p.grad,q.grad,rtol=1e-8,atol=1e-10)
        for opt in optimizers:opt.step()
    for p,q in zip(full.parameters(),reduced.parameters()):torch.testing.assert_close(p,q,rtol=1e-8,atol=1e-10)


def test_full_sampling_preserves_permutation_positions_and_batch_denominators():
    import numpy as np
    from mahjong.train_v3 import Dataset
    ds=Dataset.__new__(Dataset);ds.full_index=np.array([-1,0,-1,1,2,-1,-1,3,-1],dtype=np.int32)
    ds.keys=('obs',);ds.config={'feature_set':'enhanced'}
    ds.arrays={'obs':np.arange(4,dtype=np.float32)[:,None,None,None],
               'mask':np.ones((4,235),bool),'action':np.arange(4,dtype=np.int16)}
    expected=np.random.RandomState(25).permutation(9)
    batches=list(ds.training_batches(4,'cpu',25))
    assert [b[-1] for b in batches]==[4,4,1]
    for i,(inputs,mask,action,denominator) in enumerate(batches):
        ids=ds.full_index[expected[i*4:(i+1)*4]];ids=ids[ids>=0]
        assert action.tolist()==ids.tolist()
        assert inputs['obs'].flatten().tolist()==ids.tolist()
