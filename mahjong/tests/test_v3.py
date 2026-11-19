import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from MahjongGB import RegularShanten, MahjongShanten
from mahjong.features import encode
from mahjong.features_v3 import hand_shapes, candidate_features, INDEX
from mahjong.observation import SeatView
from mahjong.model import make_model,input_keys
from mahjong.botzone.session import Session
from mahjong.env import Environment


def view(seat=0):
    v=SeatView(seat);v.request2obs('Wind 1')
    v.request2obs('Deal W1 W1 W1 W1 W2 W3 W4 W5 W6 T1 T2 B1 F1')
    return v


def test_candidate_shapes_and_no_mutation():
    v=view();v.request2obs('Draw B2');before=copy.deepcopy(v.__dict__);a=encode(v,'v3')
    assert a['candidate'].shape==(34,58) and a['history'].shape==(16,3)
    assert np.isfinite(a['candidate']).all() and np.min(a['candidate'])>=0
    assert before==v.__dict__
    for action in v.valid:
        if not 2<=action<36:continue
        i=action-2;h=v.hand.copy();h.remove(v.TILE_LIST[i]);row=a['candidate'][i]
        assert row[1]*16==RegularShanten(tuple(h))[0]
        assert row[6]*16==MahjongShanten(pack=(),hand=tuple(h))
        assert row[8]*128==sum(a['obs'].reshape(34,36)[19,j]*4 for j in np.flatnonzero(row[24:]))
    for k,x in a.items():np.testing.assert_array_equal(x,encode(v,'v3')[k])


def test_native_special_shape_calls_have_no_state_pollution():
    hands=[('W1','W1','W2','W3','W4','W5','W6','T1','T2','B1','F1','J2','J3'),
           ('W1','W1','W2','W2','W3','W3','T1','T1','T9','T9','B1','B1','J1')]
    initial=[hand_shapes.__wrapped__(h,0) for h in hands]
    for i in range(20):
        for j,h in enumerate(reversed(hands)):
            hand_shapes.__wrapped__(h[:10],1)
            assert hand_shapes.__wrapped__(h,0)==initial[1-j]


def test_history_meld_structure_and_relative_seats():
    for seat in range(4):
        v=view(seat);prev=(seat+3)%4
        v.request2obs('Player %d Play W2'%prev);v.request2obs('Player %d Chi W3'%seat)
        a=encode(v,'v3');assert a['melds'][0,0].tolist()==[1,3,4]
        assert a['history'][-2].tolist()==[4,2,2]
        assert a['history'][-1].tolist()==[1,3,3]
        v.request2obs('Player %d AnGang'%((seat+2)%4));a=encode(v,'v3')
        assert a['melds'][2,0].tolist()==[4,0,0] and a['history'][-1,2]==0


def test_hidden_future_perturbation_and_cold_session():
    env=Environment(feature_set='v3');env.reset({'seed':54321});seat=next(iter(env.core.obs))
    a=encode(env.views[seat],'v3');env.core.hands[(seat+1)%4]=['J3']*13;env.core.tileWall=[['W9']*21]*4
    for k,x in a.items():np.testing.assert_array_equal(x,encode(env.views[seat],'v3')[k])
    reqs=['0 0 1','1 0 0 0 0 W1 W1 W1 W1 W2 W3 W4 W5 W6 T1 T2 B1 F1','3 3 PLAY J2','2 B2']
    choose=lambda a:int(np.flatnonzero(a['action_mask'])[0]);warm=Session(choose,feature_set='v3');cold=Session(choose,feature_set='v3')
    rs=[warm.handle(r) for r in reqs]
    for r,res in zip(reqs[:-1],rs[:-1]):cold.handle(r,forced=res)
    assert cold.handle(reqs[-1])==rs[-1]
    for k,x in encode(warm.view,'v3').items():np.testing.assert_array_equal(x,encode(cold.view,'v3')[k])


def test_v3_shapes_backward_strict_and_old_models():
    torch.set_num_threads(1);v=view();v.request2obs('Draw B2');a=encode(v,'v3')
    cfg=dict(feature_set='v3',feature_version='v3',architecture='v3_resnet');m=make_model(cfg)
    args=[torch.from_numpy(a[k]).unsqueeze(0).float() for k in input_keys(cfg)]
    logits,value=m(*args);assert logits.shape==(1,235) and value.shape==(1,)
    logits.sum().backward();assert m.conv[0].weight.grad.norm()>0 and m.discard[0].weight.grad.norm()>0
    assert 1000000 <= sum(p.numel() for p in m.parameters()) <=3000000
    restored=make_model(cfg);restored.load_state_dict(m.state_dict(),strict=True)
    torch.testing.assert_close(restored(*args)[0],logits)
    with pytest.raises(ValueError):make_model(dict(feature_set='v3',architecture='small_resnet'))
    assert sum(p.numel() for p in make_model({}).parameters())==55468
