import copy
import numpy as np
import pytest
import torch
from mahjong.observation import SeatView
from mahjong.features import encode
from mahjong.model import make_model
from mahjong.botzone.session import Session


def view(seat=0):
    v=SeatView(seat);v.request2obs('Wind 1')
    v.request2obs('Deal W1 W1 W1 W1 W2 W3 W4 W5 W6 T1 T2 B1 F1')
    return v


def test_shape_deterministic_private_and_rotation():
    v=view();a=encode(v,'enhanced')['obs']
    assert a.shape==(34,4,9) and a.min()>=0 and a.max()<=1
    np.testing.assert_array_equal(a[:6],encode(v)['obs'])
    np.testing.assert_array_equal(a,encode(v,'enhanced')['obs'])
    # Encoder never reads accidental omniscient attributes, future, or results.
    v.opponent_hands=['J3']*39;v.future_wall=['B9']*84;v.final_reward=999
    np.testing.assert_array_equal(a,encode(v,'enhanced')['obs'])
    for seat in range(4):
        w=view(seat);w.request2obs('Player %d Play J2'%((seat+3)%4))
        x=encode(w,'enhanced')['obs'].reshape(34,36)
        assert x[9,32]==.25 and x[17,32]==1 and x[18,32]==.25 and x[19,32]==.75


def test_meld_no_double_count_concealed_and_rob_kong():
    v=view();v.request2obs('Player 3 Play W2');v.request2obs('Player 0 Chi W3')
    x=encode(v,'enhanced')['obs'].reshape(34,36)
    assert list(x[18,1:4])==[.25,.25,.25] and x[30,0]==1
    assert x[10,1:4].sum()==.75
    v=view();v.request2obs('Player 0 AnGang W1')
    x=encode(v,'enhanced')['obs'].reshape(34,36)
    assert x[19,0]==0 and x[18,0]==0 and x[10,0]==1
    v.request2obs('Draw B2');x=encode(v,'enhanced')['obs'].reshape(34,36)
    assert x[28,0]==1 and x[33,0]==1
    v=view();v.request2obs('Player 3 Play J2');v.request2obs('Player 1 Peng')
    v.request2obs('Player 1 BuGang J2')
    x=encode(v,'enhanced')['obs'].reshape(34,36)
    assert x[15,32]==1 and x[31,0]==1 and x[18,32]==1
    v=view();v.request2obs('Player 2 AnGang')
    assert encode(v,'enhanced')['obs'].reshape(34,36)[26,0]==.25


def test_cold_reconstruction_and_last_wall():
    requests=['0 0 1','1 0 0 0 0 W1 W1 W1 W1 W2 W3 W4 W5 W6 T1 T2 B1 F1','3 3 PLAY J2','2 B2']
    recorded=[];inputs=[]
    def choose(a):inputs.append(a['obs'].copy());return int(np.flatnonzero(a['action_mask'])[0])
    warm=Session(choose,feature_set='enhanced')
    for r in requests:recorded.append(warm.handle(r))
    cold=Session(choose,feature_set='enhanced')
    for r,response in zip(requests[:-1],recorded[:-1]):cold.handle(r,forced=response)
    assert cold.handle(requests[-1])==recorded[-1]
    np.testing.assert_array_equal(encode(warm.view,'enhanced')['obs'],encode(cold.view,'enhanced')['obs'])
    v=view();v.tileWall[1]=0;v.request2obs('Draw B2')
    assert encode(v,'enhanced')['obs'][32].max()==1


def test_architecture_versions_strict_restore():
    for f in ['basic','enhanced']:
        for a in ['tiny_cnn','small_resnet']:
            cfg=dict(feature_set=f,architecture=a);m=make_model(cfg)
            other=make_model(cfg);other.load_state_dict(m.state_dict(),strict=True)
            x=torch.randn(2,6 if f=='basic' else 34,4,9)
            torch.testing.assert_close(m(x)[0],other(x)[0])
    with pytest.raises(ValueError):make_model(dict(feature_set='enhanced',feature_version='v1'))
    with pytest.raises(RuntimeError):make_model(dict(feature_set='enhanced')).load_state_dict(make_model({}).state_dict())


def test_real_judge_hidden_state_perturbation():
    from mahjong.env import Environment
    env=Environment(feature_set='enhanced');env.reset({'seed':12345})
    seat=next(iter(env.core.obs));before=encode(env.views[seat],'enhanced')['obs'].copy()
    for i in range(4):
        if i!=seat:env.core.hands[i]=['J3']*13
    env.core.tileWall=[['W9']*21 for _ in range(4)]
    env.core.reward=[999,-333,-333,-333]
    np.testing.assert_array_equal(before,encode(env.views[seat],'enhanced')['obs'])
