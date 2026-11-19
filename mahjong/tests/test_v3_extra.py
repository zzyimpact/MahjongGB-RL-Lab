import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from mahjong.features import encode
from mahjong.tests.test_v3 import view
from mahjong.cache_identity import check
from mahjong.arena_v3 import aggregate
from mahjong.train_v3 import train


def test_kong_and_last_wall_v3():
    v=view();v.request2obs('Draw B3');v.request2obs('Player 0 AnGang W1')
    # No decision occurs between a kong and its replacement draw.
    v.request2obs('Draw B2');a=encode(v,'v3');assert a['obs'][33].max()==1
    assert a['melds'][0,0].tolist()==[4,1,0]
    for action in v.valid:
        if 2<=action<36:assert a['candidate'][action-2,2]==14/16 # seven pairs unavailable
    v=view();v.request2obs('Player 3 Play J2');v.request2obs('Player 1 Peng');v.request2obs('Player 1 BuGang J2')
    a=encode(v,'v3');assert a['melds'][1,0].tolist()==[3,33,4]
    assert a['history'][-1].tolist()==[2,7,33] and a['obs'][31].max()==1
    v=view();v.tileWall[1]=0;v.request2obs('Draw B2');a=encode(v,'v3')
    assert a['obs'][32].max()==1 and all(x<133 for x in v.valid)


def test_cache_guard_rejects_wrong_source(tmp_path):
    check(tmp_path,create=True);p=tmp_path/'feature_provenance.json';value=json.loads(p.read_text());value['feature_version']='stale';p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Stale'):check(tmp_path)


def test_aggregate_pairs_by_wall_and_opponent():
    rows=[dict(candidate=c,opponent=o,seed=s,reward=s+d) for c,d in [('A',0),('B',2)] for o in ['r','t'] for s in range(12)]
    result=aggregate(rows);assert result['paired_differences']['B - A']['ci95']==[2,2]
    assert result['scores']['A']['walls']==12
    with pytest.raises(ValueError):aggregate(rows[:-1])


def test_training_resume_matches_uninterrupted(tmp_path,monkeypatch):
    import mahjong.train_v3 as module
    root=tmp_path/'data';root.mkdir();check(root,create=True)
    cfg=json.loads(Path('mahjong/configs/v3/D_retrained.json').read_text())
    cfg.update(device='cpu',feature_set='enhanced',architecture='tiny_cnn',seed=199)
    cfg['bc'].update(epochs=3,min_epochs=3,patience=9,batch_size=8)
    rng=np.random.RandomState(7);n=24
    mask=np.zeros((n,235),bool);mask[:,2:6]=True;action=rng.randint(2,6,n).astype(np.int16)
    arrays=dict(obs=rng.rand(n,34,4,9).astype(np.float16),mask=mask,action=action)
    manifest=dict(schema=3,complete=True,groups={'a':'train','b':'val'},games=[dict(game_id=0,wall_id='a',split='train'),dict(game_id=1,wall_id='b',split='val')],summary={s:dict(samples=n,forced_action_counts={}) for s in ['train','val']})
    (root/'manifest.json').write_text(json.dumps(manifest))
    for split,gid in [('train',0),('val',1)]:
        for k,a in dict(arrays,game_id=np.full(n,gid,np.int32)).items():np.save(root/(split+'_'+k+'.npy'),a)
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg));full=tmp_path/'full';part=tmp_path/'part';train(path,root,full)
    original=module.atomic_save
    def interrupted(state,path):
        original(state,path)
        if path.name=='last.pt' and state['bc']['epoch']==2:raise RuntimeError('simulated power interruption')
    monkeypatch.setattr(module,'atomic_save',interrupted)
    with pytest.raises(RuntimeError,match='simulated'):train(path,root,part)
    monkeypatch.setattr(module,'atomic_save',original);train(path,root,part,resume=True)
    a=torch.load(full/'last.pt');b=torch.load(part/'last.pt')
    for k,v in a['policy'].items():torch.testing.assert_close(v,b['policy'][k],rtol=0,atol=0)
    assert a['bc']['updates']==b['bc']['updates']==9
    assert a['bc']['history'][-1]['val']==b['bc']['history'][-1]['val']


def test_custom_arena_registry_blocks_final_development_overlap(tmp_path):
    from mahjong.arena_v3 import run,wall_hash
    prior=tmp_path/'development';prior.mkdir()
    (prior/'arena_config.json').write_text(json.dumps(dict(purpose='development',wall_ids=[wall_hash(170000000)])))
    with pytest.raises(ValueError,match='overlap'):
        run({'C':'artifacts/round3/C/checkpoint.pt'},{'random':None},'reports/round3/raw/split_manifest.json',tmp_path/'final',170000000,2,1,'final',registry_root=tmp_path)
