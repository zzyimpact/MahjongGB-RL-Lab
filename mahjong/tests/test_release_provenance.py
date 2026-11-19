"""Release verification must reject a substituted final-evaluation checkpoint."""
import hashlib
import json
import pytest
from mahjong.verify_round4 import verify_sealed_evaluation


def test_final_evaluation_is_bound_to_preselected_weights(tmp_path):
    def write(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path
    identity={'sha256':'abc123'}
    selection=write('selection.json',dict(candidates={'BC':identity},final_walls=500,final_seed=112000000))
    sha=hashlib.sha256(selection.read_bytes()).hexdigest()
    arena=dict(candidates=['BC'],opponents=['random'],purpose='final',walls=500,seed_start=112000000,identities={'BC':identity})
    write('arena_final/arena_config.json',arena)
    result=dict(games=2000,summary={'BC vs random':{'reward':{'walls':500}}})
    write('arena_final/results.json',result)
    write('test_metrics.json',dict(seal_sha256=sha,models={'BC':{'identity':identity}}))
    write('confirmatory_statistics.json',dict(selection_sha256=sha))
    write('wall_audit_final.json',dict(overlaps={'final vs train':0},unique_walls={'final':500}))
    assert verify_sealed_evaluation(tmp_path,sha)['test_and_arena_bound_to_seal']
    result['games']=1996;write('arena_final/results.json',result)
    with pytest.raises(ValueError,match='budget incomplete'):verify_sealed_evaluation(tmp_path,sha)
    result['games']=2000;write('arena_final/results.json',result)
    arena['identities']['BC']={'sha256':'substituted'};write('arena_final/arena_config.json',arena)
    with pytest.raises(ValueError,match='checkpoint differs'):verify_sealed_evaluation(tmp_path,sha)
    arena['identities']['BC']=identity;write('arena_final/arena_config.json',arena)
    write('wall_audit_final.json',dict(overlaps={'final vs train':1},unique_walls={'final':500}))
    with pytest.raises(ValueError,match='independence'):verify_sealed_evaluation(tmp_path,sha)
