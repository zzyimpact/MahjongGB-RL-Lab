"""Read-only full cache verification, including cross-split byte coincidences."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .gamecore.engine import GameCore
from .replay import replay_game


def main():
    p=argparse.ArgumentParser();p.add_argument('--basic',type=Path,required=True);p.add_argument('--enhanced',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('output exists')
    basic=json.loads((a.basic/'manifest.json').read_text());enhanced=json.loads((a.enhanced/'manifest.json').read_text())
    assert basic['games']==enhanced['games'] and basic['groups']==enhanced['groups']
    splits=('train','val','test');hashes={s:set() for s in splits};selected={};cached={};legal=0
    for split in splits:
        arrays={k:np.load(a.enhanced/(split+'_'+k+'.npy'),mmap_mode='r') for k in ('obs','mask','action','game_id')}
        originals={k:np.load(a.basic/(split+'_'+k+'.npy'),mmap_mode='r') for k in arrays}
        for start in range(0,len(arrays['action']),8192):
            sl=slice(start,start+8192)
            np.testing.assert_array_equal(arrays['obs'][sl,:6],originals['obs'][sl])
            for k in ('mask','action','game_id'):np.testing.assert_array_equal(arrays[k][sl],originals[k][sl])
            masks=arrays['mask'][sl];acts=arrays['action'][sl]
            assert masks[np.arange(len(acts)),acts].all();legal+=len(acts)
            for ob,ma,ac in zip(arrays['obs'][sl],masks,acts):
                hashes[split].add(hashlib.blake2b(ob.tobytes()+ma.tobytes()+ac.tobytes(),digest_size=16).digest())
        ids=np.unique(arrays['game_id'])
        selected[split]=[int(ids[0]),int(ids[len(ids)//2]),int(ids[-1])]
        for gid in selected[split]:
            sel=arrays['game_id']==gid;cached[gid]={k:v[sel] for k,v in arrays.items()}
    checked=[]
    with gzip.open(basic['source'],'rt') as f:
        for gid,line in enumerate(f):
            if gid not in cached:continue
            obs,mask,act,_=zip(*replay_game(json.loads(line),feature_set='enhanced'))
            # Float16 storage is lossless for this binary-fraction feature schema.
            np.testing.assert_array_equal(np.array(obs),cached[gid]['obs'].astype(np.float32))
            np.testing.assert_array_equal(mask,cached[gid]['mask']);np.testing.assert_array_equal(act,cached[gid]['action']);checked.append(gid)
    teacher_walls=[]
    for seed in range(1000):
        core=GameCore();core.r.seed(seed);core.reset()
        teacher_walls.append(hashlib.sha256(core.originalTileWall.encode()).hexdigest())
    result=dict(identical_assignments=True,identical_basic_prefix_labels_masks_game_ids=True,legal_samples=legal,
                source_reconstruction_checked_game_ids=checked,enhanced_float16_storage_lossless=True,
                unique_enhanced_samples={s:len(hashes[s]) for s in splits},
                cross_split_unique_enhanced_byte_overlap={s+'_'+t:len(hashes[s]&hashes[t]) for s,t in [('train','val'),('train','test'),('val','test')]},
                historical_teacher=dict(seeds=[0,999],unique_walls=len(set(teacher_walls)),ijcai_overlap=len(set(teacher_walls)&set(basic['groups'])),used_in_round3=False))
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
