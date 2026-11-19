"""Exact source checks and cross-split SHA-256 record duplication counts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .data_v3 import Recorder,KEYS
from .replay import seat_pairs
from .cache_identity import check


def audit(root,source,output):
    root=Path(root);guard=check(root);manifest=json.loads((root/'manifest.json').read_text());summary={};sets={}
    keys=('obs','history','melds','candidate','mask','action')
    for split in ('train','val','test'):
        arrays={k:np.load(root/(split+'_'+k+'.npy'),mmap_mode='r') for k in keys+('game_id',)};n=len(arrays['action']);hashes=[]
        for start in range(0,n,4096):
            batch={k:np.array(arrays[k][start:start+4096]) for k in keys}
            for i in range(len(batch['action'])):
                h=hashlib.sha256()
                for k in keys:h.update(batch[k][i].tobytes())
                hashes.append(h.digest())
        previous=set().union(*sets.values())
        sets[split]=set(hashes)
        summary[split]=dict(stored_nonforced=n,unique_encoded_records=len(sets[split]),duplicates_within=n-len(sets[split]),
            records_matching_previous_split=sum(h in previous for h in hashes))
    # Reconstruct 3 whole games, one from each split; compare all stored fields.
    selected={next(g['game_id'] for g in manifest['games'] if g['split']==s):s for s in ('train','val','test')};replayed=[]
    with gzip.open(source,'rt') as f:
        for gid,line in enumerate(f):
            if gid not in selected:continue
            split=selected[gid];game=json.loads(line);rows=[]
            for seat in range(4):
                session=Recorder()
                for req,res in seat_pairs(game['logs'],str(seat)):session.feed(req,res)
                rows.extend(session.rows)
            ids=np.load(root/(split+'_game_id.npy'),mmap_mode='r');indices=np.flatnonzero(ids==gid)
            assert len(indices)==len(rows)
            for k in keys:
                a=np.load(root/(split+'_'+k+'.npy'),mmap_mode='r');expected=np.asarray([r['action_mask' if k=='mask' else k] for r in rows],dtype=a.dtype)
                np.testing.assert_array_equal(a[indices],expected)
            replayed.append(dict(game_id=gid,split=split,decisions=len(rows),identical=True))
    result=dict(feature_provenance=guard,split_statistics=manifest['summary'],duplicate_statistics=summary,replay_spot_checks=replayed,
        warning='byte repetition does not imply wall leakage; forced observations were not materialized and are excluded from duplicate counts',
        manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/final_v3');p.add_argument('--source',default='data/ijcai2026/final2_all.jsonl.gz');p.add_argument('--output',required=True);a=p.parse_args();audit(a.data,a.source,a.output)
