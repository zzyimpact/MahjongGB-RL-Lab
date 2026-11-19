"""Recover original full-sample positions without storing forced observations.

The full permutation and batch denominators match round-three BC. GroupNorm
acts per sample, so omitting zero-gradient forced rows preserves its objective.
This is an optional sampling control, not additional expert data.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .replay import replay_game
from .data_v2 import wall_identity
from .cache_identity import check


def worker(job):
    gid,line,expected=job
    game=json.loads(line)
    if wall_identity(game)[1]!=expected:raise ValueError('wall mismatch')
    samples=replay_game(game,'basic')
    return gid,np.asarray([x[1] for x in samples],dtype=bool),np.asarray([x[2] for x in samples],dtype=np.int16)


def prepare(source,root,workers=6):
    root=Path(root);check(root);manifest=json.loads((root/'manifest.json').read_text())
    if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=manifest['source_sha256']:raise ValueError('corpus identity mismatch')
    output=root/'train_full_index.npy';record=root/'full_sampling.json'
    if output.exists() or record.exists():raise FileExistsError('full sampling map already exists')
    games={g['game_id']:g for g in manifest['games'] if g['split']=='train'}
    masks=np.load(root/'train_mask.npy',mmap_mode='r');actions=np.load(root/'train_action.npy',mmap_mode='r');gids=np.load(root/'train_game_id.npy',mmap_mode='r')
    jobs=[]
    with gzip.open(source,'rt') as f:
        for gid,line in enumerate(f):
            if gid in games:jobs.append((gid,line,games[gid]['wall_id']))
    mapping=np.full(manifest['summary']['train']['samples'],-1,dtype=np.int32);full_cursor=nf_cursor=0;start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for gid,mask,action in pool.map(worker,jobs,chunksize=16):
            nf=mask.sum(1)>1;n=int(nf.sum());end=nf_cursor+n
            if not (np.array_equal(mask[nf],masks[nf_cursor:end]) and np.array_equal(action[nf],actions[nf_cursor:end]) and (gids[nf_cursor:end]==gid).all()):raise ValueError('nonforced cache ordering mismatch')
            mapping[full_cursor:full_cursor+len(action)][nf]=np.arange(nf_cursor,end)
            full_cursor+=len(action);nf_cursor=end
    if full_cursor!=len(mapping) or nf_cursor!=len(actions):raise ValueError('sample counts mismatch')
    np.save(output,mapping)
    result=dict(schema=1,source_sha256=manifest['source_sha256'],manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
                index_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),samples=full_cursor,nonforced=nf_cursor,games=len(games),seconds=time.monotonic()-start,
                method='replay game/seat ordering, validate every nonforced mask/action/game_id, forced index -1')
    record.write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--data',required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.data,a.workers),indent=2))


if __name__=='__main__':main()
