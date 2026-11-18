"""Versioned, wall-disjoint IJCAI preparation. Old shards are never accepted.

The actual ordered 136-tile wall is fingerprinted, not inferred from a game ID.
A srand must map to exactly one wall; identical walls always share a split.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .replay import replay_game

SCHEMA = 2

def wall_identity(game):
    init = game['initdata']
    init = json.loads(init) if isinstance(init, str) else init
    tiles = init['walltiles'].split()
    from .observation import SeatView
    if Counter(tiles) != Counter({t: 4 for t in SeatView.TILE_LIST}):
        raise ValueError('not a 136-tile no-flower wall')
    srand = str(init['srand'])
    displayed = game['logs'][0]['output']['display']['srand']
    if str(displayed) != srand:
        raise ValueError('INIT srand disagrees with initdata')
    return srand, hashlib.sha256(' '.join(tiles).encode()).hexdigest()

def split_groups(groups, seed=20261119):
    groups = sorted(set(groups))
    if len(groups) < 3:
        raise ValueError('at least three independent walls required')
    np.random.RandomState(seed).shuffle(groups)
    n = max(1, round(len(groups) * .1))
    return {g: ('test' if i < n else 'val' if i < 2*n else 'train') for i,g in enumerate(groups)}

def scan(path, seed):
    games, srand_to_wall, wall_to_srands = [], {}, defaultdict(set)
    with gzip.open(path, 'rt') as f:
        for gid, line in enumerate(f):
            game = json.loads(line)
            srand, wall = wall_identity(game)
            if srand in srand_to_wall and srand_to_wall[srand] != wall:
                raise ValueError('srand reused for different ordered walls')
            srand_to_wall[srand] = wall
            wall_to_srands[wall].add(srand)
            games.append(dict(game_id=gid, srand=srand, wall_id=wall))
    assignment = split_groups(wall_to_srands, seed)
    for g in games:
        g['split'] = assignment[g['wall_id']]
    return dict(schema=SCHEMA, source=str(path), source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                split_seed=seed, groups=assignment, games=games,
                unique_srands=len(srand_to_wall), unique_walls=len(assignment),
                walls_with_multiple_srands=sum(len(s)>1 for s in wall_to_srands.values()))

def worker(job):
    directory, shard, records, feature_set = job
    samples, gids, errors, accepted = [], [], [], []
    for gid, line in records:
        try:
            rows = replay_game(json.loads(line), feature_set=feature_set)
            if not rows:
                raise ValueError('empty replay')
            samples.extend(rows)
            gids.extend([gid]*len(rows))
            accepted.append(gid)
        except Exception as e:
            errors.append(dict(game_id=gid, error=repr(e)))
    if samples:
        obs, mask, action, seat = zip(*samples)
        np.savez(Path(directory)/('part_%04d.npz'%shard), obs=np.asarray(obs,dtype=np.float16),
                 mask=np.asarray(mask,dtype=bool),action=np.asarray(action,dtype=np.int16),
                 game_id=np.asarray(gids,dtype=np.int32))
    return dict(shard=shard, accepted=accepted, errors=errors, samples=len(samples))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--feature-set',choices=['basic','enhanced'],default='basic')
    p.add_argument('--seed',type=int,default=20261119)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--shard-games',type=int,default=100)
    a=p.parse_args()
    if a.output.exists(): p.error('output already exists; caches cannot silently bypass split generation')
    a.output.mkdir(parents=True)
    start=time.monotonic()
    manifest=scan(a.input,a.seed)
    manifest.update(feature_set=a.feature_set,feature_version='v1' if a.feature_set=='basic' else 'v2')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    jobs=[]
    with gzip.open(a.input,'rt') as f:
        chunk=[]
        for gid,line in enumerate(f):
            chunk.append((gid,line))
            if len(chunk)==a.shard_games:
                jobs.append((str(a.output),len(jobs),chunk,a.feature_set));chunk=[]
        if chunk: jobs.append((str(a.output),len(jobs),chunk,a.feature_set))
    stats=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(worker,jobs):
            stats.append(result)
            print(json.dumps(dict(shard=result['shard'],samples=result['samples'],errors=len(result['errors']),seconds=time.monotonic()-start)),flush=True)
    game_split=np.array([g['split'] for g in manifest['games']])
    paths=sorted(a.output.glob('part_*.npz'))
    sizes=Counter()
    for path in paths:
        with np.load(path) as d: sizes.update(game_split[d['game_id']].tolist())
    summary={}
    hashes={s:set() for s in sizes}
    for split,n in sizes.items():
        channels=6 if a.feature_set=='basic' else 34
        arrays={k:np.lib.format.open_memmap(a.output/(split+'_'+k+'.npy'),mode='w+',dtype=dtype,shape=(n,)+shape)
                for k,dtype,shape in [('obs',np.float16,(channels,4,9)),('mask',bool,(235,)),('action',np.int16,()),('game_id',np.int32,())]}
        cursor=0; classes=Counter(); forced=0; game_ids=set()
        for path in paths:
            with np.load(path) as data:
                sel=game_split[data['game_id']]==split
                if not sel.any():continue
                batch={k:data[k][sel] for k in arrays}
                count=len(batch['action'])
                if not batch['mask'][np.arange(count),batch['action']].all():raise ValueError('illegal label')
                for k in arrays:arrays[k][cursor:cursor+count]=batch[k]
                cursor+=count
                classes.update(map(int,batch['action']));forced+=int((batch['mask'].sum(1)==1).sum())
                game_ids.update(map(int,batch['game_id']))
                # Canonical basic obs fingerprints permit comparison across feature variants.
                for ob,ma,ac in zip(batch['obs'][:,:6],batch['mask'],batch['action']):
                    hashes[split].add(hashlib.blake2b(ob.tobytes()+ma.tobytes()+ac.tobytes(),digest_size=16).digest())
        for array in arrays.values():array.flush()
        summary[split]=dict(samples=n,games=len(game_ids),walls=len({manifest['games'][g]['wall_id'] for g in game_ids}),
                            forced_samples=forced,nonforced_fraction=1-forced/n,action_counts=dict(classes),unique_basic_samples=len(hashes[split]))
    overlap={a+'_'+b:len(hashes[a]&hashes[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    manifest.update(complete=True,summary=summary,reconstruction=stats,cross_split_unique_basic_byte_overlap=overlap,seconds=time.monotonic()-start)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for path in paths:path.unlink() # Only intermediate files created in this new output directory.
    print(json.dumps(dict(summary=summary,byte_overlap=overlap,seconds=manifest['seconds'])),flush=True)

if __name__=='__main__': main()
