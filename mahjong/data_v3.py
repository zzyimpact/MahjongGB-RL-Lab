"""Replay current visible features; preserve round-3 wall assignments exactly.

Only nonforced decisions are stored. Forced metrics are exact analytical
counts (masked CE=0, accuracy=1); raw counts are never disguised as new data.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .replay import RecordingSession, response_to_action, seat_pairs
from .data_v2 import wall_identity

KEYS = ('obs','history','melds','candidate','mask','action','game_id')
DTYPES = dict(obs=np.float16,history=np.int8,melds=np.int8,candidate=np.float16,mask=bool,action=np.int16,game_id=np.int32)
SHAPES = dict(obs=(34,4,9),history=(16,3),melds=(4,4,3),candidate=(34,58),mask=(235,),action=(),game_id=())

class Recorder(RecordingSession):
    def __init__(self):
        super().__init__('v3')
        self.rows=[];self.forced=Counter();self.counts=Counter()
    def _choose(self, arrays):
        if not self.pending:raise ValueError('missing label')
        action=response_to_action(np.flatnonzero(arrays['action_mask']).tolist(),self.pending.pop(0))
        self.samples.append(None) # Feed's response-validation decision counter.
        self.counts[action]+=1
        if arrays['action_mask'].sum()==1:self.forced[action]+=1
        else:self.rows.append(dict(arrays,action=action))
        return action


def worker(job):
    root, shard, records, game_map = job
    rows=[];summary={};start=time.monotonic()
    for gid,line in records:
        game=json.loads(line);srand,wall=wall_identity(game)
        if wall!=game_map[gid]['wall_id']:raise ValueError('raw source/manifest disagreement')
        if game.get('status')!='finished':raise ValueError('unfinished game')
        counts=Counter();forced=Counter()
        for seat in range(4):
            session=Recorder()
            for request,response in seat_pairs(game['logs'],str(seat)):
                session.feed(request,response)
            rows.extend(dict(r,game_id=gid) for r in session.rows)
            counts.update(session.counts);forced.update(session.forced)
        summary[gid]=dict(counts=dict(counts),forced=dict(forced))
    arrays={k:np.asarray([r['action_mask' if k=='mask' else k] for r in rows],dtype=DTYPES[k]) for k in KEYS}
    if any(a.shape!=(len(rows),)+SHAPES[k] for k,a in arrays.items()):raise ValueError('bad v3 feature shape')
    if not arrays['mask'][np.arange(len(rows)),arrays['action']].all():raise ValueError('illegal label')
    if not np.isfinite(arrays['candidate']).all():raise ValueError('bad candidate values')
    path=Path(root)/('part_%04d.npz'%shard);np.savez(path,**arrays)
    result=dict(shard=shard,samples=len(rows),games=summary,seconds=time.monotonic()-start)
    path.with_suffix('.json').write_text(json.dumps(result))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path('reports/round3/raw/split_manifest.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--shard-games',type=int,default=100);p.add_argument('--resume',action='store_true');a=p.parse_args()
    source=json.loads(a.manifest.read_text())
    if hashlib.sha256(a.input.read_bytes()).hexdigest()!=source['source_sha256']:raise ValueError('raw corpus identity mismatch')
    if a.output.exists() and not a.resume:p.error('output exists, use --resume only for this exact input')
    a.output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();game_map={g['game_id']:g for g in source['games']};jobs=[];existing=[]
    with gzip.open(a.input,'rt') as f:
        records=list(enumerate(f))
    for shard,first in enumerate(range(0,len(records),a.shard_games)):
        batch=records[first:first+a.shard_games];path=a.output/('part_%04d.json'%shard)
        if a.resume and path.exists() and path.with_suffix('.npz').exists():
            result=json.loads(path.read_text())
            if sorted(map(int,result['games']))!=[g for g,l in batch]:raise ValueError('resume shard configuration mismatch')
            existing.append(result)
        else:jobs.append((str(a.output),shard,batch,game_map))
    results=existing
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(worker,jobs):
            results.append(r);print(json.dumps(dict(shard=r['shard'],samples=r['samples'],elapsed=time.monotonic()-start)),flush=True)
    gs={int(g):v for r in results for g,v in r['games'].items()}
    if set(gs)!=set(game_map):raise ValueError('not all games reconstructed')
    summary={};split_id={g:game_map[g]['split'] for g in game_map}
    for split in ['train','val','test']:
        gids=[g for g in gs if split_id[g]==split];counts=Counter();forced=Counter()
        for g in gids:counts.update({int(k):v for k,v in gs[g]['counts'].items()});forced.update({int(k):v for k,v in gs[g]['forced'].items()})
        n=sum(counts.values())-sum(forced.values())
        arrays={k:np.lib.format.open_memmap(a.output/(split+'_'+k+'.npy'),mode='w+',dtype=DTYPES[k],shape=(n,)+SHAPES[k]) for k in KEYS}
        cursor=0
        for r in sorted(results,key=lambda r:r['shard']):
            with np.load(a.output/('part_%04d.npz'%r['shard'])) as data:
                sel=np.asarray([split_id[int(g)]==split for g in data['game_id']]);size=int(sel.sum())
                for k in KEYS:arrays[k][cursor:cursor+size]=data[k][sel]
                cursor+=size
        assert cursor==n
        for v in arrays.values():v.flush()
        summary[split]=dict(samples=sum(counts.values()),stored_nonforced=n,forced=sum(forced.values()),games=len(gids),
                            walls=len({game_map[g]['wall_id'] for g in gids}),action_counts=dict(counts),forced_action_counts=dict(forced))
    manifest=dict(schema=3,feature_version='v3',complete=True,source_sha256=source['source_sha256'],
                  parent_manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),groups=source['groups'],games=source['games'],
                  summary=summary,stored='nonforced only; forced observations have exactly zero masked loss',seconds=time.monotonic()-start)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(summary=summary,seconds=manifest['seconds'])),flush=True)

if __name__=='__main__':main()
