"""Resumable round-four arena with registered, disjoint wall populations.

The primary selection score is the equally weighted mean across five frozen
opponents. Inference resamples walls, never individual seat games.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from .arena import identity, init_worker, worker, summarize, cluster_stats
from .gamecore.engine import GameCore

# Round1 source/report specifies teacher 1000-1999; a round3 audit instead
# recorded 0-999. Conservatively exclude both documented populations.
HISTORICAL_RANGES = [(0,1000),(1000,1000),(2026,2400),(100000,100),(7000000,4),(8000000,100),(9000000,300),(8100000,50),
                     (9100000,300),(9200000,300),(42000000,320)]


def wall_hash(seed):
    core=GameCore();core.r.seed(seed);core.reset()
    return hashlib.sha256(core.originalTileWall.encode()).hexdigest()


def aggregate(wall_records):
    candidates=sorted({r['candidate'] for r in wall_records})
    opponents=sorted({r['opponent'] for r in wall_records});values={}
    for c in candidates:
        rows={}
        for r in wall_records:
            if r['candidate']==c:rows.setdefault(r['seed'],[]).append(r)
        if any(sorted(x['opponent'] for x in rs)!=opponents for rs in rows.values()):
            raise ValueError('aggregate needs all frozen opponents on every wall')
        values[c]={s:sum(x['reward'] for x in rs)/len(opponents) for s,rs in rows.items()}
    paired={}
    for i,c in enumerate(candidates):
        for b in candidates[:i]:
            if values[c].keys()!=values[b].keys():raise ValueError('aggregate walls not paired')
            paired[c+' - '+b]=cluster_stats([values[c][s]-values[b][s] for s in sorted(values[c])])
    return dict(rule='equal mean over frozen opponents; resample shared walls',opponents=opponents,
        scores={c:cluster_stats(list(v.values())) for c,v in values.items()},paired_differences=paired)


def run(candidates, opponents, manifest, output, seed, walls, workers=6, purpose='development', resume=False, registry_root='runs/final_v3'):
    if walls<2 or workers<1 or not candidates or not opponents:raise ValueError('Arena needs >=2 walls, workers, candidates and opponents')
    if set(candidates)&{'random','teacher'}:raise ValueError('Reserved policy name used for checkpoint')
    output=Path(output);paths=dict(candidates)
    for name,path in opponents.items():
        if path is not None:
            if name in paths and paths[name]!=path:raise ValueError('policy name collision')
            paths[name]=path
    identities={name:identity(path) for name,path in paths.items()}
    manifest=Path(manifest);bc=json.loads(manifest.read_text());excluded=set(bc['groups'])
    for start,n in HISTORICAL_RANGES:excluded.update(wall_hash(s) for s in range(start,start+n))
    # Repeated development comparisons may intentionally reuse the same wall
    # set. A final population may never overlap any development population.
    for registered in Path(registry_root).glob('**/arena_config.json'):
        previous=json.loads(registered.read_text())
        if previous['purpose']!=purpose:excluded.update(previous['wall_ids'])
    for registered in Path(registry_root).glob('**/training_walls.json'):
        excluded.update(json.loads(registered.read_text())['planned_wall_ids'])
    ids=[wall_hash(s) for s in range(seed,seed+walls)]
    if len(set(ids))!=walls or set(ids)&excluded:raise ValueError('arena wall populations overlap')
    config=dict(purpose=purpose,seed_start=seed,walls=walls,wall_ids=ids,seats=[0,1,2,3],identities=identities,
        candidates=list(candidates),opponents=list(opponents),bc_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        selection_rule='equal-weight mean reward over all named frozen opponents',bootstrap_replicates=5000,
        bootstrap_seed=20261119,excluded_historical_ranges=HISTORICAL_RANGES)
    # JSON serialization normalizes tuples, which matters for resume equality.
    config=json.loads(json.dumps(config));records=[];elapsed=0.
    if output.exists():
        if not resume:raise FileExistsError(output)
        if json.loads((output/'arena_config.json').read_text())!=config:raise ValueError('arena resume identity/config mismatch')
        raw=(output/'games.jsonl').read_bytes() if (output/'games.jsonl').exists() else b''
        complete=raw[:raw.rfind(b'\n')+1]
        records=[json.loads(line) for line in complete.splitlines()]
        # Each persisted job must contain all four seats; discard only the
        # incomplete trailing wall-job, retaining all complete earlier jobs.
        if len(records)%4:records=records[:len(records)//4*4]
        (output/'games.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        if (output/'progress.json').exists():elapsed=json.loads((output/'progress.json').read_text())['seconds']
    else:
        output.mkdir(parents=True);(output/'arena_config.json').write_text(json.dumps(config,indent=2))
    completed={(r['seed'],r['candidate'],r['opponent']) for r in records}
    jobs=[(s,c,o) for s in range(seed,seed+walls) for o in opponents for c in candidates if (s,c,o) not in completed]
    start=time.monotonic()
    with (output/'games.jsonl').open('a') as log, ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(paths,list(excluded))) as pool:
        for i,rows in enumerate(pool.map(worker,jobs)):
            records.extend(rows)
            for r in rows:log.write(json.dumps(r)+'\n')
            log.flush()
            progress=dict(jobs_done=len(records)//4,jobs_total=len(jobs)+len(completed),seconds=elapsed+time.monotonic()-start)
            (output/'progress.json').write_text(json.dumps(progress))
            if i%20==0:print(json.dumps(progress),flush=True)
    result=summarize(records);result.update(aggregate=aggregate(result['wall_records']),games=len(records),seconds=elapsed+time.monotonic()-start)
    (output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['aggregate']),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',action='append',required=True);p.add_argument('--opponent',action='append',required=True)
    p.add_argument('--manifest',default='reports/round3/raw/split_manifest.json');p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=110000000);p.add_argument('--walls',type=int,default=32);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--registry-root',default='runs/final_v3');p.add_argument('--purpose',choices=['development','final','confirmation'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    candidates=dict(x.split('=',1) for x in a.candidate);opponents={}
    for x in a.opponent:
        if '=' in x:k,v=x.split('=',1);opponents[k]=v
        else:
            if x not in ('random','teacher'):raise ValueError('unknown opponent')
            opponents[x]=None
    run(candidates,opponents,a.manifest,a.output,a.seed,a.walls,a.workers,a.purpose,a.resume,a.registry_root)

if __name__=='__main__':main()
