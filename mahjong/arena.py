"""Frozen-opponent duplicate arena: four seats per wall; wall-cluster inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .model import make_model
from .features import encode
from .gamecore.engine import GameCore
from .observation import SeatView
from .teacher import HeuristicTeacher


def load_checkpoint(path):
    state=torch.load(path,map_location='cpu');model=make_model(state['config'])
    weights={k[len('module.'):]:v for k,v in state['policy'].items() if k.startswith('module.')}
    model.load_state_dict(weights,strict=True);model.eval()
    return model,state['config']

def identity(path):
    path=Path(path)
    state=torch.load(path,map_location='cpu')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,config=state['config'])

def init_worker(paths,excluded):
    global MODELS,TEACHER,EXCLUDED
    torch.set_num_threads(1)
    MODELS={k:load_checkpoint(p) for k,p in paths.items()};TEACHER=HeuristicTeacher();EXCLUDED=set(excluded)

def choose(name,view,rng):
    if len(view.valid)==1:return view.valid[0]
    if name=='random':return int(rng.choice(view.valid))
    if name=='teacher':return TEACHER.choose(view)
    model,config=MODELS[name]
    arrays=encode(view,config.get('feature_set','basic'))
    with torch.inference_mode():
        logits,_=model(torch.from_numpy(arrays['obs']).unsqueeze(0))
        mask=torch.from_numpy(arrays['action_mask']).unsqueeze(0)
        return int(logits.masked_fill(~mask,-1e9).argmax(-1))

def play(seed,seat,candidate,opponent):
    core=GameCore();core.r.seed(seed);core.reset()
    wall=hashlib.sha256(core.originalTileWall.encode()).hexdigest()
    if wall in EXCLUDED:raise ValueError('Arena wall overlaps BC data')
    views=[SeatView(i) for i in range(4)];offsets=[0]*4
    rngs=[np.random.RandomState(np.random.SeedSequence([seed,seat,i,123]).generate_state(1)[0]) for i in range(4)]
    turns=decisions=0
    while not core.done:
        for i in range(4):
            events=core.agents[i].events
            for event in events[offsets[i]:]:views[i].request2obs(event)
            offsets[i]=len(events)
        commands={}
        for i in core.obs:
            action=choose(candidate if i==seat else opponent,views[i],rngs[i])
            if action not in views[i].valid:raise ValueError('illegal policy action')
            commands[core.agent_names[i]]=views[i].action2response(action);decisions+=1
        core.step(commands);turns+=1
        if core.invalid_info:raise RuntimeError(str(core.invalid_info))
        if turns>2000:raise RuntimeError('unterminated game')
    scores=list(core.reward or [0]*4)
    return dict(seed=seed,wall_id=wall,seat=seat,candidate=candidate,opponent=opponent,
                scores=scores,reward=scores[seat]/32.,score_difference=scores[seat]-np.mean([scores[i] for i in range(4) if i!=seat]),
                win=scores[seat]>0,loss=scores[seat]<0,draw=max(scores)==0,turns=turns,decisions=decisions)

def worker(job):
    seed,candidate,opponent=job
    return [play(seed,seat,candidate,opponent) for seat in range(4)]

def cluster_stats(values,seed=20261119):
    x=np.asarray(values,dtype=float);n=len(x)
    if n<2:return dict(mean=float(x.mean()),walls=n,se=None,ci95=None)
    rng=np.random.RandomState(seed)
    means=x[rng.randint(0,n,size=(5000,n))].mean(1)
    return dict(mean=float(x.mean()),walls=n,se=float(x.std(ddof=1)/np.sqrt(n)),ci95=np.quantile(means,[.025,.975]).tolist())

def summarize(records):
    groups={};wall_rows=[]
    for r in records:groups.setdefault((r['candidate'],r['opponent'],r['seed']),[]).append(r)
    for (c,o,s),rows in groups.items():
        if sorted(r['seat'] for r in rows)!=list(range(4)) or len({r['wall_id'] for r in rows})!=1:
            raise ValueError('incomplete or mismatched duplicate wall')
        wall_rows.append(dict(candidate=c,opponent=o,seed=s,wall_id=rows[0]['wall_id'],
            **{k:float(np.mean([r[k] for r in rows])) for k in ('reward','score_difference','win','loss','draw')}))
    comparisons={};summary={}
    for c,o in sorted({(r['candidate'],r['opponent']) for r in records}):
        rows=[r for r in wall_rows if r['candidate']==c and r['opponent']==o]
        summary[c+' vs '+o]={k:cluster_stats([r[k] for r in rows]) for k in ('reward','score_difference','win','loss','draw')}
    for o in sorted({r['opponent'] for r in records}):
        candidates=sorted({r['candidate'] for r in records if r['opponent']==o})
        for i,c in enumerate(candidates):
            for b in candidates[:i]:
                arows={r['seed']:r for r in wall_rows if r['candidate']==c and r['opponent']==o}
                brows={r['seed']:r for r in wall_rows if r['candidate']==b and r['opponent']==o}
                if arows.keys()!=brows.keys():raise ValueError('unpaired walls')
                if any(arows[s]['wall_id']!=brows[s]['wall_id'] for s in arows):raise ValueError('unpaired wall identity')
                comparisons[c+' - '+b+' vs '+o]=cluster_stats([arows[s]['reward']-brows[s]['reward'] for s in sorted(arows)])
    return dict(summary=summary,paired_differences=comparisons,wall_records=wall_rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',action='append',required=True,help='name=checkpoint')
    p.add_argument('--opponent',action='append',required=True,help='random, teacher, or name=checkpoint')
    p.add_argument('--walls',type=int,default=300);p.add_argument('--seed',type=int,default=9000000)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--purpose',choices=['development','final'],required=True)
    a=p.parse_args()
    if a.output.exists():p.error('output already exists')
    if a.walls<2 or a.workers<1:p.error('need >=2 walls and >=1 worker')
    candidates=dict(v.split('=',1) for v in a.candidate);opponents={};paths=dict(candidates)
    for v in a.opponent:
        if v in ('random','teacher'):opponents[v]=None
        else:
            name,path=v.split('=',1)
            if name in paths and paths[name]!=path:raise ValueError('policy name collision')
            paths[name]=path;opponents[name]=path
    identities={name:identity(path) for name,path in paths.items()}
    manifest=json.loads(a.manifest.read_text());excluded=list(manifest['groups'])
    # Assert all generated walls are distinct before dispatch, and disjoint from every BC split.
    wall_ids=[]
    for seed in range(a.seed,a.seed+a.walls):
        core=GameCore();core.r.seed(seed);core.reset();wall_ids.append(hashlib.sha256(core.originalTileWall.encode()).hexdigest())
    if len(set(wall_ids))!=a.walls or set(wall_ids)&set(excluded):raise ValueError('wall overlap')
    a.output.mkdir(parents=True)
    config=dict(purpose=a.purpose,seed_start=a.seed,walls=a.walls,seats=[0,1,2,3],identities=identities,
                opponents=list(opponents),candidates=list(candidates),bootstrap_seed=20261119,bootstrap_replicates=5000,
                bc_manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),wall_ids=wall_ids)
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    jobs=[(s,c,o) for s in range(a.seed,a.seed+a.walls) for o in opponents for c in candidates]
    records=[];start=time.monotonic()
    with (a.output/'games.jsonl').open('w') as log, ProcessPoolExecutor(max_workers=a.workers,initializer=init_worker,initargs=(paths,excluded)) as pool:
        for i,rows in enumerate(pool.map(worker,jobs)):
            records.extend(rows)
            for r in rows:log.write(json.dumps(r)+'\n')
            log.flush()
            if i%max(1,len(candidates)*len(opponents)*10)==0:
                print(json.dumps(dict(wall_jobs=i+1,total=len(jobs),seconds=time.monotonic()-start)),flush=True)
    result=summarize(records);result.update(seconds=time.monotonic()-start,games=len(records))
    (a.output/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(seconds=result['seconds'],games=len(records),summary=result['summary'])),flush=True)

if __name__=='__main__':main()
