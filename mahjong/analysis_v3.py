"""Merge comparable arenas and estimate paired wall/training-seed uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .arena import summarize,cluster_stats
from .arena_v3 import aggregate


def merge(directories,output):
    records={};sources=[]
    for directory in directories:
        path=Path(directory)/'games.jsonl';sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for line in path.read_text().splitlines():
            row=json.loads(line);key=(row['candidate'],row['opponent'],row['seed'],row['seat'])
            if key in records and records[key]!=row:raise ValueError('duplicate evaluation identity with different result')
            records[key]=row
    result=summarize(list(records.values()));result['aggregate']=aggregate(result['wall_records']);result['sources']=sources
    Path(output).write_text(json.dumps(result,indent=2));return result


def seed_stability(results,families,output):
    rows=json.loads(Path(results).read_text())['wall_records'];x={}
    for family,names in families.items():
        matrix=[];wallsets=[]
        for name in names:
            grouped={}
            for row in rows:
                if row['candidate']==name:grouped.setdefault(row['seed'],[]).append(row)
            if not grouped:raise ValueError('missing training seed candidate '+name)
            wallsets.append(sorted(grouped));matrix.append([np.mean([r['reward'] for r in grouped[s]]) for s in sorted(grouped)])
        if any(s!=wallsets[0] for s in wallsets):raise ValueError('unpaired seed arenas')
        x[family]=np.asarray(matrix)
    if len({v.shape for v in x.values()})!=1:raise ValueError('families need matched seed and wall counts')
    nseeds,nwalls=next(iter(x.values())).shape;rng=np.random.RandomState(20261119);boot={k:[] for k in x}
    for _ in range(5000):
        a=rng.randint(0,nseeds,nseeds);b=rng.randint(0,nwalls,nwalls)
        for k,v in x.items():boot[k].append(float(v[a[:,None],b[None,:]].mean()))
    result=dict(training_seeds=nseeds,walls=nwalls,method='two-level paired bootstrap: resample training-seed pairs and common walls; only 3 seeds limits population inference',families={},paired={})
    for k,v in x.items():result['families'][k]=dict(mean=float(v.mean()),seed_means=v.mean(1).tolist(),seed_std=float(v.mean(1).std(ddof=1)),hierarchical_ci95=np.quantile(boot[k],[.025,.975]).tolist(),wall_only=cluster_stats(v.mean(0)))
    keys=sorted(x)
    for i,k in enumerate(keys):
        for b in keys[:i]:result['paired'][k+' - '+b]=dict(mean=float((x[k]-x[b]).mean()),hierarchical_ci95=np.quantile(np.asarray(boot[k])-boot[b],[.025,.975]).tolist())
    Path(output).write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    m=s.add_parser('merge');m.add_argument('--input',nargs='+',required=True);m.add_argument('--output',required=True)
    b=s.add_parser('seeds');b.add_argument('--input',required=True);b.add_argument('--families',required=True,help='JSON mapping family to aligned seed model names');b.add_argument('--output',required=True)
    a=p.parse_args();result=merge(a.input,a.output) if a.cmd=='merge' else seed_stability(a.input,json.loads(a.families),a.output)
    print(json.dumps(result.get('aggregate',result),indent=2))

if __name__=='__main__':main()
