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
    rows=json.loads(Path(results).read_text())['wall_records'];x={};reference_walls=None
    for family,names in families.items():
        matrix=[];wallsets=[]
        for name in names:
            grouped={}
            for row in rows:
                if row['candidate']==name:grouped.setdefault(row['seed'],[]).append(row)
            if not grouped:raise ValueError('missing training seed candidate '+name)
            wallsets.append(sorted(grouped));matrix.append([np.mean([r['reward'] for r in grouped[s]]) for s in sorted(grouped)])
        if any(s!=wallsets[0] for s in wallsets):raise ValueError('unpaired seed arenas')
        if reference_walls is not None and reference_walls!=wallsets[0]:raise ValueError('unpaired family walls')
        reference_walls=wallsets[0]
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


def confirmatory(results,selection,output):
    arena=json.loads(Path(results).read_text());sealed=json.loads(Path(selection).read_text());roles=sealed['roles'];rows=arena['wall_records']
    pairs=[(roles['selected_bc'],'D_original'),(roles['selected_bc'],roles['same_recipe_d']),(roles['selected_ppo'],roles['selected_bc'])]
    def wall_means(name):
        grouped={}
        for r in rows:
            if r['candidate']==name:grouped.setdefault(r['seed'],[]).append(r['reward'])
        if any(len(v)!=5 for v in grouped.values()):raise ValueError('confirmation requires five frozen opponents')
        return {k:float(np.mean(v)) for k,v in grouped.items()}
    out={};rng=np.random.RandomState(20261119)
    for a,b in pairs:
        av,bv=wall_means(a),wall_means(b)
        if av.keys()!=bv.keys() or len(av)!=sealed['final_walls']:raise ValueError('incomplete paired final arena')
        diff=np.array([av[s]-bv[s] for s in sorted(av)]);n=len(diff)
        samples=diff[rng.randint(0,n,size=(20000,n))].mean(1)
        out[a+' - '+b]=dict(mean=float(diff.mean()),walls=n,se=float(diff.std(ddof=1)/np.sqrt(n)),
            ci95=np.quantile(samples,[.025,.975]).tolist(),bonferroni_family_ci95=np.quantile(samples,[.025/len(pairs),1-.025/len(pairs)]).tolist())
    result=dict(contrasts=out,bootstrap_replicates=20000,seed=20261119,method='paired wall bootstrap; three prespecified contrasts; Bonferroni percentile intervals are approximate',
                selection_sha256=hashlib.sha256(Path(selection).read_bytes()).hexdigest())
    Path(output).write_text(json.dumps(result,indent=2));return result


def followup(results,plan_path,output):
    plan=json.loads(Path(plan_path).read_text());arena=json.loads(Path(results).read_text())
    config=json.loads((Path(results).parent/'arena_config.json').read_text())
    if config['purpose']!='confirmation' or config['seed_start']!=plan['seed'] or config['walls']!=plan['walls']:
        raise ValueError('Follow-up population differs from registered plan')
    if set(config['candidates'])!=set(plan['candidates']):raise ValueError('Follow-up candidates changed')
    for name,item in plan['candidates'].items():
        if config['identities'][name]['sha256']!=item['sha256']:raise ValueError('Follow-up checkpoint changed')
    names=plan['primary_contrast'].split(' - ');rows=arena['wall_records'];opponents=config['opponents']
    def values(name):
        r={}
        for row in rows:
            if row['candidate']==name:r.setdefault(row['seed'],{})[row['opponent']]=row['reward']
        if len(r)!=plan['walls'] or any(set(x)!=set(opponents) for x in r.values()):raise ValueError('Incomplete follow-up walls/opponents')
        return r
    a,b=map(values,names)
    if a.keys()!=b.keys():raise ValueError('Unpaired follow-up walls')
    rng=np.random.RandomState(plan['bootstrap_seed']);seeds=sorted(a)
    diffs={o:np.array([a[s][o]-b[s][o] for s in seeds]) for o in opponents}
    diff=np.mean(list(diffs.values()),axis=0);samples=diff[rng.randint(0,len(diff),(plan['bootstrap_replicates'],len(diff)))].mean(1)
    contrast=dict(mean=float(diff.mean()),walls=len(diff),se=float(diff.std(ddof=1)/np.sqrt(len(diff))),ci95=np.quantile(samples,[.025,.975]).tolist())
    by_opponent={o:cluster_stats(v,seed=plan['bootstrap_seed']) for o,v in diffs.items()}
    accepted=contrast['ci95'][0]>0 and all(x['ci95'][1]>=-.02 for x in by_opponent.values())
    result=dict(plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),contrast=plan['primary_contrast'],statistics=contrast,
        opponent_differences=by_opponent,accepted=accepted,recommended_model=names[0] if accepted else names[1],
        method='One separately registered follow-up contrast, new500-wall population;20k paired wall bootstrap; original final pool is exploratory for this late-snapshot selection.')
    Path(output).write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    m=s.add_parser('merge');m.add_argument('--input',nargs='+',required=True);m.add_argument('--output',required=True)
    b=s.add_parser('seeds');b.add_argument('--input',required=True);b.add_argument('--families',required=True,help='JSON mapping family to aligned seed model names');b.add_argument('--output',required=True)
    c=s.add_parser('confirmatory');c.add_argument('--input',required=True);c.add_argument('--selection',required=True);c.add_argument('--output',required=True)
    f=s.add_parser('followup');f.add_argument('--input',required=True);f.add_argument('--plan',required=True);f.add_argument('--output',required=True)
    a=p.parse_args()
    if a.cmd=='merge':result=merge(a.input,a.output)
    elif a.cmd=='seeds':result=seed_stability(a.input,json.loads(a.families),a.output)
    elif a.cmd=='confirmatory':result=confirmatory(a.input,a.selection,a.output)
    else:result=followup(a.input,a.plan,a.output)
    print(json.dumps(result.get('aggregate',result),indent=2))

if __name__=='__main__':main()
