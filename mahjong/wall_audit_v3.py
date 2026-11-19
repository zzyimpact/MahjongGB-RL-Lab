"""Verify physical-wall independence across every registered population."""
import argparse
import hashlib
import json
from pathlib import Path
from .arena_v3 import HISTORICAL_RANGES,wall_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/final_v3');p.add_argument('--output',required=True);a=p.parse_args()
    manifest=json.loads(Path('reports/round3/raw/split_manifest.json').read_text())
    groups={s:{w for w,t in manifest['groups'].items() if t==s} for s in ('train','val','test')}
    groups['historical']={wall_hash(i) for start,n in HISTORICAL_RANGES for i in range(start,start+n)}
    groups.update(development=set(),final=set(),ppo_training=set());registrations=[]
    for path in Path(a.root).glob('**/arena_config.json'):
        r=json.loads(path.read_text());assert r['wall_ids']==[wall_hash(i) for i in range(r['seed_start'],r['seed_start']+r['walls'])]
        groups[r['purpose']].update(r['wall_ids']);registrations.append(dict(path=str(path),purpose=r['purpose'],walls=r['walls'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for path in Path(a.root).glob('**/training_walls.json'):
        r=json.loads(path.read_text());groups['ppo_training'].update(r['planned_wall_ids']);registrations.append(dict(path=str(path),purpose='ppo_training',planned_walls=len(r['planned_wall_ids'])))
    overlaps={};names=sorted(groups)
    for i,n in enumerate(names):
        for b in names[:i]:
            overlap=len(groups[n]&groups[b]);overlaps[n+' vs '+b]=overlap
            if overlap:raise AssertionError('wall leakage: '+n+' vs '+b)
    result=dict(unique_walls={k:len(v) for k,v in groups.items()},overlaps=overlaps,registrations=registrations,
        note='Development repeats share walls intentionally; PPO ablations may share training walls. Those are unions, not extra independent walls.')
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result['unique_walls']))

if __name__=='__main__':main()
