"""Evaluate completed BC jobs on the registered development population."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from .arena_v3 import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--root',type=Path,default=Path('runs/final_v3'))
    p.add_argument('--workers',type=int,default=4);p.add_argument('--walls',type=int,default=128);p.add_argument('--seed',type=int,default=111000000)
    p.add_argument('--epochs',type=int,nargs='+',help='Compare saved training stages instead of only the val-selected checkpoint');a=p.parse_args()
    opponents={'random':None,'teacher':None,'rl_v1':'artifacts/round3/references/rl_v1.pt','C':'artifacts/round3/C/checkpoint.pt','D':'artifacts/round3/D/checkpoint.pt'}
    queue_id=hashlib.sha256(json.dumps([a.models,a.epochs]).encode()).hexdigest()[:8]
    state={};status=a.root/('development_queue_'+queue_id+'.json')
    for name in a.models:
        model=a.root/name;output=a.root/('arena_'+name+('_epochs' if a.epochs else ''))
        while not (model/'summary.json').exists():time.sleep(10)
        if (output/'results.json').exists():result=json.loads((output/'results.json').read_text())
        else:
            state[name]={'status':'running'};status.write_text(json.dumps(state,indent=2))
            candidates={name:str(model/'checkpoint.pt')} if not a.epochs else {name+'_e%02d'%e:str(model/('epoch_%03d.pt'%e)) for e in a.epochs}
            result=run(candidates,opponents,'reports/round3/raw/split_manifest.json',output,a.seed,a.walls,a.workers,'development',output.exists(),a.root)
        state[name]=dict(status='completed',scores=result['aggregate']['scores']);status.write_text(json.dumps(state,indent=2))
        print(json.dumps(dict(model=name,**state[name])),flush=True)

if __name__=='__main__':main()
