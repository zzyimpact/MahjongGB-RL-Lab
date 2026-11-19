"""Evaluate completed BC jobs on the registered development population."""
import argparse
import json
from pathlib import Path
import time
from .arena_v3 import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--root',type=Path,default=Path('runs/final_v3'))
    p.add_argument('--workers',type=int,default=4);p.add_argument('--walls',type=int,default=128);p.add_argument('--seed',type=int,default=111000000);a=p.parse_args()
    opponents={'random':None,'teacher':None,'rl_v1':'artifacts/round3/references/rl_v1.pt','C':'artifacts/round3/C/checkpoint.pt','D':'artifacts/round3/D/checkpoint.pt'}
    state={};status=a.root/'development_queue.json'
    for name in a.models:
        model=a.root/name;output=a.root/('arena_'+name)
        while not (model/'summary.json').exists():time.sleep(10)
        if (output/'results.json').exists():result=json.loads((output/'results.json').read_text())
        else:
            state[name]={'status':'running'};status.write_text(json.dumps(state,indent=2))
            result=run({name:str(model/'checkpoint.pt')},opponents,'reports/round3/raw/split_manifest.json',output,a.seed,a.walls,a.workers,'development',output.exists(),a.root)
        state[name]=dict(status='completed',score=result['aggregate']['scores'][name]);status.write_text(json.dumps(state,indent=2))
        print(json.dumps(dict(model=name,**state[name])),flush=True)

if __name__=='__main__':main()
