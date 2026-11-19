"""Measure exact-state expert-label ambiguity on training rows only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/final_v3');p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.data)
    keys=('obs','mask','history','melds','candidate','action');arrays={k:np.load(root/('train_'+k+'.npy'),mmap_mode='r') for k in keys}
    tables={'D':{},'V3':{}};n=len(arrays['action'])
    for start in range(0,n,4096):
        batch={k:np.array(v[start:start+4096]) for k,v in arrays.items()}
        for i,action in enumerate(batch['action']):
            h=hashlib.sha256();h.update(batch['obs'][i].tobytes());h.update(batch['mask'][i].tobytes());basic=h.digest()
            for k in ('history','melds','candidate'):h.update(batch[k][i].tobytes())
            for name,key in [('D',basic),('V3',h.digest())]:
                row=tables[name].setdefault(key,[0,0]);row[0]|=1<<int(action);row[1]+=1
    result={}
    for name,table in tables.items():
        conflicts=[r for r in table.values() if r[0]&(r[0]-1)]
        result[name]=dict(training_rows=n,unique_visible_states=len(table),states_with_multiple_expert_labels=len(conflicts),
            rows_in_conflicting_states=sum(r[1] for r in conflicts),row_conflict_fraction=sum(r[1] for r in conflicts)/n)
    result['interpretation']='Different labels can reflect expert stochasticity or omitted visible context; this is not proof of corrupt labels or hidden leakage. Training split only.'
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
