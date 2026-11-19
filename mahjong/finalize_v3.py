"""Seal candidate identities before opening held-out test metrics or final arena."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import torch
from .arena import identity,load_checkpoint
from .train_v3 import Dataset,evaluate


def seal(candidates,output,reason):
    output=Path(output)
    if output.exists():raise FileExistsError('Do not rewrite a final-selection seal')
    selection=dict(sealed_time=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        rationale=reason,selection_rule='development mean reward across five frozen opponents; validation CE and complexity secondary',
        candidates={k:identity(v) for k,v in candidates.items()},final_seed=112000000,final_walls=300,
        heldout_test='same 51 walls as round3, not used for this round selection; historical results already public')
    output.write_text(json.dumps(selection,indent=2));return selection


def test(seal_path,data,output,device='cpu'):
    selected=json.loads(Path(seal_path).read_text());output=Path(output)
    if output.exists():raise FileExistsError('Test metrics already exist')
    torch.set_num_threads(2);results={}
    for name,record in selected['candidates'].items():
        path=Path(record['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Sealed checkpoint mutated')
        model,cfg=load_checkpoint(path);model.to(device)
        start=time.monotonic();results[name]=dict(metrics=evaluate(model,Dataset(data,'test',cfg),device),seconds=time.monotonic()-start,identity=record)
    result=dict(seal_sha256=hashlib.sha256(Path(seal_path).read_bytes()).hexdigest(),models=results)
    output.write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('seal');s.add_argument('--candidate',action='append',required=True);s.add_argument('--output',required=True);s.add_argument('--reason',required=True)
    t=sub.add_parser('test');t.add_argument('--seal',required=True);t.add_argument('--data',default='data/final_v3');t.add_argument('--output',required=True);t.add_argument('--device',default='cpu')
    a=p.parse_args()
    result=seal(dict(x.split('=',1) for x in a.candidate),a.output,a.reason) if a.command=='seal' else test(a.seal,a.data,a.output,a.device)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
