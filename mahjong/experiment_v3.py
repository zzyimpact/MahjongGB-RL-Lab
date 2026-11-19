"""Persistent sequential BC job queue, safe to resume after interruption."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--configs',type=Path,nargs='+',required=True)
    p.add_argument('--data',type=Path,default=Path('data/final_v3'));p.add_argument('--root',type=Path,default=Path('runs/final_v3'))
    p.add_argument('--wait-for',type=Path);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    queue_id=hashlib.sha256(';'.join(str(p) for p in a.configs).encode()).hexdigest()[:8]
    status_path=a.root/('queue_'+queue_id+'.json');status={}
    if status_path.exists():status=json.loads(status_path.read_text())
    while a.wait_for and not a.wait_for.exists():time.sleep(10)
    for config in a.configs:
        name=config.stem;out=a.root/name
        if (out/'summary.json').exists():status[name]=dict(status='completed',summary=json.loads((out/'summary.json').read_text()));continue
        command=[sys.executable,'-m','mahjong.train_v3','--config',str(config),'--data',str(a.data),'--output',str(out)]
        if (out/'last.pt').exists():command.append('--resume')
        elif out.exists():raise RuntimeError('Incomplete job without a resumable last.pt: '+str(out))
        status[name]=dict(status='running',command=command,start_time=time.time());status_path.write_text(json.dumps(status,indent=2))
        with (a.root/(name+'.log')).open('a') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        status[name].update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode,end_time=time.time())
        status_path.write_text(json.dumps(status,indent=2))
        if result.returncode:raise RuntimeError('BC job failed: '+name)
    status_path.write_text(json.dumps(status,indent=2))

if __name__=='__main__':main()
