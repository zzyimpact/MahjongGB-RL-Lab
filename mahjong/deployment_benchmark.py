"""Execute exported ZIP and single-file bots in clean temporary directories."""
import argparse
import json
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import numpy as np
from .tests.test_export import play_game
from .verify_submission import verify


class Process:
    def __init__(self,root,script,model,stderr):
        self.err=open(stderr,'w');self.proc=subprocess.Popen(['/usr/bin/time','-l',sys.executable,script],cwd=root,
            env=dict(os.environ,BOT_MODEL_PATH=str(Path(model).resolve())),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.err,text=True,bufsize=1);self.latencies=[];self.first=True
    def send(self,request):
        lines=['1',request] if self.first else [request];self.first=False;return self.raw(lines)
    def raw(self,lines):
        start=time.perf_counter();self.proc.stdin.write('\n'.join(lines)+'\n');self.proc.stdin.flush()
        if not select.select([self.proc.stdout],[],[],30)[0]:raise TimeoutError('exported bot request exceeded 30 seconds')
        response=self.proc.stdout.readline().strip()
        for _ in range(3):self.proc.stdout.readline()
        if self.proc.stdout.readline().strip()!='>>>BOTZONE_REQUEST_KEEP_RUNNING<<<':raise RuntimeError('exported bot protocol failed')
        self.latencies.append(time.perf_counter()-start);return response
    def close(self):
        self.proc.stdin.close();self.proc.wait(timeout=30);self.proc.stdout.close();self.err.close()
        if self.proc.returncode:raise RuntimeError('exported bot exited with '+str(self.proc.returncode))


def benchmark(directory,games=8):
    directory=Path(directory).resolve();check=verify(directory);results={};all_histories={}
    for kind,script in [('zip','__main__.py'),('single','mahjong_bot.py')]:
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            if kind=='zip':
                with zipfile.ZipFile(directory/'mahjong_bot.zip') as z:z.extractall(root)
            else:shutil.copy2(directory/'mahjong_bot.py',root/script)
            stderr=root/'time.txt';process=Process(root,script,directory/'mahjong_model.pt',stderr);histories=[];scores=[]
            try:
                for i in range(games):
                    h,score=play_game(150000000+i,process.send,np.random.RandomState(1000+i),candidate_seat=i%4)
                    histories.append(h);scores.append(score)
            finally:process.close()
            rss=re.search(r'(\d+)\s+maximum resident set size',stderr.read_text());lat=np.asarray(process.latencies)*1000
            cold=[]
            for index in sorted({0,len(histories)-1,max(range(len(histories)),key=lambda i:len(histories[i]))}):
                h=histories[index];p=Process(root,script,directory/'mahjong_model.pt',root/'cold_time.txt')
                lines=[str(len(h))]
                for req,res in h[:-1]:lines.extend([req,res])
                try:response=p.raw(lines+[h[-1][0]])
                finally:p.close()
                if response!=h[-1][1]:raise AssertionError('cold-history mismatch')
                cold.append(dict(game_index=index,turns=len(h),seconds=p.latencies[0]))
            results[kind]=dict(games=games,seats=[i%4 for i in range(games)],illegal_actions=0,requests=len(lat),scores=scores,
                cold_start_ms=float(lat[0]),longest_history_turns=max(map(len,histories)),steady_mean_ms=float(lat[1:].mean()),steady_p95_ms=float(np.quantile(lat[1:],.95)),
                steady_max_ms=float(lat[1:].max()),max_request_ms=float(lat.max()),peak_rss_bytes=int(rss.group(1)) if rss else None,cold_replays=cold)
            all_histories[kind]=histories
    if all_histories['zip']!=all_histories['single']:raise AssertionError('ZIP and single-file decisions differ')
    result=dict(submission_check=check,runtime=dict(python=sys.version,torch=__import__('torch').__version__),local_only=True,results=results,zip_single_responses_identical=True)
    (directory/'runtime_check.json').write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--games',type=int,default=8);a=p.parse_args();print(json.dumps(benchmark(a.directory,a.games),indent=2))

if __name__=='__main__':main()
