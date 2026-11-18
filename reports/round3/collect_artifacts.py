"""Copy only round-3 evidence and selected model artifacts; never raw datasets."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/codex_v2';OUT=ROOT/'reports/round3';RAW=OUT/'raw';RAW.mkdir(exist_ok=True)

def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def compress(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with src.open('rb') as inp,dst.open('wb') as target,gzip.GzipFile(fileobj=target,mode='wb',mtime=0,filename='') as out:
        shutil.copyfileobj(inp,out)

for p in RUN.iterdir():
    if p.is_file() and p.suffix in ('.json','.txt'):copy(p,RAW/p.name)
for name in ['A','B','C','D','A0','C_seed2','D_seed2','PPO_C','test_metrics','benchmarks']:
    directory=RUN/name
    if not directory.exists():continue
    for p in directory.iterdir():
        if p.is_file() and p.suffix in ('.json','.jsonl'):copy(p,RAW/name/p.name)
    if name=='PPO_C':
        for p in directory.glob('eval_*/*.json'):copy(p,RAW/name/p.relative_to(directory))
        for p in directory.glob('eval_*/games.jsonl'):compress(p,RAW/name/p.relative_to(directory).with_suffix('.jsonl.gz'))
for name in ['arena_smoke','arena_dev','arena_final','arena_ppo_final','arena_seed2']:
    directory=RUN/name
    if not directory.exists():continue
    for p in directory.glob('*.json'):copy(p,RAW/name/p.name)
    if (directory/'games.jsonl').exists():compress(directory/'games.jsonl',RAW/name/'games.jsonl.gz')
copy(ROOT/'data/codex_v2/basic/manifest.json',RAW/'split_manifest.json')
manifest=json.loads((ROOT/'data/codex_v2/enhanced_v2/manifest.json').read_text())
(RAW/'enhanced_preparation.json').write_text(json.dumps({k:v for k,v in manifest.items() if k not in ('games','groups','reconstruction')},indent=2))
ART=ROOT/'artifacts/round3'
for name in ['A','B','C','D','A0']:
    copy(RUN/name/'checkpoint.pt',ART/name/'checkpoint.pt');copy(RUN/name/'config.json',ART/name/'config.json')
copy(RUN/'PPO_C/ppo_best.pt',ART/'PPO_C/ppo_best.pt')
copy(ROOT/'runs/bc_v1/checkpoint.pt',ART/'references/bc_v1.pt')
copy(ROOT/'runs/rl_v1/checkpoint.pt',ART/'references/rl_v1.pt')
for name in ['mahjong_v2_C','mahjong_v2_D']:
    for p in (ROOT/'submissions'/name).iterdir():copy(p,ART/name/p.name)
records=[]
for p in sorted(ART.rglob('*')):
    if p.is_file():records.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
# Record locally retained full checkpoints too, including repeat seeds and the PPO last iterate.
local=[]
for p in sorted(RUN.rglob('*.pt')):
    local.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(OUT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(dict(versioned=records,local_checkpoints=local),indent=2))
(OUT/'SOURCE_COMMIT.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True))
print('Versioned artifacts bytes:',sum(r['bytes'] for r in records))
