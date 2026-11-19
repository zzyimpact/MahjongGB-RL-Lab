"""Archive sealed models and auditable experiment records without training caches."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(root,selection,artifacts,reports):
    root=Path(root);artifacts=Path(artifacts);reports=Path(reports)
    sealed=json.loads(Path(selection).read_text());artifacts.mkdir(parents=True,exist_ok=True)
    models={}
    for name,item in sealed['candidates'].items():
        source=Path(item['path'])
        if sha(source)!=item['sha256']:raise ValueError('Sealed model changed: '+name)
        # Historical references already live in version control; new models
        # are copied byte-for-byte, including their original optimizer state.
        if source.is_relative_to(Path('artifacts/round3')):
            target=source
        else:
            target=artifacts/name/'checkpoint.pt';target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)!=item['sha256']:raise FileExistsError('Refuse to overwrite different checkpoint')
            if not target.exists():shutil.copy2(source,target)
            (target.parent/'config.json').write_text(json.dumps(item['config'],indent=2)+'\n')
        models[name]=dict(source_path=str(source),published_path=str(target),sha256=sha(target),bytes=target.stat().st_size)
    raw=reports/'raw';raw.mkdir(parents=True,exist_ok=True);records=[]
    for source in sorted(root.rglob('*')):
        if not source.is_file() or source.suffix not in ('.json','.jsonl','.txt','.log'):continue
        rel=source.relative_to(root);target=raw/rel;target.parent.mkdir(parents=True,exist_ok=True)
        content=source.read_bytes()
        if source.suffix=='.jsonl':
            target=target.with_suffix('.jsonl.gz');target.write_bytes(gzip.compress(content,mtime=0))
        else:target.write_bytes(content)
        records.append(dict(source=str(source),path=str(target),source_sha256=hashlib.sha256(content).hexdigest(),sha256=sha(target),bytes=target.stat().st_size))
    result=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),selection_sha256=sha(selection),models=models,records=records,
                note='JSONL compressed losslessly; source SHA identifies decompressed bytes. Full caches and epoch snapshots remain local; selected checkpoints retain optimizer state.')
    (reports/'ARTIFACT_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/final_v3');p.add_argument('--selection',default='runs/final_v3/selection.json')
    p.add_argument('--artifacts',default='artifacts/round4');p.add_argument('--reports',default='reports/round4');a=p.parse_args()
    result=archive(a.root,a.selection,a.artifacts,a.reports);print(json.dumps(dict(models=result['models'],record_files=len(result['records'])),indent=2))


if __name__=='__main__':main()
