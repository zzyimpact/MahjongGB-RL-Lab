"""Fail-closed Botzone release budget and source identity checks."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

MODEL_LIMIT=268435456
CODE_LIMIT=4000000


def verify(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    weight=root/'mahjong_model.pt';archive=root/'mahjong_bot.zip';single=root/'mahjong_bot.py'
    entries=[];unsafe=[];bot_config=None
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or p.suffix not in ('.py','.json') or any(x in p.parts for x in ('data','runs','logs','tests')):
                unsafe.append(info.filename)
            if info.flag_bits&1:unsafe.append(info.filename+' (encrypted)')
            content=z.read(info)
            if info.filename=='mahjong/bot_config.json':bot_config=json.loads(content)
            if len(content)!=info.file_size:raise ValueError('ZIP size inconsistency')
            if p.suffix=='.py':ast.parse(content.decode(),filename=info.filename,feature_version=(3,6))
            entries.append(dict(path=info.filename,bytes=info.file_size,sha256=hashlib.sha256(content).hexdigest()))
    ast.parse(single.read_text(),filename=str(single),feature_version=(3,6))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (weight,archive,single)}
    checks=dict(model_size=weight.stat().st_size<=MODEL_LIMIT,zip_size=archive.stat().st_size<CODE_LIMIT,
        expanded_size=sum(x['bytes'] for x in entries)<CODE_LIMIT,single_file_size=single.stat().st_size<CODE_LIMIT,
        resource_allowlist=not unsafe,runtime_weight_identity=bot_config is not None and bot_config.get('weight_sha256')==hashes['mahjong_model.pt'],hash_identity=all(manifest['files'].get(k)==v for k,v in hashes.items()))
    report=dict(passed=all(checks.values()),checks=checks,model=dict(path=str(weight.resolve()),bytes=weight.stat().st_size,limit=MODEL_LIMIT),
        code_zip=dict(path=str(archive.resolve()),bytes=archive.stat().st_size,expanded_bytes=sum(x['bytes'] for x in entries),limit_exclusive=CODE_LIMIT),
        single_file=dict(path=str(single.resolve()),bytes=single.stat().st_size,limit_exclusive=CODE_LIMIT),resources=entries,unsafe_resources=unsafe,sha256=hashes,
        online_botzone_verified=False)
    (root/'submission_check.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:raise ValueError('Submission verification failed: '+json.dumps(checks))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(verify(a.directory),indent=2))

if __name__=='__main__':main()
