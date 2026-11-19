"""Verify published models, lossless raw records and separate Botzone exports."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from .arena import load_checkpoint
from .verify_submission import verify


def digest(data):return hashlib.sha256(data).hexdigest()


def verify_sealed_evaluation(raw,selection_sha256):
    """Bind final evaluation and test records to the pre-evaluation seal."""
    raw=Path(raw);selection_path=raw/'selection.json'
    if digest(selection_path.read_bytes())!=selection_sha256:raise ValueError('Selection seal identity mismatch')
    selected=json.loads(selection_path.read_text())
    arena=json.loads((raw/'arena_final/arena_config.json').read_text())
    test=json.loads((raw/'test_metrics.json').read_text())
    confirm=json.loads((raw/'confirmatory_statistics.json').read_text())
    walls=json.loads((raw/'wall_audit_final.json').read_text())
    names=set(selected['candidates'])
    if set(arena['candidates'])!=names or set(test['models'])!=names:raise ValueError('Sealed candidate population differs from final evaluation')
    if arena['purpose']!='final' or arena['walls']!=selected['final_walls'] or arena['seed_start']!=selected['final_seed']:
        raise ValueError('Final arena population differs from seal')
    if test['seal_sha256']!=selection_sha256 or confirm['selection_sha256']!=selection_sha256:
        raise ValueError('Final statistics use a different selection seal')
    for name,record in selected['candidates'].items():
        if arena['identities'][name]['sha256']!=record['sha256'] or test['models'][name]['identity']['sha256']!=record['sha256']:
            raise ValueError('Final evaluation checkpoint differs from seal: '+name)
    if any(walls['overlaps'].values()) or walls['unique_walls']['final']!=selected['final_walls']:
        raise ValueError('Final arena wall independence failed')
    return dict(candidates=len(names),final_walls=arena['walls'],test_and_arena_bound_to_seal=True,wall_populations_disjoint=True)


def verify_archive(manifest_path,release_root):
    manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text());models={};raw_count=0
    for name,r in manifest['models'].items():
        path=Path(r['published_path']);content=path.read_bytes()
        if len(content)!=r['bytes'] or digest(content)!=r['sha256']:raise ValueError('Model identity mismatch: '+name)
        model,cfg=load_checkpoint(path)
        models[name]=dict(path=str(path),sha256=r['sha256'],parameters=sum(p.numel() for p in model.parameters()),strict_load=True)
    for r in manifest['records']:
        path=Path(r['path']);content=path.read_bytes()
        if len(content)!=r['bytes'] or digest(content)!=r['sha256']:raise ValueError('Raw archive identity mismatch: '+str(path))
        source=gzip.decompress(content) if path.name.endswith('.jsonl.gz') else content
        if digest(source)!=r['source_sha256']:raise ValueError('Lossless source identity mismatch: '+str(path))
        raw_count+=1
    evaluation=verify_sealed_evaluation(manifest_path.parent/'raw',manifest['selection_sha256'])
    releases={};allowed={r['sha256'] for r in manifest['models'].values()}
    for path in sorted(Path(release_root).iterdir()):
        if not path.is_dir() or not (path/'manifest.json').exists():continue
        if json.loads((path/'manifest.json').read_text())['source_checkpoint_sha256'] not in allowed:raise ValueError('Export is not from a sealed published checkpoint')
        check=verify(path);runtime=json.loads((path/'runtime_check.json').read_text())
        if not runtime['zip_single_responses_identical'] or any(r['illegal_actions'] for r in runtime['results'].values()):raise ValueError('Recorded deployment run failed')
        releases[path.name]=dict(passed=check['passed'],model_bytes=check['model']['bytes'],zip_bytes=check['code_zip']['bytes'],expanded_bytes=check['code_zip']['expanded_bytes'],single_bytes=check['single_file']['bytes'])
    if not releases:raise ValueError('No published deployment artifacts')
    return dict(passed=True,manifest_sha256=digest(manifest_path.read_bytes()),models=models,verified_raw_files=raw_count,releases=releases,evaluation=evaluation,
                scope='Checks archive identity and saved runtime results; does not repeat online or local gameplay',online_botzone_verified=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='reports/round4/ARTIFACT_MANIFEST.json');p.add_argument('--releases',default='artifacts/round4/releases');p.add_argument('--output',default='reports/round4/RELEASE_VERIFICATION.json');a=p.parse_args()
    result=verify_archive(a.manifest,a.releases);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
