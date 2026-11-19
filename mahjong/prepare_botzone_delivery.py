"""Repackage existing trained models; check identity and IO, never train/play."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch
from .export_bot import export

MODELS = (
    ('C_v2', 'artifacts/round3/C/checkpoint.pt', 'Historical C BC'),
    ('D_v2', 'artifacts/round3/D/checkpoint.pt', 'Historical D BC'),
    ('D_e06', 'artifacts/round4/D_legacy_e06/checkpoint.pt', 'Retrained D BC epoch 6'),
    ('V3_BC', 'artifacts/round4/V3_legacy_e01/checkpoint.pt', 'Primary V3 BC'),
    ('V3_BC_seed3', 'artifacts/round4/V3_legacy_seed3_e01/checkpoint.pt', 'Exploratory V3 BC seed 3'),
    ('V3_PPO', 'artifacts/round4/PPO_legacy_best/checkpoint.pt', 'Short PPO development-best'),
    ('V3_PPO_long', 'artifacts/round4/PPO_extended_last/checkpoint.pt', '3072-game PPO last'),
)
REQUESTS = ('0 0 0', '1 0 0 0 0 W1 W1 W1 W1 W2 W3 W4 W5 W6 T1 T2 B1 F1', '2 B2')


def _execute(directory, entry, payload, old_native=False):
    env = dict(os.environ)
    env.pop('BOT_MODEL_PATH', None)
    env.pop('BOT_SHANTEN_BACKEND', None)
    if old_native:
        prelude = ('import MahjongGB, types, sys, runpy; '
                   'old=types.ModuleType("MahjongGB"); '
                   'old.MahjongFanCalculator=MahjongGB.MahjongFanCalculator; '
                   'sys.modules["MahjongGB"]=old; '
                   'runpy.run_path(sys.argv[1], run_name="__main__")')
        command = [sys.executable, '-c', prelude, str(entry)]
    else:
        command = [sys.executable, str(entry)]
    p = subprocess.run(command, cwd=directory, env=env, input=payload,
                       text=True, capture_output=True, timeout=25)
    if p.returncode:
        raise RuntimeError(p.stderr)
    return p.stdout


def smoke(root, model_id, is_v3):
    """Three protocol requests, including one real nonforced policy decision."""
    with tempfile.TemporaryDirectory(prefix='botzone-upload-check-') as temporary:
        cwd = Path(temporary)
        (cwd / 'data').mkdir()
        shutil.copy2(root / ('mahjong_' + model_id + '.pt'), cwd / 'data')
        archive = root / ('botzone_' + model_id + '.zip')
        single = root / ('botzone_' + model_id + '.py')
        payload = json.dumps(dict(requests=[REQUESTS[0]], responses=[])) + '\n' + '\n'.join(REQUESTS[1:]) + '\n'
        def parse(output):
            return [json.loads(x) for x in output.splitlines() if x.startswith('{')]
        native = parse(_execute(cwd, archive, payload))
        expected = [x['response'] for x in native]
        assert len(expected) == 3 and expected[:2] == ['PASS', 'PASS']
        # Empty module except for the exact fan API documented by Botzone.
        compatible = parse(_execute(cwd, archive, payload, old_native=True))
        one_file = parse(_execute(cwd, single, payload, old_native=True))
        assert [x['response'] for x in compatible] == expected
        assert [x['response'] for x in one_file] == expected
        if is_v3:
            assert json.loads(compatible[0]['debug'])['shanten'] == 'python'
        simple = '1\n' + REQUESTS[0] + '\nold-data\nold-globaldata\n' + '\n'.join(REQUESTS[1:]) + '\n'
        lines = _execute(cwd, archive, simple, old_native=True).splitlines()
        assert lines[::5] == expected
        cold = json.dumps(dict(requests=list(REQUESTS), responses=expected[:2], data='old', globaldata='old')) + '\n'
        replay = parse(_execute(cwd, archive, cold, old_native=True))
        assert [x['response'] for x in replay] == expected[-1:]
        return dict(passed=True,request_count=3,responses=expected,
                    json_zip=True,json_single=True,simple_with_metadata=True,
                    cold_json_history=True,fan_only_extension=True,
                    startup_debug=json.loads(compatible[0]['debug']),
                    scope='Protocol smoke only; no game simulation, training or score experiment',
                    online_botzone_verified=False)


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for model_id, checkpoint, label in MODELS:
        root = output / model_id
        report = export(checkpoint, root, model_id=model_id)
        state = torch.load(checkpoint, map_location='cpu')
        saved = torch.load(root / ('mahjong_' + model_id + '.pt'), map_location='cpu')
        original = {k[len('module.'):]: t.cpu() for k, t in state['policy'].items() if k.startswith('module.')}
        assert set(saved) == set(original)
        assert all(torch.equal(saved[k], original[k]) for k in saved)
        runtime = smoke(root, model_id, state['config'].get('feature_set') == 'v3')
        (root / 'runtime_check.json').write_text(json.dumps(runtime, indent=2))
        row = dict(id=model_id,label=label,source_checkpoint=checkpoint,
                   source_checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                   weight_tensor_identity=True,code=str(Path(model_id)/('botzone_'+model_id+'.zip')),
                   single=str(Path(model_id)/('botzone_'+model_id+'.py')),
                   weight=str(Path(model_id)/('mahjong_'+model_id+'.pt')),
                   weight_bytes=report['model']['bytes'],zip_bytes=report['code_zip']['bytes'],
                   expanded_bytes=report['code_zip']['expanded_bytes'],sha256=report['sha256'],
                   deployment_checks_passed=runtime['passed'] and report['passed'])
        rows.append(row)
        print(model_id + ': identity, file limits and protocol smoke passed', flush=True)
    result = dict(scope='Deployment-only packaging of frozen checkpoints',models=rows,
                  training_runs=0,new_games=0,online_botzone_verified=False,
                  tested_python=sys.version,torch_version=torch.__version__,
                  platform_python36_syntax_checked=True,platform_runtime_executed=False)
    (output/'DELIVERY_MANIFEST.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.output)


if __name__=='__main__':
    main()
