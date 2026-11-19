"""Deployment-only checks. No training, game simulation or score experiments."""
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import zipfile

import MahjongGB
import numpy as np
import pytest

from mahjong.botzone import portable_shanten as portable
from mahjong.botzone.runner import _initial, _next_request
from mahjong.botzone.session import Session
from mahjong.export_bot import export
from mahjong.features import encode

NAMES = ('RegularShanten', 'SevenPairsShanten', 'ThirteenOrphansShanten',
         'KnittedStraightShanten', 'HonorsAndKnittedTilesShanten')
DEAL = '1 0 0 0 0 W1 W1 W1 W1 W2 W3 W4 W5 W6 T1 T2 B1 F1'
REQUESTS = ['0 0 0', DEAL, '2 B2']


def test_portable_five_shapes_match_native():
    rng = random.Random(20261119)
    wall = [t for t in portable.TILES for _ in range(4)]
    hands = []
    for n in (1, 4, 7, 10, 13):
        for _ in range(100):
            hands.append(tuple(rng.sample(wall, n)))
            # Dense suited shapes exercise overlapping meld/partner partitions.
            hands.append(tuple(rng.sample(wall[:36], n)))
    hands.extend(tuple(x.split()) for x in (
        'W1 W1 W1 W1 W2 W3 W4 W5 W6 W7 W8 W9 W9',
        'W1 W9 B1 B9 T1 T9 F1 F2 F3 F4 J1 J2 J3',
        'W1 W4 W7 B2 B5 B8 T3 T6 T9 F1 F1 F1 J1',
        'W1 W1 W2 W2 W3 W3 B4 B4 B5 B5 T6 T6 J1',
    ))
    for hand in hands:
        for name in NAMES:
            if len(hand) != 13 and name != 'RegularShanten':
                if not (name == 'KnittedStraightShanten' and len(hand) == 10):
                    continue
            expected = getattr(MahjongGB, name)(hand)
            actual = getattr(portable, name)(hand)
            assert (actual[0], set(actual[1])) == (expected[0], set(expected[1])), (name, hand)


def test_portable_candidate_tensor_equals_training(monkeypatch):
    from mahjong import features_v3
    for hand in (DEAL, '1 0 0 0 0 W1 W4 W7 B2 B5 B8 T3 T6 T9 F1 F1 F1 J1'):
        session = Session(lambda a: int(np.flatnonzero(a['action_mask'])[0]))
        session.handle('0 0 0'); session.handle(hand); session.view.request2obs('Draw B2')
        expected = encode(session.view, 'v3')
        with monkeypatch.context() as patch:
            for name in NAMES:
                patch.setattr(features_v3, name, getattr(portable, name))
            features_v3.hand_shapes.cache_clear()
            actual = encode(session.view, 'v3')
        features_v3.hand_shapes.cache_clear()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_simple_initial_consumes_nonempty_metadata():
    session = Session(lambda _: 0)
    stream = io.StringIO('0 0 0\nold-data\nold-globaldata\n' + DEAL + '\n')
    assert _initial(session, '1', stream) == ('simple', '0 0 0')
    assert stream.readline().strip() == DEAL


@pytest.mark.parametrize('line', ['2 B2', '"2 B2"', '{"request":"2 B2"}',
                                  '{"requests":["2 B2"],"responses":[]}'])
def test_long_running_request_formats(line):
    assert _next_request(line) == '2 B2'


def _run(root, filename, text, extra=None, old_native=False):
    env = dict(os.environ)
    env.pop('BOT_MODEL_PATH', None)
    env.pop('BOT_SHANTEN_BACKEND', None)
    if extra:
        env.update(extra)
    if old_native:
        prelude = ('import MahjongGB, types, sys, runpy; '
                   'old=types.ModuleType("MahjongGB"); '
                   'old.MahjongFanCalculator=MahjongGB.MahjongFanCalculator; '
                   'sys.modules["MahjongGB"]=old; '
                   'runpy.run_path(sys.argv[1], run_name="__main__")')
        command = [sys.executable, '-c', prelude, filename]
    else:
        command = [sys.executable, filename]
    result = subprocess.run(command, cwd=root, env=env, input=text, text=True,
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_unique_weights_json_simple_cold_and_fan_only_v3(tmp_path):
    import shutil
    root = tmp_path / 'release'
    export('artifacts/round4/V3_legacy_e01/checkpoint.pt', root, model_id='V3_BC')
    (root / 'data').mkdir()
    shutil.copy2(root / 'mahjong_V3_BC.pt', root / 'data/mahjong_V3_BC.pt')
    first = json.dumps(dict(requests=[REQUESTS[0]], responses=[])) + '\n'
    live = first + '\n'.join(REQUESTS[1:]) + '\n'
    native = _run(root, 'botzone_V3_BC.zip', live)
    fallback = _run(root, 'botzone_V3_BC.zip', live, old_native=True)
    single = _run(root, 'botzone_V3_BC.py', live, old_native=True)
    def responses(output):
        return [json.loads(line)['response'] for line in output.splitlines() if line.startswith('{')]
    assert responses(native) == responses(fallback) == responses(single)
    assert 'python' in fallback and 'native' in native
    expected = responses(native)
    assert len(expected) == 3 and expected[:2] == ['PASS', 'PASS']
    cold = json.dumps(dict(requests=REQUESTS, responses=expected[:2], data='old', globaldata='old')) + '\n'
    assert responses(_run(root, 'botzone_V3_BC.zip', cold, old_native=True)) == expected[-1:]
    simple = '1\n' + REQUESTS[0] + '\nold-data\nold-globaldata\n' + '\n'.join(REQUESTS[1:]) + '\n'
    simple_out = _run(root, 'botzone_V3_BC.zip', simple, old_native=True).splitlines()
    assert simple_out[::5] == expected
    assert simple_out[4::5] == ['>>>BOTZONE_REQUEST_KEEP_RUNNING<<<'] * 3
    # Native checkpoint weights must be bit-identical after deployment export.
    import torch
    checkpoint = torch.load('artifacts/round4/V3_legacy_e01/checkpoint.pt', map_location='cpu')
    weights = torch.load(root / 'mahjong_V3_BC.pt', map_location='cpu')
    assert all(torch.equal(t, checkpoint['policy']['module.' + k]) for k, t in weights.items())


def test_one_shot_json_without_keep_marker(tmp_path):
    import shutil
    root = tmp_path / 'release'
    export('artifacts/round3/D/checkpoint.pt', root, model_id='D', keep_running=False)
    (root / 'data').mkdir()
    shutil.copy2(root / 'mahjong_D.pt', root / 'data/mahjong_D.pt')
    out = _run(root, 'botzone_D.zip', json.dumps(dict(requests=['0 0 0'], responses=[])) + '\n')
    assert len(out.splitlines()) == 1 and json.loads(out)['response'] == 'PASS'
