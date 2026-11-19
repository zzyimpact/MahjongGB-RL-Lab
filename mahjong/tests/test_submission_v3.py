import json
from pathlib import Path
import zipfile
import pytest
from mahjong.export_bot import export
from mahjong.verify_submission import verify,CODE_LIMIT


def test_submission_limits_and_unexpected_weights_fail(tmp_path):
    root=tmp_path/'release';result=export('artifacts/round3/D/checkpoint.pt',root)
    assert result['passed'] and result['code_zip']['expanded_bytes']<3500000
    with zipfile.ZipFile(root/'mahjong_bot.zip','a',zipfile.ZIP_DEFLATED) as z:z.writestr('checkpoint.pt',b'not allowed')
    with pytest.raises(ValueError,match='verification failed'):verify(root)
    check=json.loads((root/'submission_check.json').read_text());assert not check['checks']['resource_allowlist']


def test_compressible_oversized_source_still_fails(tmp_path):
    root=tmp_path/'release';export('artifacts/round3/D/checkpoint.pt',root)
    with zipfile.ZipFile(root/'mahjong_bot.zip','a',zipfile.ZIP_DEFLATED) as z:z.writestr('padding.py','#'+'x'*CODE_LIMIT+'\n')
    with pytest.raises(ValueError):verify(root)
    check=json.loads((root/'submission_check.json').read_text())
    assert check['checks']['zip_size'] and not check['checks']['expanded_size']


def test_single_file_rejects_other_same_architecture_weights(tmp_path):
    import os,subprocess,sys,torch
    root=tmp_path/'release';export('artifacts/round3/D/checkpoint.pt',root)
    state=torch.load(root/'mahjong_model.pt',map_location='cpu');state['actor.bias'][0]+=1
    wrong=tmp_path/'wrong.pt';torch.save(state,wrong)
    process=subprocess.run([sys.executable,str(root/'mahjong_bot.py')],input='1\n0 0 0\n',text=True,capture_output=True,cwd=tmp_path,
        env=dict(os.environ,BOT_MODEL_PATH=str(wrong)))
    assert process.returncode!=0 and 'SHA-256 mismatch' in process.stderr


def test_single_file_default_user_storage_path(tmp_path):
    import os,shutil,subprocess,sys
    root=tmp_path/'release';export('artifacts/round3/D/checkpoint.pt',root)
    (root/'data').mkdir();shutil.copy2(root/'mahjong_model.pt',root/'data/mahjong_model.pt')
    env=dict(os.environ);env.pop('BOT_MODEL_PATH',None)
    process=subprocess.run([sys.executable,'mahjong_bot.py'],input='1\n0 0 0\n\n\n1 0 0 0 0 W1 W1 W1 W1 W2 W3 W4 W5 W6 T1 T2 B1 F1\n2 B2\n',
        text=True,capture_output=True,cwd=root,env=env,timeout=15)
    assert process.returncode==0,process.stderr
    assert process.stdout.count('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')==3
