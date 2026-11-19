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
