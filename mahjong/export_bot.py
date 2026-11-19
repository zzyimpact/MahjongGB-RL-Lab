"""Separate weights plus an allowlisted ZIP and a standalone Python code file."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import zipfile
import torch
from .model import make_model, model_spec

# Explicit allowlist prevents new training modules/caches entering a release.
SOURCES=('__init__.py','rules.py','observation.py','features.py','features_v3.py',
         'model.py','model_v3.py','botzone/__init__.py','botzone/session.py',
         'botzone/inference.py','botzone/runner.py','botzone/shanten_backend.py',
         'botzone/portable_shanten.py')

STARTUP = ('import os\n'
           'for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):\n'
           '    os.environ.setdefault(name, "1")\n')

SINGLE_LOADER='''# Optional single-file equivalent of the ZIP. Upload its paired weights separately.
# External runtime dependencies: numpy, torch, PyMahjongGB. No pip install here.
import importlib.abc
import importlib.util
import json
import sys

__STARTUP__

_BUNDLE = __BUNDLE__
class _MahjongBundle(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def _key(self, fullname):
        name = fullname.replace('.', '/')
        return name + '/__init__.py' if name + '/__init__.py' in _BUNDLE else name + '.py'
    def find_spec(self, fullname, path=None, target=None):
        if self._key(fullname) in _BUNDLE:
            return importlib.util.spec_from_loader(fullname, self, is_package=self.is_package(fullname))
        return None
    def is_package(self, fullname):
        return self._key(fullname).endswith('/__init__.py')
    def get_filename(self, fullname):
        return '/__mahjong_bundle__/' + self._key(fullname)
    def get_data(self, path):
        return _BUNDLE[path.split('/__mahjong_bundle__/', 1)[-1]].encode('utf-8')
    def get_source(self, fullname):
        return _BUNDLE[self._key(fullname)]
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.__file__ = self.get_filename(module.__name__)
        if self.is_package(module.__name__):
            module.__path__ = [module.__file__.rsplit('/', 1)[0]]
        exec(compile(self.get_source(module.__name__), module.__file__, 'exec'), module.__dict__)

sys.meta_path.insert(0, _MahjongBundle())
from mahjong.botzone.runner import main
if __name__ == '__main__':
    main()
'''


def export(checkpoint, output, model_id=None, keep_running=True):
    if model_id is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', model_id):
        raise ValueError('model_id must be a safe unique filename identifier')
    checkpoint=Path(checkpoint);output=Path(output);state=torch.load(checkpoint,map_location='cpu');cfg=state['config'];model=make_model(cfg)
    weights={k[len('module.'):]:v.detach().cpu() for k,v in state['policy'].items() if k.startswith('module.')}
    model.load_state_dict(weights,strict=True);output.mkdir(parents=True,exist_ok=False)
    weight_name='mahjong_%s.pt' % model_id if model_id else 'mahjong_model.pt'
    zip_name='botzone_%s.zip' % model_id if model_id else 'mahjong_bot.zip'
    single_name='botzone_%s.py' % model_id if model_id else 'mahjong_bot.py'
    torch.save(weights,output/weight_name,_use_new_zipfile_serialization=False,pickle_protocol=2)
    source=Path(__file__).parent
    config=dict(model_spec(cfg),capacity=cfg.get('capacity',128),weight_path='data/'+weight_name,
                weight_sha256=hashlib.sha256((output/weight_name).read_bytes()).hexdigest(),
                model_id=model_id,keep_running=keep_running,io_mode='auto')
    bundle={'mahjong/'+name:(source/name).read_text() for name in SOURCES}
    # Only the deployment import is adapted; feature formulas are unchanged.
    native_import='from MahjongGB import (RegularShanten,'
    if bundle['mahjong/features_v3.py'].count(native_import)!=1:
        raise ValueError('V3 shanten import changed; review deployment adapter')
    bundle['mahjong/features_v3.py']=bundle['mahjong/features_v3.py'].replace(
        native_import,'from .botzone.shanten_backend import (RegularShanten,')
    bundle['mahjong/bot_config.json']=json.dumps(config)
    with zipfile.ZipFile(output/zip_name,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('__main__.py',STARTUP+'from mahjong.botzone.runner import main\nmain()\n')
        for name,content in bundle.items():z.writestr(name,content)
    (output/single_name).write_text(SINGLE_LOADER.replace('__STARTUP__',STARTUP).replace('__BUNDLE__',repr(bundle)))
    manifest=dict(game='mahjong',training_counters=state.get('counters',{}),iteration=state.get('iteration',0),config=cfg,
        source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()},
        weight_file=weight_name,zip_file=zip_name,single_file=single_name,
        model_id=model_id,keep_running=keep_running,weight_embedding=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from .verify_submission import verify
    return verify(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-id',help='Unique user-storage weight filename, so several bots can coexist')
    p.add_argument('--no-keep-running',action='store_true');a=p.parse_args()
    print(json.dumps(export(a.checkpoint,a.output,a.model_id,not a.no_keep_running),indent=2))

if __name__=='__main__':main()
