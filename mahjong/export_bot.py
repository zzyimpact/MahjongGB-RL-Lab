"""Separate weights plus an allowlisted ZIP and a standalone Python code file."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import torch
from .model import make_model, model_spec

# Explicit allowlist prevents new training modules/caches entering a release.
SOURCES=('__init__.py','rules.py','observation.py','features.py','features_v3.py',
         'model.py','model_v3.py','botzone/__init__.py','botzone/session.py',
         'botzone/inference.py','botzone/runner.py')

SINGLE_LOADER='''# Single code file; upload mahjong_model.pt separately to user storage.
# External runtime dependencies: numpy, torch, PyMahjongGB. No pip install here.
import importlib.abc
import importlib.util
import json
import sys

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


def export(checkpoint, output):
    checkpoint=Path(checkpoint);output=Path(output);state=torch.load(checkpoint,map_location='cpu');cfg=state['config'];model=make_model(cfg)
    weights={k[len('module.'):]:v.detach().cpu() for k,v in state['policy'].items() if k.startswith('module.')}
    model.load_state_dict(weights,strict=True);output.mkdir(parents=True,exist_ok=False)
    torch.save(weights,output/'mahjong_model.pt',_use_new_zipfile_serialization=False,pickle_protocol=2)
    source=Path(__file__).parent
    config=dict(model_spec(cfg),capacity=cfg.get('capacity',128),weight_path='data/mahjong_model.pt',weight_sha256=hashlib.sha256((output/'mahjong_model.pt').read_bytes()).hexdigest())
    bundle={'mahjong/'+name:(source/name).read_text() for name in SOURCES}
    bundle['mahjong/bot_config.json']=json.dumps(config)
    with zipfile.ZipFile(output/'mahjong_bot.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('__main__.py','from mahjong.botzone.runner import main\nmain()\n')
        for name,content in bundle.items():z.writestr(name,content)
    (output/'mahjong_bot.py').write_text(SINGLE_LOADER.replace('__BUNDLE__',repr(bundle)))
    manifest=dict(game='mahjong',training_counters=state.get('counters',{}),iteration=state.get('iteration',0),config=cfg,
        source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()},weight_embedding=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from .verify_submission import verify
    return verify(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(export(a.checkpoint,a.output),indent=2))

if __name__=='__main__':main()
