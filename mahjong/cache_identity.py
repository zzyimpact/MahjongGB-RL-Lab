"""Semantic feature-source guard shared by generation and training."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

FEATURE_SOURCES=('features.py','features_v3.py','observation.py','botzone/session.py','replay.py','rules.py')

def fingerprint():
    root=Path(__file__).parent
    return dict(feature_version='v3',sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in FEATURE_SOURCES},
                PyMahjongGB=importlib.metadata.version('PyMahjongGB'))

def check(root, create=False):
    path=Path(root)/'feature_provenance.json';current=fingerprint()
    if not path.exists():
        if not create:raise ValueError('Missing feature provenance; rebuild cache')
        path.write_text(json.dumps(current,indent=2))
    if json.loads(path.read_text())!=current:raise ValueError('Stale feature cache: source or native shanten version changed')
    return current
