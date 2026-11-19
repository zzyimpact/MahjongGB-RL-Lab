"""Deployment adapter: native shanten if present, otherwise bundled Python."""
import os
import MahjongGB

NAMES = ('RegularShanten', 'SevenPairsShanten', 'ThirteenOrphansShanten',
         'KnittedStraightShanten', 'HonorsAndKnittedTilesShanten')
requested = os.environ.get('BOT_SHANTEN_BACKEND', 'auto')
if requested not in ('auto', 'native', 'python'):
    raise ValueError('BOT_SHANTEN_BACKEND must be auto, native or python')
available = all(hasattr(MahjongGB, name) for name in NAMES)
if requested == 'native' and not available:
    raise RuntimeError('This MahjongGB runtime has no complete native shanten API')
if available and requested != 'python':
    BACKEND = 'native'
    for name in NAMES:
        globals()[name] = getattr(MahjongGB, name)
else:
    from . import portable_shanten
    BACKEND = 'python'
    for name in NAMES:
        globals()[name] = getattr(portable_shanten, name)
