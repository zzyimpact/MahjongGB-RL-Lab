"""Versioned, public-information-only features shared by replay/RL/Botzone.

v2 has 34 planes; the first six are byte-identical to v1. Seat-indexed
planes use [self, next, opposite, previous]. Tile padding stays zero.
"""
import numpy as np
OBS_SHAPE = (6, 4, 9)
ACTION_COUNT = 235
FEATURE_CHANNELS = {'basic': 6, 'enhanced': 34}


def encode(view, feature_set='basic'):
    if feature_set == 'v3':
        from .features_v3 import encode_v3
        return encode_v3(view, encode(view, 'enhanced'))
    if feature_set not in FEATURE_CHANNELS:
        raise ValueError('Unsupported feature set: ' + feature_set)
    observation = np.zeros((FEATURE_CHANNELS[feature_set], 36), dtype=np.float32)
    observation[0, view.OFFSET_TILE['F%d' % (view.seatWind + 1)]] = 1
    observation[1, view.OFFSET_TILE['F%d' % (view.prevalentWind + 1)]] = 1
    for tile in set(view.hand):
        observation[2:2 + view.hand.count(tile), view.OFFSET_TILE[tile]] = 1
    if feature_set == 'enhanced':
        concealed = np.zeros(34, dtype=np.float32)
        for seat in range(4):
            for tile in view.history[seat]:
                observation[6+seat, view.OFFSET_TILE[tile]] += .25
            hidden_kongs = 0
            for kind, tile, offer in view.packs[seat]:
                if tile == 'CONCEALED':
                    hidden_kongs += 1
                    continue
                if kind == 'CHI':
                    tiles = [tile[0]+str(int(tile[1])+d) for d in (-1,0,1)]
                else:
                    tiles = [tile] * (4 if kind == 'GANG' else 3)
                for t in tiles:
                    observation[10+seat, view.OFFSET_TILE[t]] += .25
                if kind == 'GANG' and offer == 0:
                    hidden_kongs += 1
                    if seat == 0:
                        concealed[view.OFFSET_TILE[tile]] += 4
            observation[20+seat,:34] = view.tileWall[seat]/32.
            observation[24+seat,:34] = hidden_kongs/4.
        event = getattr(view, 'last_event', '').split()
        if event and event[0] == 'Draw':
            observation[28,:34] = 1
        elif len(event) >= 3 and event[0] == 'Player':
            seat = (int(event[1])-view.seatWind)%4
            verb = event[2]
            if verb in ('Play','BuGang') and seat != 0:
                # BuGang must use the current event tile, never stale curTile.
                observation[14+seat,view.OFFSET_TILE[event[3]]] = 1
                observation[31 if verb == 'BuGang' else 29,:34] = 1
            elif verb in ('Chi','Peng') and seat == 0:
                observation[30,:34] = 1
        for tile,idx in view.OFFSET_TILE.items():
            public = view.shownTiles.get(tile,0)
            observation[18,idx] = public/4.
            observation[19,idx] = (4-public-view.hand.count(tile)-concealed[idx])/4.
        observation[32,:34] = float(view.wallLast)
        observation[33,:34] = float(view.isAboutKong or
            (event and event[0]=='Draw' and getattr(view,'last_draw_kong',False)))
        if observation.min() < 0 or observation.max() > 1:
            raise ValueError('Inconsistent public tile/wall counts')
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    mask[view.valid] = True
    return {'obs':observation.reshape(FEATURE_CHANNELS[feature_set],4,9),'action_mask':mask}
