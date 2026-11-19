"""Structured visible context and candidate-discard information, feature v3.

Shanten features describe shape potential, NOT a guaranteed legal 8-fan win.
All candidate information is derived from the current player's visible state.
"""
from collections import Counter
from functools import lru_cache
import numpy as np
from MahjongGB import (RegularShanten, SevenPairsShanten, ThirteenOrphansShanten,
                       KnittedStraightShanten, HonorsAndKnittedTilesShanten)
from .observation import SeatView

TILES = tuple(SeatView.TILE_LIST)
INDEX = SeatView.OFFSET_TILE
CANDIDATE_DIM = 58
HISTORY_LENGTH = 16


@lru_cache(maxsize=65536)
def hand_shapes(hand, packs):
    """Native shanten + useful tiles; cache contains no hidden/live state."""
    regular = RegularShanten(hand)
    results = [regular]
    for fun in (SevenPairsShanten, ThirteenOrphansShanten, KnittedStraightShanten, HonorsAndKnittedTilesShanten):
        # The special forms require a closed 13-tile standing hand, except
        # knitted straight which can have one fixed meld (10 standing tiles).
        eligible = not packs or (fun is KnittedStraightShanten and packs == 1)
        results.append(fun(hand) if eligible else (14, ()))
    best = min(r[0] for r in results)
    useful = tuple(sorted({t for s, ts in results if s == best for t in ts}, key=INDEX.get))
    return tuple(int(r[0]) for r in results), useful


def candidate_features(view, base_obs):
    out = np.zeros((34, CANDIDATE_DIM), np.float32)
    unseen = base_obs.reshape(34, 36)[19, :34] * 4
    fixed = []
    for kind, tile, offer in view.packs[0]:
        if kind == 'CHI':
            fixed.extend(tile[0] + str(int(tile[1]) + d) for d in (-1, 0, 1))
        else:
            fixed.extend([tile] * (4 if kind == 'GANG' else 3))
    for action in view.valid:
        if not 2 <= action < 36:
            continue
        idx = action - 2
        hand = list(view.hand)
        hand.remove(TILES[idx])
        shape, useful = hand_shapes(tuple(sorted(hand, key=INDEX.get)), len(view.packs[0]))
        counts = Counter(hand)
        total = hand + fixed
        suited = [sum(t[0] == suit for t in total) for suit in 'WTB']
        x = out[idx]
        x[0] = 1
        x[1:6] = np.asarray(shape) / 16.
        x[6] = min(shape) / 16.
        x[7] = len(useful) / 64.
        x[8] = sum(unseen[INDEX[t]] for t in useful) / 128.
        x[9] = sum(n >= 2 for n in counts.values()) / 8.
        x[10] = sum(n >= 3 for n in counts.values()) / 4.
        x[11] = sum(n == 1 for n in counts.values()) / 16.
        x[12] = sum(t[0] in 'FJ' for t in total) / 32.
        x[13] = sum(t[0] in 'WTB' and t[1] in '19' for t in total) / 32.
        x[14] = max(suited) / 32.
        x[15] = sum(n > 0 for n in suited) / 4.
        x[16] = float(all(t[0] in 'WTB' and t[1] not in '19' for t in total))
        x[17] = float(all(k == 'GANG' and o == 0 for k, t, o in view.packs[0]))
        x[18] = len(view.packs[0]) / 4.
        x[19] = sum(min(counts[t], 3) for t in ('J1', 'J2', 'J3')) / 16.
        winds = {'F%d' % (view.seatWind + 1), 'F%d' % (view.prevalentWind + 1)}
        x[20] = sum(min(counts[t], 3) for t in winds) / 8.
        x[21] = sum(t in ('T2', 'T3', 'T4', 'T6', 'T8', 'J2') for t in total) / 32.
        x[22] = sum(t[0] in 'FJ' or t[1] in '19' for t in total) / 32.
        x[23] = unseen[idx] / 4.
        for tile in useful:
            x[24 + INDEX[tile]] = 1
    return out


def encode_v3(view, base):
    history = np.zeros((HISTORY_LENGTH, 3), np.int64)
    events = getattr(view, 'public_actions', [])[-HISTORY_LENGTH:]
    if events:
        history[-len(events):] = events
    melds = np.zeros((4, 4, 3), np.int64)
    for seat, packs in enumerate(view.packs):
        for slot, (kind, tile, offer) in enumerate(packs):
            concealed = kind == 'GANG' and offer == 0
            typ = 4 if concealed else {'CHI': 1, 'PENG': 2, 'GANG': 3}[kind]
            source = 0 if concealed else ((seat - (1 if kind == 'CHI' else offer)) % 4) + 1
            melds[seat, slot] = (typ, 0 if tile == 'CONCEALED' else INDEX[tile] + 1, source)
    return dict(base, history=history, melds=melds,
                candidate=candidate_features(view, base['obs']))
