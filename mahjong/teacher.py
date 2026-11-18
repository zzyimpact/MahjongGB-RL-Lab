"""Heuristic teacher policy for BC data generation and diagnostic evaluation.

Uses only seat-visible information from SeatView (own hand, public melds,
discards, wall counts). Greedy shanten minimization with tile-utility
tie-breaks; no fan-value planning and no defense, so it is a modest teacher,
not a strong bot. Pure Python: no torch, no framework imports.
"""

from .observation import SeatView

OFFSET_ACT = SeatView.OFFSET_ACT
OFFSET_TILE = SeatView.OFFSET_TILE
TILE_LIST = SeatView.TILE_LIST


def tile_counts(tiles):
    counts = [0] * 34
    for tile in tiles:
        counts[OFFSET_TILE[tile]] += 1
    return counts


def shanten_standard(counts, open_melds):
    """Shanten of the standard 4-sets+pair form; -1 means a complete hand."""
    c = list(counts)
    best = [8]

    def rec(i, mentsu, taatsu, pair):
        while i < 34 and c[i] == 0:
            i += 1
        m = mentsu + open_melds
        if i == 34:
            t = taatsu
            if m + t > 4:
                t = 4 - m
            s = 8 - 2 * m - t - pair
            if s < best[0]:
                best[0] = s
            return
        if c[i] >= 3:
            c[i] -= 3
            rec(i, mentsu + 1, taatsu, pair)
            c[i] += 3
        if i < 27 and i % 9 <= 6 and c[i + 1] > 0 and c[i + 2] > 0:
            c[i] -= 1
            c[i + 1] -= 1
            c[i + 2] -= 1
            rec(i, mentsu + 1, taatsu, pair)
            c[i] += 1
            c[i + 1] += 1
            c[i + 2] += 1
        if c[i] >= 2:
            if not pair:
                c[i] -= 2
                rec(i, mentsu, taatsu, 1)
                c[i] += 2
            if m + taatsu < 4:
                c[i] -= 2
                rec(i, mentsu, taatsu + 1, pair)
                c[i] += 2
        if m + taatsu < 4:
            if i < 27 and i % 9 <= 7 and c[i + 1] > 0:
                c[i] -= 1
                c[i + 1] -= 1
                rec(i, mentsu, taatsu + 1, pair)
                c[i] += 1
                c[i + 1] += 1
            if i < 27 and i % 9 <= 6 and c[i + 2] > 0:
                c[i] -= 1
                c[i + 2] -= 1
                rec(i, mentsu, taatsu + 1, pair)
                c[i] += 1
                c[i + 2] += 1
        saved = c[i]
        c[i] = 0
        rec(i + 1, mentsu, taatsu, pair)
        c[i] = saved

    rec(0, 0, 0, 0)
    return best[0]


def shanten(counts, open_melds):
    """Minimum over standard form and seven pairs (MCR allows 4-of-a-kind as two pairs)."""
    best = shanten_standard(counts, open_melds)
    if open_melds == 0:
        pairs = sum(x // 2 for x in counts)
        seven = 6 - pairs
        if seven < best:
            best = seven
    return best


def _best_discard_shanten(counts, open_melds):
    best = 9
    for i in range(34):
        if counts[i]:
            counts[i] -= 1
            s = shanten(counts, open_melds)
            counts[i] += 1
            if s < best:
                best = s
    return best


class HeuristicTeacher:
    """Deterministic given the view; choose() returns a legal action index."""

    def __init__(self, aggressive_melds=False):
        self.aggressive_melds = aggressive_melds

    def choose(self, view):
        valid = view.valid
        if not valid:
            raise ValueError("teacher asked to act without legal actions")
        if OFFSET_ACT["Hu"] in valid:
            return OFFSET_ACT["Hu"]
        plays = [a for a in valid if OFFSET_ACT["Play"] <= a < OFFSET_ACT["Chi"]]
        if plays and len(plays) == len(valid):
            return self._discard(view, plays)
        # Own draw turn: Play / AnGang / BuGang choices.
        if any(OFFSET_ACT["AnGang"] <= a < OFFSET_ACT["BuGang"] for a in valid) or any(
            a >= OFFSET_ACT["BuGang"] for a in valid
        ):
            return self._draw_turn(view, valid)
        # Response to a discard: Chi / Peng / Gang / Pass.
        return self._response(view, valid)

    def _utility(self, view, idx, counts):
        """How much we want to keep tile idx; lower means better discard."""
        c = counts[idx]
        u = 6.0 * (c - 1)
        if idx < 27:
            num = idx % 9
            if num >= 1:
                u += 3.0 * counts[idx - 1]
            if num <= 7:
                u += 3.0 * counts[idx + 1]
            if num >= 2:
                u += 1.0 * counts[idx - 2]
            if num <= 6:
                u += 1.0 * counts[idx + 2]
            if num == 0 or num == 8:
                u -= 0.5
        else:
            wind = idx - 27
            if wind < 4 and (wind == view.seatWind or wind == view.prevalentWind):
                u += 2.0
            if wind >= 4:
                u += 2.0
        u -= 0.25 * view.shownTiles[TILE_LIST[idx]]
        return u

    def _discard(self, view, plays):
        counts = tile_counts(view.hand)
        open_melds = len(view.packs[0])
        best_key = None
        best_action = plays[0]
        for action in plays:
            idx = action - OFFSET_ACT["Play"]
            counts[idx] -= 1
            s = shanten(counts, open_melds)
            counts[idx] += 1
            key = (s, self._utility(view, idx, counts))
            if best_key is None or key < best_key:
                best_key = key
                best_action = action
        return best_action

    def _draw_turn(self, view, valid):
        counts = tile_counts(view.hand)
        open_melds = len(view.packs[0])
        before = _best_discard_shanten(counts, open_melds)
        for action in valid:
            if OFFSET_ACT["AnGang"] <= action < OFFSET_ACT["BuGang"]:
                idx = action - OFFSET_ACT["AnGang"]
                if counts[idx] >= 4:
                    counts[idx] -= 4
                    after = shanten(counts, open_melds + 1)
                    counts[idx] += 4
                    if after <= before:
                        return action
            elif action >= OFFSET_ACT["BuGang"]:
                idx = action - OFFSET_ACT["BuGang"]
                counts[idx] -= 1
                after = _best_discard_shanten(counts, open_melds)
                counts[idx] += 1
                if after <= before:
                    return action
        return self._discard(view, [a for a in valid if OFFSET_ACT["Play"] <= a < OFFSET_ACT["Chi"]])

    def _response(self, view, valid):
        counts = tile_counts(view.hand)
        open_melds = len(view.packs[0])
        before = _best_discard_shanten(counts, open_melds)
        tile = view.curTile
        idx = OFFSET_TILE[tile]
        # Open kong: keep shape, gain a replacement draw and fan potential.
        for action in valid:
            if OFFSET_ACT["Gang"] <= action < OFFSET_ACT["AnGang"]:
                kidx = action - OFFSET_ACT["Gang"]
                counts[kidx] -= 3
                after = shanten(counts, open_melds + 1)
                counts[kidx] += 3
                if after <= before:
                    return action
        # Peng: strictly improves, or keeps shanten with a value-tile triplet.
        for action in valid:
            if OFFSET_ACT["Peng"] <= action < OFFSET_ACT["Gang"]:
                pidx = action - OFFSET_ACT["Peng"]
                counts[pidx] -= 2
                after = _best_discard_shanten(counts, open_melds + 1)
                counts[pidx] += 2
                value_tile = pidx >= 27 and (
                    pidx - 27 >= 4
                    or pidx - 27 == view.seatWind
                    or pidx - 27 == view.prevalentWind
                )
                if after < before or (after == before and value_tile and before > 0):
                    return action
        # Chi: only when it strictly improves the hand.
        best_chi = None
        best_key = None
        for action in valid:
            if OFFSET_ACT["Chi"] <= action < OFFSET_ACT["Peng"]:
                t = (action - OFFSET_ACT["Chi"]) // 3
                middle = "WTB"[t // 7] + str(t % 7 + 2)
                num = int(middle[1])
                used = [middle[0] + str(num - 1), middle[0] + str(num + 1)]
                for u in used:
                    counts[OFFSET_TILE[u]] -= 1
                after = _best_discard_shanten(counts, open_melds + 1)
                for u in used:
                    counts[OFFSET_TILE[u]] += 1
                if after < before:
                    key = (after, sum(self._utility(view, OFFSET_TILE[u], counts) for u in used))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_chi = action
        if best_chi is not None:
            return best_chi
        return OFFSET_ACT["Pass"]