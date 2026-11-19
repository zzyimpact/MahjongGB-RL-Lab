"""Pure Python equivalents of the five PyMahjongGB 1.4 shanten interfaces.

Used only by exported V3 bots when the host provides the older fan-only
MahjongGB extension. No approximation, hidden information or learned weights.
The regular-form implementation enumerates suit partitions with bounded caches;
special-form and useful-tile semantics follow shanten.cpp from PyMahjongGB 1.4.

Upstream algorithm copyright (c) 2016-2023 Jeff Wang <summer_insects@163.com>.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from functools import lru_cache
from itertools import permutations

TILES = tuple(s + str(i) for s in 'WBT' for i in range(1, 10)) + tuple(
    'F' + str(i) for i in range(1, 5)) + tuple('J' + str(i) for i in range(1, 4))
INDEX = {t: i for i, t in enumerate(TILES)}
ORPHANS = (0, 8, 9, 17, 18, 26) + tuple(range(27, 34))
KNITTED = tuple(tuple(suit * 9 + offset + k for suit, offset in enumerate(order)
                      for k in (0, 3, 6)) for order in permutations(range(3)))


def _counts(hand, allowed):
    if not isinstance(hand, tuple) or len(hand) not in allowed:
        raise TypeError('Invalid standing hand size for shanten calculation')
    counts = [0] * 34
    for tile in hand:
        counts[INDEX[tile]] += 1
    if max(counts) > 4:
        raise TypeError('More than four copies of a tile')
    return tuple(counts)


def _result(distance, useful):
    return int(distance), tuple(TILES[i] for i in sorted(useful))


@lru_cache(maxsize=8192)
def _partitions(counts, suited):
    """All (melds, incomplete melds, head) partitions, ignoring isolated tiles."""
    i = next((i for i, n in enumerate(counts) if n), None)
    if i is None:
        return ((0, 0, 0),)
    out = set()

    def take(indices, added):
        remaining = list(counts)
        for idx in indices:
            remaining[idx] -= 1
        for m, t, p in _partitions(tuple(remaining), suited):
            a, b, c = m + added[0], t + added[1], p + added[2]
            if a <= 4 and c <= 1:
                out.add((a, min(b, 4 - a), c))

    take((i,), (0, 0, 0))
    if counts[i] >= 3:
        take((i, i, i), (1, 0, 0))
    if counts[i] >= 2:
        take((i, i), (0, 1, 0))
        take((i, i), (0, 0, 1))
    if suited:
        if i <= 6 and counts[i + 1] and counts[i + 2]:
            take((i, i + 1, i + 2), (1, 0, 0))
        for step in (1, 2):
            if i + step < 9 and counts[i + step]:
                take((i, i + step), (0, 1, 0))
    # More partners always dominate fewer partners at equal meld/head counts.
    best = {}
    for m, t, p in out:
        best[m, p] = max(t, best.get((m, p), -1))
    return tuple((m, t, p) for (m, p), t in sorted(best.items()))


@lru_cache(maxsize=16384)
def _distance(counts, fixed):
    if fixed == 4:
        return -1 if any(n >= 2 for n in counts) else 0
    # Retain maximum partner count for each combined meld/head count.
    states = {(fixed, 0): 0}
    for group, suited in ((counts[:9], True), (counts[9:18], True),
                          (counts[18:27], True), (counts[27:], False)):
        following = {}
        for (m, p), t in states.items():
            for dm, dt, dp in _partitions(group, suited):
                nm, np = m + dm, p + dp
                if nm > 4 or np > 1:
                    continue
                nt = min(t + dt, 4 - nm)
                following[nm, np] = max(nt, following.get((nm, np), -1))
        states = following
    return min(8 - 2 * m - t - p for (m, p), t in states.items())


def _regular(counts, fixed):
    distance = _distance(counts, fixed)
    useful = set()
    for i, n in enumerate(counts):
        # Match upstream, including the formal fifth-copy wait at shanten=0.
        if n == 4 and distance > 0:
            continue
        if n == 0:
            if i >= 27:
                continue
            base, rank = i // 9 * 9, i % 9
            if not any(counts[base + r] for r in range(max(0, rank - 2), min(9, rank + 3))):
                continue
        grown = list(counts)
        grown[i] += 1
        if _distance(tuple(grown), fixed) < distance:
            useful.add(i)
    return distance, useful


def RegularShanten(hand):
    counts = _counts(hand, (1, 4, 7, 10, 13))
    return _result(*_regular(counts, (13 - len(hand)) // 3))


def SevenPairsShanten(hand):
    counts = _counts(hand, (13,))
    return _result(6 - sum(n // 2 for n in counts),
                   {i for i, n in enumerate(counts) if n % 2})


def ThirteenOrphansShanten(hand):
    counts = _counts(hand, (13,))
    pair = any(counts[i] >= 2 for i in ORPHANS)
    return _result(13 - int(pair) - sum(counts[i] > 0 for i in ORPHANS),
                   {i for i in ORPHANS if not pair or not counts[i]})


def KnittedStraightShanten(hand):
    counts = _counts(hand, (10, 13))
    best, useful = 99, set()
    for pattern in KNITTED:
        remainder = list(counts)
        missing = set()
        for i in pattern:
            if remainder[i]:
                remainder[i] -= 1
            else:
                missing.add(i)
        distance, extra = _regular(tuple(remainder), (13 - len(hand)) // 3 + 3)
        distance += len(missing)
        current = missing | extra
        if distance < best:
            best, useful = distance, current
        elif distance == best:
            useful |= current
    return _result(best, useful)


def HonorsAndKnittedTilesShanten(hand):
    counts = _counts(hand, (13,))
    best, useful = 99, set()
    for pattern in KNITTED:
        needed = pattern + tuple(range(27, 34))
        distance = 13 - sum(counts[i] > 0 for i in needed)
        current = {i for i in needed if not counts[i]}
        if distance < best:
            best, useful = distance, current
        elif distance == best:
            useful |= current
    return _result(best, useful)
