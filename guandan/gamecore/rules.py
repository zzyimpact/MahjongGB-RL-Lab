"""Botzone card identities, declared hand types and comparison rules.

Physical card IDs are retained for responses. Wild hearts may substitute any
non-joker; claims describe the represented cards, not cards removed from hand.
"""

from collections import Counter

LEVELS = "A234567890JQK"


def rank(card):
    return card % 54 // 4 if card % 54 < 52 else 13 + card % 54 - 52


def strength(r, level):
    if r >= 13:
        return 15 + r - 13
    if r == LEVELS.index(level):
        return 14
    return 12 if r == 0 else r - 1


def wild(card, level):
    return card % 54 == LEVELS.index(level) * 4


def sequence_start(ranks, length):
    # A is low in A2345 and high in TJQKA; no circular K-A-2 sequences.
    for start in range(0, 15 - length):
        natural = list(range(start, start + length))
        candidate = {x % 13 for x in natural}
        if set(ranks) == candidate:
            return start
    return None


def classify(cards, level):
    """Return (family, comparison power, length), or None for an invalid hand."""
    n = len(cards)
    if not n:
        return ("pass", 0, 0)
    if any(type(c) is not int or not 0 <= c < 108 for c in cards):
        return None
    counts = Counter(rank(c) for c in cards)
    values = sorted(counts.values())
    if n == 4 and counts == {13: 2, 14: 2}:
        return ("rocket", 0, n)
    if len(counts) == 1:
        r = next(iter(counts))
        if n == 1 or n == 2:
            return ("single" if n == 1 else "pair", strength(r, level), n)
        if r < 13 and n == 3:
            return ("three", strength(r, level), n)
        if r < 13 and 4 <= n <= 10:
            return ("bomb", strength(r, level), n)
        return None
    if n == 5 and values == [2, 3]:
        r = next(r for r, count in counts.items() if count == 3)
        if r < 13:
            return ("set", strength(r, level), n)
    if max(counts) >= 13:
        return None
    if n == 5 and values == [1] * 5:
        start = sequence_start(counts, 5)
        if start is not None:
            flush = len({c % 4 for c in cards}) == 1
            return ("flush" if flush else "straight", start, n)
    if n == 6 and values in ([2, 2, 2], [3, 3]):
        start = sequence_start(counts, len(counts))
        if start is not None:
            return ("triple_pairs" if len(counts) == 3 else "three_straight", start, n)
    return None


def beats(new, previous):
    if new is None or new[0] == "pass":
        return False
    if previous is None or previous[0] == "pass":
        return True

    def tier(x):
        if x[0] == "rocket":
            return (4, 0)
        if x[0] == "bomb":
            return (3, x[2]) if x[2] >= 6 else (1, x[2])
        return (2, 0) if x[0] == "flush" else (0, 0)

    a, b = tier(new), tier(previous)
    if a != b:
        return a > b
    return new[0] == previous[0] and new[2] == previous[2] and new[1] > previous[1]


def validate(hand, action, claim, level, previous=None):
    if len(action) != len(claim) or len(set(action)) != len(action):
        return False
    if not set(action).issubset(hand):
        return False
    represented = Counter(c % 54 for c in claim)
    for card in action:
        if not wild(card, level):
            if represented[card % 54] == 0:
                return False
            represented[card % 54] -= 1
    if any(c >= 52 and count for c, count in represented.items()):
        return False
    kind = classify(claim, level)
    if kind is None:
        return False
    if kind[0] == "pass":
        return previous is not None
    return beats(kind, classify(previous, level) if previous else None)
