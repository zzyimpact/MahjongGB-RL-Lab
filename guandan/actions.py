"""Deterministic candidate construction shared by training and submitted Bots.

One canonical physical realization per rank pattern is offered, plus natural
flushes and wildcard realizations. This is a policy candidate set, not an
enumeration of every strategically distinct physical-card subset.
"""

from itertools import product
from .gamecore.rules import LEVELS, rank, wild, validate


def candidates(view, capacity=256):
    hand = sorted(view["hand"])
    level = view["level"]
    previous = view.get("last_claim")
    result = {}

    def offer(action, claim):
        if validate(hand, action, claim, level, previous):
            key = (tuple(sorted(action)), tuple(sorted(c % 54 for c in claim)))
            result[key] = [list(action), list(claim)]

    def realize(pattern, suit=None):
        remaining = hand[:]
        action, claim = [], []
        for r in pattern:
            matches = [
                c for c in remaining if rank(c) == r and (suit is None or c % 4 == suit)
            ]
            matches.sort(key=lambda c: (wild(c, level), c))
            if matches:
                card = matches[0]
                represented = card
            else:
                jokers = [c for c in remaining if wild(c, level)]
                if not jokers or r >= 13:
                    return
                card = jokers[0]
                represented = r * 4 + (suit if suit is not None else 1)
            remaining.remove(card)
            action.append(card)
            claim.append(represented)
        offer(action, claim)

    offer([], [])
    for card in hand:
        offer([card], [card])
    for r in range(15):
        for n in range(2, 11 if r < 13 else 3):
            realize([r] * n)
    for r, pair in product(range(13), range(15)):
        if r != pair:
            realize([r] * 3 + [pair] * 2)
    for length, copies in ((5, 1), (3, 2), (2, 3)):
        for start in range(15 - length):
            pattern = [
                r % 13 for r in range(start, start + length) for _ in range(copies)
            ]
            realize(pattern)
            if length == 5:
                for suit in range(4):
                    realize(pattern, suit)
    realize([13, 13, 14, 14])
    ordered = sorted(result.values(), key=lambda x: (len(x[0]), x[1], x[0]))
    total = len(ordered)
    if total > capacity:
        # Evenly retain small and large hands; stable across processes and Python versions.
        ordered = [ordered[i * (total - 1) // (capacity - 1)] for i in range(capacity)]
    if not ordered:
        raise RuntimeError("No legal candidate for an active player")
    return ordered, total
