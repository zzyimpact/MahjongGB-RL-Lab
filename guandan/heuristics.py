"""Basic deterministic tribute/return policies; these are not learned."""

from .gamecore.rules import rank, strength, wild


def tribute(hand, level):
    allowed = [c for c in hand if not wild(c, level)]
    return [max(allowed, key=lambda c: (strength(rank(c), level), -c))]


def return_card(hand, level):
    allowed = [c for c in hand if 1 <= rank(c) <= 9 and not wild(c, level)]
    if not allowed:
        # Defensive fallback for a hand with no eligible low card.
        allowed = [c for c in hand if not wild(c, level)]
    return [min(allowed, key=lambda c: (strength(rank(c), level), c))]
