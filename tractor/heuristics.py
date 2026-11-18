"""Simple, replaceable non-learning policies for deal and cover stages."""


def call_card(card, hand_before, called, snatched, level):
    if card % 54 < 52 and "A234567890JQK"[card % 54 // 4] == level:
        if called == -1:
            return [card]
        mate = (card + 54) % 108
        if snatched == -1 and mate in hand_before:
            return [card, mate]
    return []


def cover(public_cards, hand):
    """Baseline returns the eight received bottom cards unchanged."""
    if len(public_cards) != 8 or not set(public_cards).issubset(hand):
        raise ValueError("Invalid bottom-card request")
    return list(public_cards)
