"""Canonical physical-card candidates, constructed from seat-visible state."""

from .gamecore.reference import ReferenceJudge, Error
from .gamecore.move_generator import move_generator


def rules_for(view):
    judge = ReferenceJudge()
    judge.level, judge.major = view["level"], view["major"]
    judge.point_order = list("234567890JQKA")
    judge.Major = ["jo", "Jo"]
    judge._setMajor()
    judge.banker_pos = view.get("banker", 0)
    judge.history = view["history"]
    judge.player_decks = [[] for _ in range(4)]
    judge.player_decks[view["id"]] = view["hand"]
    judge.mv_gen = move_generator(judge.level, judge.major)
    return judge


def candidates(view, capacity=256):
    judge = rules_for(view)
    p = view["id"]
    options = judge._get_action_options(p)
    unique = {}
    for option in options:
        physical = judge._name2id_seq(option, view["hand"])
        try:
            judge._checkLegalMove(physical, p)
        except Error:
            continue
        unique[tuple(sorted(physical))] = sorted(physical)
    ordered = sorted(unique.values(), key=lambda x: (len(x), x))
    total = len(ordered)
    if total > capacity:
        ordered = [ordered[i * (total - 1) // (capacity - 1)] for i in range(capacity)]
    if not ordered:
        raise RuntimeError("No legal Tractor candidate")
    return ordered, total
