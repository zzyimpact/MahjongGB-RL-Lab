"""Tractor preparation plus the course judge's trick resolution rules."""

import random
from .reference import ReferenceJudge
from .move_generator import move_generator
from ..heuristics import call_card, cover


class GameCore(ReferenceJudge):
    def reset(self, seed=0, level="2", banker=-1):
        rng = random.Random(seed)
        self.level = level
        self.point_order = list("234567890JQKA")
        self.Major = ["jo", "Jo"]
        deck = list(range(108))
        rng.shuffle(deck)
        self.player_decks = [[] for _ in range(4)]
        called = snatched = -1
        self.major = None
        self.preparation = []
        for i, card in enumerate(deck[:100]):
            p = i % 4
            action = call_card(card, self.player_decks[p], called, snatched, level)
            self.player_decks[p].append(card)
            self.preparation.append(
                {
                    "player": p,
                    "card": card,
                    "response": action,
                    "called": called,
                    "snatched": snatched,
                }
            )
            if action:
                if len(action) == 1:
                    called = p
                else:
                    snatched = p
                self.major = "shcd"[card % 4]
        caller = snatched if snatched >= 0 else called
        self.banker_pos = (
            banker if banker >= 0 else (caller if caller >= 0 else rng.randrange(4))
        )
        if self.major is None:
            self.major = rng.choice("shcd")
        self.public_card = deck[100:]
        self.player_decks[self.banker_pos].extend(self.public_card)
        self.covered_card = cover(self.public_card, self.player_decks[self.banker_pos])
        for c in self.covered_card:
            self.player_decks[self.banker_pos].remove(c)
        self._setMajor()
        self.mv_gen = move_generator(level, self.major)
        self.score, self.round = 0, 0
        self.history, self.played_cards = [], [[] for _ in range(4)]
        self.reward, self.done = None, False
        self.current = self.banker_pos
        self.rewards = [0.0] * 4
        return self.view(self.current)

    def view(self, player):
        return {
            "id": player,
            "hand": sorted(self.player_decks[player]),
            "level": self.level,
            "major": self.major,
            "history": [list(x) for x in self.history] if len(self.history) < 4 else [],
            "played": [list(x) for x in self.played_cards],
            "counts": [len(h) for h in self.player_decks],
            "banker": self.banker_pos,
        }

    def step(self, cards):
        if self.done:
            raise RuntimeError("Cannot step a completed hand")
        p = self.current
        if len(set(cards)) != len(cards) or not set(cards).issubset(
            self.player_decks[p]
        ):
            raise ValueError("Cards not in the current hand")
        super().step({"player": p, "action": cards})
        # Training reward is the farmers' score margin around 80, with a team sign.
        # Intermediate trick points remain available as metrics, not duplicated rewards.
        if self.done:
            margin = (self.score - 80) / 80.0
            self.rewards = [
                margin if (i - self.banker_pos) % 2 else -margin for i in range(4)
            ]
