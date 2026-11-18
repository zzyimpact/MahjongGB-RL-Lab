"""Four-seat single-hand Guandan judge with deterministic preparation policies."""

import random
from .rules import LEVELS, rank, strength, validate
from ..heuristics import tribute, return_card


class GameCore:
    def reset(self, seed=0, level="2", tribute_count=0, first=0, last=3):
        self.rng = random.Random(seed)
        self.level = level
        deck = list(range(108))
        self.rng.shuffle(deck)
        self.hands = [deck[i * 27 : (i + 1) * 27] for i in range(4)]
        self.initial_hands = [list(x) for x in self.hands]
        self.played = [[] for _ in range(4)]
        self.history, self.finished = [], []
        self.current = 0
        self.last_player, self.last_action, self.last_claim = -1, [], None
        self.done, self.rewards = False, [0.0] * 4
        self.preparation = {
            "tribute": tribute_count,
            "first": first,
            "last": last,
            "tribute_cards": {},
            "return_cards": {},
            "resist": False,
        }
        if tribute_count:
            self._prepare(tribute_count, first, last)
        return self.view(self.current)

    def _prepare(self, count, first, last):
        if count not in (1, 2):
            raise ValueError("tribute_count must be 0, 1 or 2")
        donors = [last] if count == 1 else [last, (last + 2) % 4]
        resist = sum(sum(rank(c) == 14 for c in self.hands[p]) for p in donors) == 2
        self.preparation["resist"] = resist
        if resist:
            self.current = first
            return
        gifts = {p: tribute(self.hands[p], self.level)[0] for p in donors}
        donors.sort(key=lambda p: (-strength(rank(gifts[p]), self.level), p != last))
        recipients = [first] if count == 1 else [first, (first + 2) % 4]
        # First transfer all tribute cards; then recipients choose from their actual hands.
        for donor, receiver in zip(donors, recipients):
            self.hands[donor].remove(gifts[donor])
            self.hands[receiver].append(gifts[donor])
            self.preparation["tribute_cards"][str(donor)] = gifts[donor]
        for donor, receiver in zip(donors, recipients):
            card = return_card(self.hands[receiver], self.level)[0]
            self.hands[receiver].remove(card)
            self.hands[donor].append(card)
            self.preparation["return_cards"][str(receiver)] = card
        self.current = donors[0]

    def view(self, player):
        return {
            "id": player,
            "hand": sorted(self.hands[player]),
            "level": self.level,
            "last_claim": None if self.last_claim is None else list(self.last_claim),
            "last_player": self.last_player,
            "played": [list(x) for x in self.played],
            "counts": [len(h) for h in self.hands],
            "finished": list(self.finished),
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Cannot step a completed hand")
        player = self.current
        cards, claim = action
        if not validate(self.hands[player], cards, claim, self.level, self.last_claim):
            raise ValueError("Illegal Guandan move")
        for card in cards:
            self.hands[player].remove(card)
            self.played[player].append(card)
        self.history.append({"player": player, "response": [list(cards), list(claim)]})
        if cards:
            self.last_player, self.last_action, self.last_claim = (
                player,
                list(cards),
                list(claim),
            )
            if not self.hands[player]:
                self.finished.append(player)
                self.done = len(self.finished) == 3 or (
                    len(self.finished) == 2
                    and self.finished[0] % 2 == self.finished[1] % 2
                )
        if self.done:
            winner = self.finished[0]
            order = self.finished + [p for p in range(4) if p not in self.finished]
            mate_position = order.index((winner + 2) % 4)
            score = 4 - mate_position
            self.rewards = [
                float(score if p % 2 == winner % 2 else -score) for p in range(4)
            ]
            return
        next_player = (player + 1) % 4
        while True:
            if next_player == self.last_player:
                # The winning hand was not beaten. A finished winner lends the lead.
                next_player = (
                    (next_player + 2) % 4
                    if next_player in self.finished
                    else next_player
                )
                self.last_player, self.last_action, self.last_claim = -1, [], None
            if next_player not in self.finished:
                break
            next_player = (next_player + 1) % 4
        self.current = next_player
