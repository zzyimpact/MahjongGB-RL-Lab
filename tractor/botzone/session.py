"""Incrementally reconstruct only the information delivered to this Bot."""

from ..heuristics import call_card, cover
from ..features import encode


class Session:
    def __init__(self, choose, capacity=128):
        self.choose, self.capacity = choose, capacity
        self.hand, self.played = [], [[] for _ in range(4)]
        self.seat = None

    def _record(self, player, cards):
        for c in cards:
            if c not in self.played[player]:
                self.played[player].append(c)
                if player == self.seat:
                    self.hand.remove(c)

    def handle(self, request, forced=None):
        stage, glob = request["stage"], request["global"]
        bank, level = glob["banking"], glob["level"]
        if stage == "deal":
            delivered = request["deliver"]
            result = (
                call_card(
                    delivered[0], self.hand, bank["called"], bank["snatched"], level
                )
                if forced is None
                else forced
            )
            self.hand.extend(delivered)
            return result
        if stage == "cover":
            self.hand.extend(request["deliver"])
            result = cover(request["deliver"], self.hand) if forced is None else forced
            for c in result:
                self.hand.remove(c)
            return result
        if stage != "play":
            raise ValueError("Unknown Tractor stage: " + stage)
        previous, current, previous_lead, current_lead = request["history"]
        self.seat = (current_lead + len(current)) % 4
        for moves, leader in ((previous, previous_lead), (current, current_lead)):
            for i, cards in enumerate(moves):
                self._record((leader + i) % 4, cards)
        self.view = {
            "id": self.seat,
            "hand": sorted(self.hand),
            "level": level,
            "major": bank["major"],
            "banker": bank["banker"],
            "history": current,
            "played": [list(x) for x in self.played],
            "counts": [25 - len(x) for x in self.played],
        }
        if forced is None:
            arrays, options, _ = encode(self.view, self.capacity)
            result = options[self.choose(arrays)]
        else:
            result = forced
        self._record(self.seat, result)
        return result
