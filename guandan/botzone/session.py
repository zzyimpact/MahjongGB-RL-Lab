"""Seat-private Guandan history and tribute reconstruction."""

from ..features import encode
from ..heuristics import tribute, return_card
from ..gamecore.rules import rank, strength


class Session:
    def __init__(self, choose, capacity=128):
        self.choose, self.capacity = choose, capacity
        self.hand, self.played = [], [[] for _ in range(4)]
        self.seat = None
        self.last_player, self.last_claim = -1, None
        self.transferred = set()

    def _transfer(self, source, target, card, key):
        if key in self.transferred:
            return
        if source == self.seat:
            self.hand.remove(card)
        if target == self.seat:
            self.hand.append(card)
        self.transferred.add(key)

    def _exchange(self, glob):
        if not glob["tribute"] or glob.get("resist"):
            return
        gifts = {int(p): c for p, c in glob.get("tribute_cards", {}).items() if c >= 0}
        if len(gifts) != glob["tribute"]:
            return
        donors = sorted(
            gifts,
            key=lambda p: (-strength(rank(gifts[p]), glob["level"]), p != glob["last"]),
        )
        receivers = (
            [glob["first"]]
            if len(donors) == 1
            else [glob["first"], (glob["first"] + 2) % 4]
        )
        for donor, receiver in zip(donors, receivers):
            self._transfer(donor, receiver, gifts[donor], ("tribute", donor))
        returns = glob.get("return_cards", {})
        for donor, receiver in zip(donors, receivers):
            card = returns.get(str(receiver), returns.get(receiver, -1))
            if card >= 0:
                self._transfer(receiver, donor, card, ("return", receiver))

    def _record(self, player, action, claim):
        fresh = any(c not in self.played[player] for c in action)
        for c in action:
            if c not in self.played[player]:
                self.played[player].append(c)
                if player == self.seat:
                    self.hand.remove(c)
        if fresh:
            self.last_player, self.last_claim = player, list(claim)

    def handle(self, request, forced=None):
        stage, glob = request["stage"], request["global"]
        self.level = glob["level"]
        if stage == "deal":
            self.seat, self.hand = request["your_id"], list(request["deliver"])
            return []
        self._exchange(glob)
        if stage == "tribute":
            return (
                ([] if glob.get("resist") else tribute(self.hand, self.level))
                if forced is None
                else forced
            )
        if stage == "return":
            return (
                ([] if glob.get("resist") else return_card(self.hand, self.level))
                if forced is None
                else forced
            )
        if stage != "play":
            raise ValueError("Unknown Guandan stage: " + stage)
        for record in request["history"]:
            self._record(record["player"], *record["response"])
        finished = request.get("done", [])
        suffix_passes = 0
        for record in reversed(request["history"]):
            if record["response"][0]:
                break
            suffix_passes += 1
        borrowed = (
            self.last_player in finished
            and (self.last_player + 2) % 4 == self.seat
            and suffix_passes >= 4 - len(finished)
        )
        if self.last_player == self.seat or borrowed:
            self.last_player, self.last_claim = -1, None
        self.view = {
            "id": self.seat,
            "hand": sorted(self.hand),
            "level": self.level,
            "last_player": self.last_player,
            "last_claim": self.last_claim,
            "played": [list(x) for x in self.played],
            "counts": [27 - len(x) for x in self.played],
            "finished": finished,
        }
        if forced is None:
            arrays, options, _ = encode(self.view, self.capacity)
            result = options[self.choose(arrays)]
        else:
            result = forced
        self._record(self.seat, *result)
        return result
