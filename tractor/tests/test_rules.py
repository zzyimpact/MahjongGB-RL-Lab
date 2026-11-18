import copy
import random
import unittest
import numpy as np
from tractor.gamecore.engine import GameCore as Tractor
from tractor.botzone.session import Session as TractorSession
from tractor.features import encode as tractor_encode
from tractor.actions import candidates as tractor_candidates, rules_for


class RulesTest(unittest.TestCase):
    def test_private_view_and_seed_repeatability(self):
        game = Tractor()
        game.reset(seed=95)
        seat = game.current
        baseline = game.view(seat)
        other = [p for p in range(4) if p != seat]
        hands = game.player_decks
        hands[other[0]][0], hands[other[1]][0] = hands[other[1]][0], hands[other[0]][0]
        self.assertEqual(game.view(seat), baseline)
        baseline['hand'].clear()
        self.assertTrue(game.view(seat)['hand'])
        game.reset(seed=95)
        again = Tractor()
        again.reset(seed=95)
        self.assertEqual(game.view(game.current), again.view(again.current))

    def test_tractor_pair_follow_order_and_bottom_multiplier(self):
        view = {
            "id": 1,
            "hand": [16, 70, 20, 24],
            "level": "2",
            "major": "h",
            "banker": 0,
            "history": [[8, 62, 12]],
            "played": [[] for _ in range(4)],
            "counts": [25] * 4,
        }
        options, _ = tractor_candidates(view)
        self.assertTrue(options)
        for cards in options:
            self.assertTrue({16, 70}.issubset(cards))
        judge = rules_for(view)
        judge.history = [[0, 54], [48, 102], [40, 94], [44, 98]]
        judge.score = 0
        self.assertEqual(
            judge._checkWinner(3), 0
        )  # AA beats KK regardless of card ID ordering.
        judge.history = [[8], [12], [16], [20]]
        judge.covered_card = [36, 90]  # two tens, 20 bottom points
        judge.score = 0
        judge._reveal(3, 1)
        self.assertEqual(judge.score, 40)

    def test_tractor_protocol_and_complete_games(self):
        for seed in range(30):
            game = Tractor()
            game.reset(seed, level="234567890JQKA"[seed % 13])
            expected = {}

            def choose(arrays):
                for key, value in expected.items():
                    np.testing.assert_array_equal(arrays[key], value)
                return int(np.flatnonzero(arrays["action_mask"])[-1])

            bots = [TractorSession(choose) for _ in range(4)]
            for item in game.preparation:
                glob = {
                    "level": game.level,
                    "banking": {"called": item["called"], "snatched": item["snatched"]},
                }
                response = bots[item["player"]].handle(
                    {"stage": "deal", "deliver": [item["card"]], "global": glob}
                )
                self.assertEqual(response, item["response"])
            glob = {
                "level": game.level,
                "banking": {"major": game.major, "banker": game.banker_pos},
            }
            bots[game.banker_pos].handle(
                {"stage": "cover", "deliver": game.public_card, "global": glob}
            )
            previous, previous_lead = [], game.current
            current_lead = game.current
            steps = 0
            while not game.done:
                p = game.current
                current = game.history if len(game.history) < 4 else []
                # A completed trick is previous history, not an unfinished current trick.
                view = game.view(p)
                view["history"] = current
                expected, _, _ = tractor_encode(view)
                request = {
                    "stage": "play",
                    "global": glob,
                    "history": [previous, current, previous_lead, current_lead],
                }
                action = bots[p].handle(request)
                game.step(action)
                steps += 1
                if len(game.history) == 4:
                    previous = copy.deepcopy(game.history)
                    previous_lead = current_lead
                    current_lead = game.current
                self.assertLessEqual(steps, 100)
            self.assertEqual(
                sorted(
                    c
                    for xs in game.player_decks
                    + game.played_cards
                    + [game.covered_card]
                    for c in xs
                ),
                list(range(108)),
            )


if __name__ == "__main__":
    unittest.main()
