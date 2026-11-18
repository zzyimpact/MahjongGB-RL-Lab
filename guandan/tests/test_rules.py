import copy
import random
import unittest
import numpy as np
from guandan.gamecore.engine import GameCore as Guandan
from guandan.botzone.session import Session as GuandanSession
from guandan.features import encode as guandan_encode
from guandan.gamecore.rules import classify, validate, beats


class RulesTest(unittest.TestCase):
    def test_private_view_and_seed_repeatability(self):
        game = Guandan()
        game.reset(seed=95)
        seat = game.current
        baseline = game.view(seat)
        other = [p for p in range(4) if p != seat]
        hands = game.hands
        hands[other[0]][0], hands[other[1]][0] = hands[other[1]][0], hands[other[0]][0]
        self.assertEqual(game.view(seat), baseline)
        baseline['hand'].clear()
        self.assertTrue(game.view(seat)['hand'])
        game.reset(seed=95)
        again = Guandan()
        again.reset(seed=95)
        self.assertEqual(game.view(game.current), again.view(again.current))

    def test_guandan_patterns_and_wildcards(self):
        self.assertEqual(classify([0, 54, 4, 58, 8, 62], "7")[0], "triple_pairs")
        self.assertEqual(classify([0, 1, 2, 4, 5, 6], "7")[0], "three_straight")
        self.assertEqual(classify([0, 36, 40, 44, 48], "7")[0], "flush")
        self.assertIsNone(classify([44, 48, 0, 4, 8], "7"))
        self.assertFalse(validate([4], [4], [52], "2"))
        self.assertTrue(validate([4, 8], [4, 8], [8, 62], "2"))
        self.assertFalse(validate([4, 8], [4, 4], [8, 62], "2"))
        self.assertTrue(beats(("bomb", 1, 6), ("flush", 9, 5)))
        self.assertFalse(beats(("bomb", 12, 5), ("flush", 0, 5)))
        self.assertTrue(beats(("rocket", 0, 4), ("bomb", 14, 10)))

    def test_guandan_protocol_and_complete_games(self):
        for seed in range(30):
            game = Guandan()
            game.reset(seed, level="234567890JQKA"[seed % 13], tribute_count=seed % 3)
            expected = {}

            def choose(arrays):
                for key, value in expected.items():
                    np.testing.assert_array_equal(arrays[key], value)
                return int(np.flatnonzero(arrays["action_mask"])[-1])

            bots = [GuandanSession(choose) for _ in range(4)]
            glob = dict(game.preparation, level=game.level)
            for p, bot in enumerate(bots):
                bot.handle(
                    {
                        "stage": "deal",
                        "deliver": game.initial_hands[p],
                        "your_id": p,
                        "global": glob,
                    }
                )
            # Reconstruct completed automatic exchanges from public metadata.
            for bot in bots:
                bot._exchange(glob)
            steps = 0
            while not game.done:
                p = game.current
                expected, _, _ = guandan_encode(game.view(p))
                request = {
                    "stage": "play",
                    "global": glob,
                    "history": game.history[-4:],
                    "done": game.finished,
                    "pass_on": (
                        game.last_player if game.last_player in game.finished else -1
                    ),
                }
                action = bots[p].handle(request)
                game.step(action)
                steps += 1
                self.assertLess(steps, 1000)
                self.assertEqual(
                    sorted(c for xs in game.hands + game.played for c in xs),
                    list(range(108)),
                )
            self.assertEqual(sum(game.rewards), 0)
            self.assertTrue(all(game.rewards[p] != 0 for p in game.finished))


if __name__ == "__main__":
    unittest.main()
