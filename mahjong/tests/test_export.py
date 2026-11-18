"""Local Botzone deployment checks without uploading anything.

Exports the ZIP from a checkpoint, runs the exported __main__.py as a real
subprocess over stdin/stdout, plays full judge games through platform
messages, verifies cold-history replay reproduces the same responses, and
measures per-request latency against the 1-second platform limit.

Set BOT_TEST_CHECKPOINT to choose the checkpoint (default runs/bc_v1).
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

import numpy as np

from mahjong.botzone.session import Session
from mahjong.gamecore.engine import GameCore
from mahjong.observation import SeatView
from mahjong.tests.test_protocol import platform_messages

CHECKPOINT = Path(os.environ.get("BOT_TEST_CHECKPOINT", "runs/bc_v1/checkpoint.pt"))
BOT_FILES = (
    "__main__.py",
    "mahjong/model.py",
    "mahjong/features.py",
    "mahjong/observation.py",
    "mahjong/rules.py",
    "mahjong/botzone/runner.py",
    "mahjong/botzone/session.py",
    "mahjong/botzone/inference.py",
    "mahjong/gamecore/engine.py",
)


class BotProcess:
    def __init__(self, cwd, env):
        self.proc = subprocess.Popen(
            [sys.executable, "__main__.py"],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.latencies = []
        self.first = True

    def send(self, lines):
        # Long-running mode: the very first line is the turn count.
        if isinstance(lines, str):
            lines = [lines]
        if self.first:
            lines = ["1"] + lines
            self.first = False
        return self.send_raw(lines)

    def send_raw(self, lines):
        start = time.monotonic()
        self.proc.stdin.write("\n".join(lines) + "\n")
        self.proc.stdin.flush()
        response = self.proc.stdout.readline().strip()
        for _ in range(3):
            self.proc.stdout.readline()
        marker = self.proc.stdout.readline().strip()
        self.latencies.append(time.monotonic() - start)
        if marker != ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<":
            raise AssertionError("missing keep-running marker: %r" % marker)
        return response

    def close(self):
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.proc.wait(timeout=10)


def play_game(seed, seat0_handler, rng):
    """Judge game; seat 0 uses seat0_handler(request)->response, others random."""
    core = GameCore()
    core.r.seed(seed)
    core.reset()
    bots = [Session(lambda arrays: int(rng.choice(np.flatnonzero(arrays["action_mask"])))) for _ in range(4)]
    observers = [SeatView(i) for i in range(4)]
    offsets = [0] * 4
    discards = {}
    history0 = []
    steps = 0
    while not core.done:
        if core.state == 0:
            core.step({core.agent_names[core.curPlayer]: "Play " + discards[core.curPlayer]})
        responses = {}
        for p in range(4):
            events = core.agents[p].events[offsets[p]:]
            offsets[p] += len(events)
            for event in events:
                observers[p].request2obs(event)
            for message in platform_messages(events, p):
                if p == 0:
                    responses[p] = seat0_handler(message)
                    history0.append((message, responses[p]))
                else:
                    responses[p] = bots[p].handle(message)
        commands = {}
        for p in core.obs:
            response = responses[p].split()
            if response[0] in ("CHI", "PENG"):
                discards[p] = response[-1]
                response = response[:-1]
            command = response[0].title()
            if response[0] == "BUGANG":
                command = "BuGang"
            commands[core.agent_names[p]] = " ".join([command] + response[1:])
        core.step(commands)
        steps += 1
        if core.reward:
            if min(core.reward) == -30 and max(core.reward) == 10:
                raise AssertionError("judge rejected bot action: %s" % core.invalid_info)
        if steps > 500:
            raise AssertionError("game did not terminate")
    return history0, list(core.reward or [0, 0, 0, 0])


class ExportDeploymentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT.exists():
            raise unittest.SkipTest("checkpoint %s not found" % CHECKPOINT)
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.export_dir = root / "export"
        subprocess.run(
            [sys.executable, "-m", "mahjong.export_bot",
             "--checkpoint", str(CHECKPOINT), "--output", str(cls.export_dir)],
            check=True,
        )
        cls.bot_dir = root / "bot"
        with zipfile.ZipFile(cls.export_dir / "mahjong_bot.zip") as z:
            z.extractall(cls.bot_dir)
        cls.env = dict(os.environ, BOT_MODEL_PATH=str(cls.export_dir / "mahjong_model.pt"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_python36_syntax(self):
        for name in BOT_FILES:
            source = (self.bot_dir / name).read_text()
            ast.parse(source, filename=name, feature_version=(3, 6))

    def test_protocol_games_and_latency(self):
        rng = np.random.RandomState(7)
        bot = BotProcess(self.bot_dir, self.env)
        try:
            for seed in (11, 22, 33):
                play_game(seed, bot.send, rng)
        finally:
            bot.close()
        self.assertGreater(len(bot.latencies), 50)
        self.assertLess(max(bot.latencies), 1.0, "platform limit is 1s per request")
        steady = bot.latencies[1:]
        print(
            json.dumps(
                {
                    "requests": len(bot.latencies),
                    "first_request_ms": round(1000 * bot.latencies[0], 2),
                    "steady_mean_latency_ms": round(1000 * sum(steady) / len(steady), 2),
                    "steady_max_latency_ms": round(1000 * max(steady), 2),
                }
            )
        )

    def test_cold_history_replay(self):
        rng = np.random.RandomState(7)
        history, _ = play_game(
            11, lambda req: _model_response(req, self.bot_dir, self.env), rng
        )
        # Restart a fresh process: replay all but the last turn, then ask the
        # last request again; the deterministic bot must give the same answer.
        bot = BotProcess(self.bot_dir, self.env)
        try:
            lines = [str(len(history))]
            for request, response in history[:-1]:
                lines.extend([request, response])
            rebuilt = bot.send_raw(lines + [history[-1][0]])
        finally:
            bot.close()
        self.assertEqual(rebuilt, history[-1][1])
        print(
            json.dumps(
                {
                    "cold_replay_turns": len(history),
                    "cold_replay_latency_ms": round(1000 * bot.latencies[0], 2),
                }
            )
        )


def _model_response(request, bot_dir, env):
    """In-process model response used to record the reference history.

    Loads config/weights from the export directory because the source tree
    has no bot_config.json (it is generated at export time).
    """
    global _CHOOSE, _SESSION
    try:
        session = _SESSION
    except NameError:
        import torch
        from mahjong.model import Model, make_model

        config = json.loads((bot_dir / "mahjong" / "bot_config.json").read_text())
        model = make_model(config)
        model.load_state_dict(torch.load(env["BOT_MODEL_PATH"], map_location="cpu"))
        model.eval()

        def choose(arrays):
            with torch.no_grad():
                obs = torch.from_numpy(arrays["obs"]).float().unsqueeze(0)
                logits, _ = model(obs)
                mask = torch.from_numpy(arrays["action_mask"]).bool().unsqueeze(0)
                return int(logits.masked_fill(~mask, -1e9).argmax(dim=-1).item())

        _CHOOSE = choose
        session = Session(choose, feature_set=config.get("feature_set", "basic"))
        _SESSION = session
    return session.handle(request)


if __name__ == "__main__":
    unittest.main()