"""Convert IJCAI-2026 Botzone replays into BC samples.

Source: HuggingFace Dannibal/ijcai-mahjong-corpora-2026, final2_all.jsonl.gz
(MIT license), raw final-stage games of the IJCAI-2026 MCR campaign. Format:
one JSON game per line; logs alternate request turns (output.content[seat])
and response turns (turn[seat].response answering the previous request).
136-tile no-flower duplicate rules, identical to the course platform.

Each seat is replayed through the deployment Session itself: the recorded
platform response is decomposed into decision labels, every label must map to
a legal action under the reconstructed private view, and the response rebuilt
by Session must equal the recorded one. Games failing any check are dropped.
"""

import argparse
import gzip
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .botzone.session import Session
from .observation import SeatView

OFFSET_TILE = SeatView.OFFSET_TILE
_DUMMY = SeatView(0)


def response_to_action(valid, label):
    """Map one platform decision label to an action index in valid."""
    t = label.split()
    verb = t[0]
    if verb == "PASS":
        action = 0
    elif verb == "HU":
        action = 1
    elif verb == "PLAY":
        action = 2 + OFFSET_TILE[t[1]]
    elif verb == "BUGANG":
        action = 201 + OFFSET_TILE[t[1]]
    elif verb == "GANG" and len(t) == 2:
        action = 167 + OFFSET_TILE[t[1]]
    else:
        action = None
        for a in valid:
            cmd = SeatView.action2response(_DUMMY, a).split()
            if verb == "PENG" and cmd[0] == "Peng":
                action = a
                break
            if verb == "GANG" and len(t) == 1 and cmd[0] == "Gang":
                action = a
                break
            if verb == "CHI" and cmd[0] == "Chi" and cmd[1] == t[1]:
                action = a
                break
        if action is None:
            raise ValueError("no legal action matches label %r" % label)
    if action not in valid:
        raise ValueError("label %r maps to illegal action %d" % (label, action))
    return action


def expand_response(response):
    """CHI X Y and PENG Y carry a follow-up discard in the same response."""
    t = response.split()
    if t[0] == "CHI":
        return ["CHI " + t[1], "PLAY " + t[2]]
    if t[0] == "PENG":
        return ["PENG", "PLAY " + t[1]]
    return [response]


class RecordingSession(Session):
    def __init__(self):
        super().__init__(self._choose)
        self.pending = []
        self.samples = []
        self.decisions = 0

    def _choose(self, arrays):
        if not self.pending:
            raise ValueError("decision without a recorded label")
        label = self.pending.pop(0)
        valid = np.flatnonzero(arrays["action_mask"]).tolist()
        action = response_to_action(valid, label)
        self.samples.append((arrays["obs"], arrays["action_mask"], action))
        return action

    def feed(self, request, response):
        self.pending = expand_response(response) if response is not None else []
        before = len(self.samples)
        rebuilt = super().handle(request)
        decided = len(self.samples) > before
        if response is None:
            if decided:
                raise ValueError("terminal request produced a decision")
            return
        if rebuilt != response:
            raise ValueError(
                "rebuilt %r != recorded %r for request %r"
                % (rebuilt, response, request)
            )
        if decided and self.pending:
            raise ValueError("unconsumed labels %r" % self.pending)
        if not decided and self.pending != ["PASS"]:
            raise ValueError(
                "non-decision request %r with labels %r" % (request, self.pending)
            )


def seat_pairs(logs, seat):
    """Yield (request, response) pairs; the final request has response None."""
    last_request = None
    for turn in logs:
        out = turn.get("output")
        if isinstance(out, dict) and out.get("command") == "request":
            content = out.get("content", {})
            if last_request is not None:
                yield last_request, None
            last_request = content.get(seat)
        elif seat in turn and isinstance(turn[seat], dict):
            entry = turn[seat]
            if entry.get("verdict", "OK") != "OK":
                raise ValueError("verdict %s" % entry.get("verdict"))
            if last_request is None:
                raise ValueError("response without request")
            yield last_request, entry.get("response", "")
            last_request = None
    if last_request is not None:
        yield last_request, None


def replay_game(game):
    if game.get("status") != "finished":
        raise ValueError("unfinished game")
    logs = game["logs"]
    sessions = [RecordingSession() for _ in range(4)]
    for seat in range(4):
        for request, response in seat_pairs(logs, str(seat)):
            if request is None:
                raise ValueError("missing request for seat %d" % seat)
            if request.startswith("1 "):
                flowers = request.split()[1:5]
                if any(int(x) for x in flowers):
                    raise ValueError("flower tiles present")
            sessions[seat].feed(request, response)
    samples = []
    for seat, session in enumerate(sessions):
        for obs, mask, action in session.samples:
            samples.append((obs, mask, action, seat))
    return samples


def _worker(args):
    path, start, count, shard_id = args
    obs_list, mask_list, act_list, gid_list, seat_list = [], [], [], [], []
    kept = dropped = 0
    errors = {}
    with gzip.open(path, "rt") as f:
        for gid, line in enumerate(itertools.islice(f, start, start + count)):
            try:
                samples = replay_game(json.loads(line))
            except Exception as e:
                dropped += 1
                key = str(e)[:80]
                errors[key] = errors.get(key, 0) + 1
                continue
            kept += 1
            for obs, mask, action, seat in samples:
                obs_list.append(obs)
                mask_list.append(mask)
                act_list.append(action)
                gid_list.append(start + gid)
                seat_list.append(seat)
    arrays = {
        "obs": np.stack(obs_list).astype(np.float32) if obs_list else np.zeros((0, 6, 4, 9), np.float32),
        "mask": np.stack(mask_list).astype(np.bool_) if mask_list else np.zeros((0, 235), np.bool_),
        "action": np.array(act_list, dtype=np.int16),
        "game_id": np.array(gid_list, dtype=np.int32),
        "seat": np.array(seat_list, dtype=np.int8),
    }
    return shard_id, arrays, {"kept": kept, "dropped": dropped, "errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="only process the first N games")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-games", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    with gzip.open(args.input, "rt") as f:
        total = sum(1 for _ in f)
    if args.limit:
        total = min(total, args.limit)
    args.output.mkdir(parents=True)
    jobs = []
    start = 0
    shard_id = 0
    while start < total:
        count = min(args.shard_games, total - start)
        jobs.append((str(args.input), start, count, shard_id))
        start += count
        shard_id += 1
    begin = time.monotonic()
    kept = dropped = samples_total = 0
    all_errors = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for shard_id, arrays, stats in pool.map(_worker, jobs):
            np.savez_compressed(args.output / ("shard_%04d.npz" % shard_id), **arrays)
            kept += stats["kept"]
            dropped += stats["dropped"]
            samples_total += len(arrays["action"])
            for key, n in stats["errors"].items():
                all_errors[key] = all_errors.get(key, 0) + n
            print(
                json.dumps(
                    {
                        "shard": shard_id,
                        "kept": stats["kept"],
                        "dropped": stats["dropped"],
                        "samples": int(len(arrays["action"])),
                        "elapsed": round(time.monotonic() - begin, 1),
                    }
                ),
                flush=True,
            )
    meta = {
        "source": "hf-mirror.com/datasets/Dannibal/ijcai-mahjong-corpora-2026 final2_all.jsonl.gz",
        "license": "MIT",
        "games_kept": kept,
        "games_dropped": dropped,
        "drop_reasons": all_errors,
        "samples": samples_total,
        "split_rule": "game_id % 10 == 0 -> val, else train",
        "seconds": time.monotonic() - begin,
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "drop_reasons"}))


if __name__ == "__main__":
    main()