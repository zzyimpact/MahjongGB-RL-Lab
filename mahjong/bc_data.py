"""Generate BC datasets by recording teacher decisions in real judge games.

Every sample is encoded with mahjong.features.encode on the same SeatView
stream used by training and deployment, so labels cannot leak hidden state.
Games are the split unit: sample game_id determines train/val membership.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .features import encode
from .gamecore.engine import GameCore
from .observation import SeatView
from .teacher import HeuristicTeacher


def play_game(seed, teacher=None):
    teacher = teacher or HeuristicTeacher()
    core = GameCore()
    core.r.seed(seed)
    core.reset()
    views = [SeatView(i) for i in range(4)]
    offsets = [0] * 4
    samples = []
    commands_log = []
    steps = 0
    while not core.done:
        for i in range(4):
            events = core.agents[i].events
            for event in events[offsets[i]:]:
                views[i].request2obs(event)
            offsets[i] = len(events)
        commands = {}
        for i in core.obs:
            arrays = encode(views[i])
            action = teacher.choose(views[i])
            if action not in views[i].valid:
                raise RuntimeError(
                    "teacher chose an illegal action: seed=%d seat=%d action=%d"
                    % (seed, i, action)
                )
            samples.append(
                (arrays["obs"], arrays["action_mask"], action, i)
            )
            commands[core.agent_names[i]] = views[i].action2response(action)
        commands_log.append(commands)
        core.step(commands)
        steps += 1
        if steps > 2000:
            raise RuntimeError("teacher game did not terminate: seed=%d" % seed)
        if core.reward and min(core.reward) == -30 and max(core.reward) == 10:
            raise RuntimeError(
                "judge rejected a teacher action: seed=%d invalid=%s"
                % (seed, core.invalid_info)
            )
    result = {
        "seed": seed,
        "rewards": list(core.reward or [0, 0, 0, 0]),
        "steps": steps,
        "samples": len(samples),
    }
    return samples, result


def _worker(args):
    start, count, shard_id = args
    teacher = HeuristicTeacher()
    obs_list, mask_list, act_list, gid_list, seat_list = [], [], [], [], []
    results = []
    for game in range(start, start + count):
        samples, result = play_game(game, teacher)
        for obs, mask, action, seat in samples:
            obs_list.append(obs)
            mask_list.append(mask)
            act_list.append(action)
            gid_list.append(game)
            seat_list.append(seat)
        results.append(result)
    return shard_id, {
        "obs": np.stack(obs_list).astype(np.float32),
        "mask": np.stack(mask_list).astype(np.bool_),
        "action": np.array(act_list, dtype=np.int16),
        "game_id": np.array(gid_list, dtype=np.int32),
        "seat": np.array(seat_list, dtype=np.int8),
    }, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-games", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0, help="first game seed")
    args = parser.parse_args()
    if args.games < 1 or args.workers < 1 or args.shard_games < 1:
        parser.error("games, workers and shard-games must be positive")
    if args.output.exists():
        parser.error("output already exists")
    args.output.mkdir(parents=True)
    jobs = []
    shard_id = 0
    done = 0
    while done < args.games:
        count = min(args.shard_games, args.games - done)
        jobs.append((args.seed + done, count, shard_id))
        done += count
        shard_id += 1
    start = time.monotonic()
    all_results = []
    total_samples = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for shard_id, arrays, results in pool.map(_worker, jobs):
            path = args.output / ("shard_%04d.npz" % shard_id)
            np.savez_compressed(path, **arrays)
            all_results.extend(results)
            total_samples += len(arrays["action"])
            print(
                json.dumps(
                    {
                        "shard": shard_id,
                        "games": len(results),
                        "samples": int(len(arrays["action"])),
                        "elapsed": round(time.monotonic() - start, 1),
                    }
                ),
                flush=True,
            )
    wins = sum(1 for r in all_results if max(r["rewards"]) > 0)
    meta = {
        "source": "heuristic teacher self-play (mahjong.teacher.HeuristicTeacher)",
        "games": len(all_results),
        "samples": total_samples,
        "seed_start": args.seed,
        "shard_games": args.shard_games,
        "split_rule": "game_id % 10 == 0 -> val, else train",
        "games_with_winner": wins,
        "mean_steps": float(np.mean([r["steps"] for r in all_results])),
        "seconds": time.monotonic() - start,
        "results": all_results,
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(
            {
                "games": meta["games"],
                "samples": total_samples,
                "games_with_winner": wins,
                "seconds": round(meta["seconds"], 1),
            }
        )
    )


if __name__ == "__main__":
    main()