"""Generate BC datasets by recording teacher decisions in real judge games.

Every sample is encoded with mahjong.features.encode on the same SeatView
stream used by training and deployment, so labels cannot leak hidden state.
Actual wall fingerprints are split reproducibly into train/val/test.
"""

import argparse
import hashlib
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
        "wall_id": hashlib.sha256(core.originalTileWall.encode()).hexdigest(),
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
    parser.add_argument("--split-seed", type=int, default=20261119)
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
        "split_rule": "actual wall hash, deterministic 80/10/10 groups",
        "games_with_winner": wins,
        "mean_steps": float(np.mean([r["steps"] for r in all_results])),
        "seconds": time.monotonic() - start,
        "results": all_results,
    }
    from .data_v2 import split_groups
    assignment = split_groups([r['wall_id'] for r in all_results], args.split_seed)
    games = [dict(game_id=r['seed'],wall_id=r['wall_id'],split=assignment[r['wall_id']]) for r in all_results]
    split_for = {g['game_id']:g['split'] for g in games}
    shards = sorted(args.output.glob('shard_*.npz'))
    summary = {}
    for split in ('train','val','test'):
        pieces = {k:[] for k in ('obs','mask','action','game_id')}
        for path in shards:
            with np.load(path) as d:
                sel=np.array([split_for[int(g)]==split for g in d['game_id']])
                for k in pieces: pieces[k].append(d[k][sel])
        arrays={k:np.concatenate(v) for k,v in pieces.items()}
        for k,v in arrays.items():np.save(args.output/(split+'_'+k+'.npy'),v)
        summary[split]=dict(samples=len(arrays['action']),games=len(set(arrays['game_id'].tolist())),
                            walls=sum(s==split for s in assignment.values()))
    manifest=dict(schema=2,complete=True,feature_set='basic',feature_version='v1',source='teacher',
                  split_seed=args.split_seed,groups=assignment,games=games,summary=summary)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
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