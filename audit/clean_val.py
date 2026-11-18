"""Audit: leakage-free BC validation on fresh IJCAI walls (games 3000+).

Read-only extraction of 200 fresh games into audit/clean_val_games/ (small,
separate from the training data), then forward-only evaluation of the saved
BC checkpoint. Outputs audit/clean_val.json.
"""

import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
from mahjong.model import Model
from mahjong.replay import replay_game

CATS = [
    ("pass", 0, 1), ("hu", 1, 2), ("play", 2, 36), ("chi", 36, 99),
    ("peng", 99, 133), ("gang", 133, 167), ("angang", 167, 201), ("bugang", 201, 235),
]


def srand_of(game):
    for turn in game["logs"][:3]:
        disp = turn.get("output", {}).get("display", {})
        if "srand" in disp:
            return disp["srand"]
    return None


def main():
    train_srands = set()
    fresh = []
    with gzip.open(ROOT / "data/ijcai2026/final2_all.jsonl.gz", "rt") as f:
        for i, line in enumerate(f):
            if i >= 3400:
                break
            game = json.loads(line)
            s = srand_of(game)
            if i < 3000:
                train_srands.add(s)
            elif i < 3200:
                fresh.append((i, s, game))
    overlap = [i for i, s, _ in fresh if s in train_srands]
    out = {
        "fresh_games": len(fresh),
        "fresh_srands": len({s for _, s, _ in fresh}),
        "overlap_with_train_srands": overlap,
    }
    obs_l, mask_l, act_l = [], [], []
    kept = 0
    for i, s, game in fresh:
        if s in train_srands:
            continue
        try:
            samples = replay_game(game)
        except Exception:
            continue
        kept += 1
        for o, m, a, seat in samples:
            obs_l.append(o)
            mask_l.append(m)
            act_l.append(a)
    obs = np.stack(obs_l)
    mask = np.stack(mask_l)
    act = np.array(act_l, dtype=np.int64)
    out["games_used"] = kept
    out["samples"] = len(act)

    state = torch.load(ROOT / "runs/bc_v1/checkpoint.pt", map_location="cpu")
    model = Model(state["config"]["hidden"])
    model.load_state_dict({k[7:]: v for k, v in state["policy"].items()})
    model.eval()
    torch.set_num_threads(4)
    total = correct = 0
    by_cat = defaultdict(lambda: [0, 0])
    nonpass = [0, 0]
    single = [0, 0]
    with torch.no_grad():
        for s in range(0, len(act), 1024):
            ot = torch.from_numpy(obs[s:s + 1024])
            mt = torch.from_numpy(mask[s:s + 1024])
            at = torch.from_numpy(act[s:s + 1024])
            logits, _ = model(ot)
            pred = logits.masked_fill(~mt, -1e9).argmax(-1)
            ok = pred == at
            total += len(at)
            correct += int(ok.sum())
            legal = mask[s:s + 1024].sum(1)
            single[0] += int((ok & (legal == 1)).sum())
            single[1] += int((legal == 1).sum())
            nonpass[0] += int((ok & (at != 0)).sum())
            nonpass[1] += int((at != 0).sum())
            for name, lo, hi in CATS:
                sel = (at >= lo) & (at < hi)
                by_cat[name][0] += int((ok & sel).sum())
                by_cat[name][1] += int(sel.sum())
    out["bc_on_fresh_walls"] = {
        "accuracy": round(correct / total, 4),
        "nonpass_acc": round(nonpass[0] / max(nonpass[1], 1), 4),
        "nonpass_n": nonpass[1],
        "single_legal_frac": round(single[1] / total, 4),
        "by_cat": {
            k: {"n": v[1], "acc": round(v[0] / v[1], 4) if v[1] else None}
            for k, v in by_cat.items()
        },
    }
    # Also evaluate the RL v1 actor on the same fresh samples for reference.
    rl = torch.load(ROOT / "runs/rl_v1/checkpoint.pt", map_location="cpu")
    model.load_state_dict({k[7:]: v for k, v in rl["policy"].items()})
    model.eval()
    total = correct = nonpass_c = nonpass_n = play_c = play_n = 0
    with torch.no_grad():
        for s in range(0, len(act), 1024):
            ot = torch.from_numpy(obs[s:s + 1024])
            mt = torch.from_numpy(mask[s:s + 1024])
            at = torch.from_numpy(act[s:s + 1024])
            logits, _ = model(ot)
            pred = logits.masked_fill(~mt, -1e9).argmax(-1)
            ok = pred == at
            total += len(at)
            correct += int(ok.sum())
            nonpass_c += int((ok & (at != 0)).sum())
            nonpass_n += int((at != 0).sum())
            sel = (at >= 2) & (at < 36)
            play_c += int((ok & sel).sum())
            play_n += int(sel.sum())
    out["rl_on_fresh_walls"] = {
        "accuracy": round(correct / total, 4),
        "nonpass_acc": round(nonpass_c / max(nonpass_n, 1), 4),
        "play_acc": round(play_c / max(play_n, 1), 4),
        "note": "agreement with IJCAI agents' actions, NOT win rate; RL optimizes reward not imitation",
    }
    (ROOT / "audit/clean_val.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()