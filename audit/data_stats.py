"""Audit: BC dataset statistics, validation recomputation, wall-leakage probe.

Read-only. Outputs audit/data_stats.json.
- action category distribution, legal-count stats, single-legal fraction
- BC val metrics recomputed from the saved checkpoint (forward only)
- IJCAI duplicate-wall analysis: srand groups across the game_id%10 split
- exact (obs, action) duplicate hashes across train/val
"""

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
from mahjong.model import Model

CATS = [
    ("pass", 0, 1), ("hu", 1, 2), ("play", 2, 36), ("chi", 36, 99),
    ("peng", 99, 133), ("gang", 133, 167), ("angang", 167, 201), ("bugang", 201, 235),
]


def cat_of(a):
    for name, lo, hi in CATS:
        if lo <= a < hi:
            return name
    return "?"


def load_source(directory):
    obs, mask, act, gid = [], [], [], []
    for p in sorted(Path(directory).glob("shard_*.npz")):
        d = np.load(p)
        obs.append(d["obs"])
        mask.append(d["mask"])
        act.append(d["action"])
        gid.append(d["game_id"])
    return (
        np.concatenate(obs), np.concatenate(mask),
        np.concatenate(act).astype(np.int64), np.concatenate(gid),
    )


def dist_stats(act, mask, gid):
    n = len(act)
    cats = Counter(cat_of(int(a)) for a in act)
    legal = mask.sum(axis=1)
    return {
        "samples": n,
        "games": len(np.unique(gid)),
        "action_counts": dict(cats),
        "action_frac": {k: round(v / n, 4) for k, v in cats.items()},
        "mean_legal_actions": round(float(legal.mean()), 3),
        "single_legal_frac": round(float((legal == 1).mean()), 4),
        "legal_eq_2_frac": round(float((legal == 2).mean()), 4),
    }


def main():
    out = {}
    ij = load_source(ROOT / "data/bc_ijcai_v1")
    te = load_source(ROOT / "data/bc_teacher_v1")
    out["ijcai_v1"] = dist_stats(ij[2], ij[1], ij[3])
    out["teacher_v1"] = dist_stats(te[2], te[1], te[3])

    # ---- BC validation recomputation (forward only, saved checkpoint) ----
    state = torch.load(ROOT / "runs/bc_v1/checkpoint.pt", map_location="cpu")
    model = Model(state["config"]["hidden"])
    model.load_state_dict({k[7:]: v for k, v in state["policy"].items()})
    model.eval()
    torch.set_num_threads(4)

    def val_metrics(obs, mask, act, gid, source_offset):
        g = gid.astype(np.int64) + source_offset
        sel = g % 10 == 0
        o, m, a = obs[sel], mask[sel], act[sel]
        total = correct = 0
        by_cat = defaultdict(lambda: [0, 0])
        single_correct = single_n = 0
        nonpass_correct = nonpass_n = 0
        with torch.no_grad():
            for s in range(0, len(a), 1024):
                ot = torch.from_numpy(o[s:s + 1024])
                mt = torch.from_numpy(m[s:s + 1024])
                at = torch.from_numpy(a[s:s + 1024])
                logits, _ = model(ot)
                pred = logits.masked_fill(~mt, -1e9).argmax(-1)
                ok = pred == at
                total += len(at)
                correct += int(ok.sum())
                legal = m[s:s + 1024].sum(1)
                single_correct += int((ok & (legal == 1)).sum())
                single_n += int((legal == 1).sum())
                nonpass_correct += int((ok & (at != 0)).sum())
                nonpass_n += int((at != 0).sum())
                for name, lo, hi in CATS:
                    selc = (at >= lo) & (at < hi)
                    by_cat[name][0] += int((ok & selc).sum())
                    by_cat[name][1] += int(selc.sum())
        return {
            "val_samples": total,
            "accuracy": round(correct / total, 4),
            "accuracy_by_cat": {
                k: {"n": v[1], "acc": round(v[0] / v[1], 4) if v[1] else None}
                for k, v in by_cat.items()
            },
            "single_legal_acc": round(single_correct / max(single_n, 1), 4),
            "single_legal_n": single_n,
            "nonpass_acc": round(nonpass_correct / max(nonpass_n, 1), 4),
            "nonpass_n": nonpass_n,
        }

    out["bc_val_recomputed"] = {
        "ijcai": val_metrics(ij[0], ij[1], ij[2], ij[3], 0),
        "teacher": val_metrics(te[0], te[1], te[2], te[3], 10_000_000),
    }

    # ---- Duplicate-wall leakage probe (IJCAI source) ----
    # srand per game line from the raw replay INIT display.
    srands = []
    with gzip.open(ROOT / "data/ijcai2026/final2_all.jsonl.gz", "rt") as f:
        for i, line in enumerate(f):
            if i >= 3000:
                break
            d = json.loads(line)
            s = None
            for turn in d["logs"][:3]:
                disp = turn.get("output", {}).get("display", {})
                if "srand" in disp:
                    s = disp["srand"]
                    break
            srands.append(s)
    out["srand_extracted"] = sum(s is not None for s in srands)
    groups = defaultdict(list)
    for gid, s in enumerate(srands):
        if s is not None:
            groups[s].append(gid)
    sizes = Counter(len(v) for v in groups.values())
    out["wall_groups"] = {
        "distinct_srands": len(groups),
        "games_covered": sum(len(v) for v in groups.values()),
        "group_size_distribution": dict(sizes),
    }
    val_gids = set(g for g in range(3000) if g % 10 == 0)
    contaminated = []
    for s, gids in groups.items():
        v = [g for g in gids if g in val_gids]
        t = [g for g in gids if g not in val_gids]
        if v and t:
            contaminated.append({"srand": s, "val_gids": v, "train_gids": t})
    out["cross_split_wall_groups"] = {
        "count": len(contaminated),
        "val_games_with_train_wall": sorted({g for c in contaminated for g in c["val_gids"]}),
        "examples": contaminated[:5],
    }
    n_val_contaminated = len({g for c in contaminated for g in c["val_gids"]})
    out["cross_split_wall_groups"]["val_game_fraction"] = round(
        n_val_contaminated / max(len(val_gids), 1), 4
    )

    # Exact (obs, mask, action) duplicates across the IJCAI train/val split.
    obs, mask, act, gid = ij
    is_val = gid % 10 == 0
    train_hashes = set()
    for i in np.flatnonzero(~is_val):
        h = hashlib.blake2b(
            obs[i].tobytes() + mask[i].tobytes() + act[i].tobytes(), digest_size=16
        ).digest()
        train_hashes.add(h)
    dup = 0
    val_n = 0
    for i in np.flatnonzero(is_val):
        val_n += 1
        h = hashlib.blake2b(
            obs[i].tobytes() + mask[i].tobytes() + act[i].tobytes(), digest_size=16
        ).digest()
        if h in train_hashes:
            dup += 1
    out["exact_obs_action_duplicates"] = {
        "val_samples": val_n,
        "val_samples_with_exact_train_match": dup,
        "fraction": round(dup / max(val_n, 1), 4),
        "note": "blake2b of float32 obs + bool mask + int16 action bytes",
    }
    (ROOT / "audit/data_stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()