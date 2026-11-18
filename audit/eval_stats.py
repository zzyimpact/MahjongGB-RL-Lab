"""Audit: evaluation statistics with confidence intervals and paired tests.

Read-only analysis of runs/eval_v1/*.json per-episode records.
Outputs audit/eval_stats.json.
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def ci95(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return {
        "n": len(x),
        "mean": round(float(m), 4),
        "std": round(float(x.std(ddof=1)), 4),
        "se": round(float(se), 4),
        "ci95_low": round(float(m - 1.96 * se), 4),
        "ci95_high": round(float(m + 1.96 * se), 4),
    }


def main():
    out = {"per_eval": {}, "paired": {}, "checks": {}}
    data = {}
    for p in sorted((ROOT / "runs/eval_v1").glob("*.json")):
        d = json.load(open(p))
        scores = [r["controlled_mean_reward"] for r in d["records"]]
        seats = Counter(tuple(r["controlled_seats"]) for r in d["records"])
        seeds = [r["seed"] for r in d["records"]]
        out["per_eval"][p.stem] = {
            **ci95(scores),
            "wins": d["wins"],
            "draws": d["draws"],
            "losses": d["losses"],
            "seat_distribution": {str(k): v for k, v in seats.items()},
            "seed_range": [min(seeds), max(seeds)],
            "opponent": d["opponent"],
            "policy": d.get("policy", "model"),
            "untrained": d.get("untrained", False),
        }
        data[p.stem] = {r["seed"]: r for r in d["records"]}
        # reward granularity check: multiples of 1/32
        bad = [
            r["seed"]
            for r in d["records"]
            if any(abs(v * 32 - round(v * 32)) > 1e-6 for v in r["rewards"])
        ]
        out["checks"][p.stem + "_reward_granularity_ok"] = not bad
        zero_sum_bad = [
            r["seed"] for r in d["records"] if abs(sum(r["rewards"])) > 1e-6
        ]
        out["checks"][p.stem + "_zero_sum_ok"] = not zero_sum_bad

    for opp in ("random", "teacher"):
        bc = data["bc_vs_" + opp]
        rl = data["rl_vs_" + opp]
        common = sorted(set(bc) & set(rl))
        diff = [rl[s]["controlled_mean_reward"] - bc[s]["controlled_mean_reward"] for s in common]
        seat_match = all(
            bc[s]["controlled_seats"] == rl[s]["controlled_seats"] for s in common
        )
        out["paired"]["rl_minus_bc_vs_" + opp] = {
            **ci95(diff),
            "rl_better_walls": sum(d > 0 for d in diff),
            "bc_better_walls": sum(d < 0 for d in diff),
            "tied_walls": sum(d == 0 for d in diff),
            "same_seats": seat_match,
            "note": "identical seeds => identical walls and seats; opponent actions diverge interactively after the first differing decision (not a frozen-opponent duplicate)",
        }
    # BC vs untrained on same walls
    bc = data["bc_vs_random"]
    un = data["untrained_vs_random"]
    common = sorted(set(bc) & set(un))
    diff = [bc[s]["controlled_mean_reward"] - un[s]["controlled_mean_reward"] for s in common]
    out["paired"]["bc_minus_untrained_vs_random"] = ci95(diff)
    (ROOT / "audit/eval_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()