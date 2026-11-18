"""Audit: artifact inventory, checkpoint loadability, weight identity checks.

Read-only. Outputs audit/inventory.json.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe_checkpoint(path):
    state = torch.load(path, map_location="cpu")
    info = {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "keys": sorted(state.keys()),
        "loadable": True,
    }
    if "policy" in state:
        pol = state["policy"]
        info["policy_tensors"] = len(pol)
        info["policy_dtypes"] = sorted({str(v.dtype) for v in pol.values()})
        info["policy_numel"] = int(sum(v.numel() for v in pol.values()))
        info["policy_keys"] = sorted(pol.keys())
    if "config" in state:
        info["config"] = state["config"]
    if "counters" in state:
        info["counters"] = state["counters"]
    if "iteration" in state:
        info["iteration"] = state["iteration"]
    if "bc" in state:
        info["bc_meta"] = state["bc"]
    return info, state


def main():
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }
    import tensordict
    import MahjongGB

    out["tensordict"] = tensordict.__version__
    out["PyMahjongGB"] = getattr(MahjongGB, "__version__", "unknown (no __version__)")

    checkpoints = {}
    states = {}
    for path in sorted(ROOT.glob("runs/*/checkpoint.pt")):
        info, state = describe_checkpoint(path)
        checkpoints[info["path"]] = info
        states[info["path"]] = state
    out["checkpoints"] = checkpoints

    # Weight identity: exported deployment weights vs BC vs RL.
    export = ROOT / "submissions/mahjong_v1/mahjong_model.pt"
    exported = torch.load(export, map_location="cpu")
    bc = states["runs/bc_v1/checkpoint.pt"]["policy"]
    rl = states["runs/rl_v1/checkpoint.pt"]["policy"]
    strip = lambda d: {k[len("module."):]: v for k, v in d.items() if k.startswith("module.")}
    bc_m, rl_m = strip(bc), strip(rl)
    out["export_identity"] = {
        "export_sha256": sha256(export),
        "export_keys": sorted(exported.keys()),
        "export_numel": int(sum(v.numel() for v in exported.values())),
        "matches_rl_v1_exactly": all(
            torch.equal(exported[k], rl_m[k]) for k in exported
        ) and set(exported) == set(rl_m),
        "matches_bc_v1_exactly": all(
            torch.equal(exported[k], bc_m[k]) for k in exported
        ) and set(exported) == set(bc_m),
        "rl_vs_bc_actor_changed": not torch.equal(rl_m["actor.weight"], bc_m["actor.weight"]),
        "rl_vs_bc_critic_changed": not torch.equal(rl_m["critic.weight"], bc_m["critic.weight"]),
        "rl_vs_bc_conv_changed": not torch.equal(rl_m["conv.0.weight"], bc_m["conv.0.weight"]),
    }
    # Critic at BC time vs fresh inits (BC trainer used --seed 0; RL config seed 2026).
    from mahjong.model import Model

    inits = {}
    for seed in (0, 2026):
        torch.manual_seed(seed)
        inits[seed] = Model(64).state_dict()
    out["bc_critic_vs_fresh_init"] = {
        "critic_weight_equal_to_seed0_init": torch.equal(
            bc_m["critic.weight"], inits[0]["critic.weight"]
        ),
        "actor_weight_equal_to_seed0_init": torch.equal(
            bc_m["actor.weight"], inits[0]["actor.weight"]
        ),
        "critic_weight_equal_to_seed2026_init": torch.equal(
            bc_m["critic.weight"], inits[2026]["critic.weight"]
        ),
    }
    # BC checkpoint optimizer state: empty (fresh) or populated?
    bc_opt = states["runs/bc_v1/checkpoint.pt"]["optimizer"]
    out["bc_checkpoint_optimizer"] = {
        "param_groups": len(bc_opt["param_groups"]),
        "lr": bc_opt["param_groups"][0]["lr"],
        "state_entries": len(bc_opt["state"]),
    }
    rl_state = states["runs/rl_v1/checkpoint.pt"]
    out["rl_checkpoint_optimizer"] = {
        "param_groups": len(rl_state["optimizer"]["param_groups"]),
        "lr": rl_state["optimizer"]["param_groups"][0]["lr"],
        "state_entries": len(rl_state["optimizer"]["state"]),
    }
    # Evaluation files
    out["evaluations"] = {
        p.name: {
            "episodes": json.load(open(p))["episodes"],
            "size_bytes": p.stat().st_size,
        }
        for p in sorted((ROOT / "runs/eval_v1").glob("*.json"))
    }
    # Datasets
    datasets = {}
    for meta_path in sorted(ROOT.glob("data/*/meta.json")):
        meta = json.load(open(meta_path))
        shards = list(meta_path.parent.glob("shard_*.npz"))
        n = 0
        for s in shards:
            with np.load(s) as d:
                n += len(d["action"])
        datasets[str(meta_path.parent.relative_to(ROOT))] = {
            "shards": len(shards),
            "samples_recounted": n,
            "meta_claims": {
                k: meta.get(k)
                for k in ("games", "games_kept", "games_dropped", "samples", "source", "license", "split_rule")
                if k in meta
            },
        }
    out["datasets"] = datasets
    (ROOT / "audit/inventory.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("python", "torch", "tensordict", "export_identity", "bc_critic_vs_fresh_init", "bc_checkpoint_optimizer")}, indent=2))
    print("checkpoints:", list(checkpoints))
    print("datasets:", json.dumps(datasets, indent=1))


if __name__ == "__main__":
    main()