"""Behavior cloning on recorded decision samples (teacher or human replays).

Trains the actor head with masked cross-entropy; the critic stays at its
initialization for PPO to fit. Saves a checkpoint in mahjong.train format so
RL can resume directly from BC weights:

    python -m mahjong.train --resume runs/bc/checkpoint.pt --output runs/rl ...

Split unit is the whole game: unique game id % 10 == 0 goes to validation.
"""

import argparse
import bisect
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .policy import make_policy


class ShardDataset(Dataset):
    def __init__(self, sources, split):
        self.shards = []
        self.cum = [0]
        self.indices = []
        for source_idx, directory in enumerate(sources):
            for path in sorted(Path(directory).glob("shard_*.npz")):
                data = np.load(path)
                gid = data["game_id"].astype(np.int64) + source_idx * 10_000_000
                rows = np.flatnonzero((gid % 10 == 0) if split == "val" else (gid % 10 != 0))
                if len(rows) == 0:
                    continue
                shard_id = len(self.shards)
                self.shards.append(
                    {
                        "obs": data["obs"],
                        "mask": data["mask"],
                        "action": data["action"].astype(np.int64),
                    }
                )
                self.indices.extend((shard_id, int(r)) for r in rows)
                self.cum.append(len(self.indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        shard_id, row = self.indices[i]
        shard = self.shards[shard_id]
        return (
            torch.from_numpy(shard["obs"][row]),
            torch.from_numpy(shard["mask"][row]),
            torch.tensor(shard["action"][row]),
        )


def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    by_verb = {}
    with torch.no_grad():
        for obs, mask, action in loader:
            obs, mask, action = obs.to(device), mask.to(device), action.to(device)
            logits, _ = model(obs)
            logits = logits.masked_fill(~mask, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, action, reduction="sum")
            pred = logits.argmax(dim=-1)
            total += len(action)
            correct += int((pred == action).sum())
            loss_sum += float(loss)
            for category, lo, hi in (("pass", 0, 1), ("hu", 1, 2), ("play", 2, 36), ("meld", 36, 235)):
                sel = (action >= lo) & (action < hi)
                n = int(sel.sum())
                if n:
                    hit = int(((pred == action) & sel).sum())
                    agg = by_verb.setdefault(category, [0, 0])
                    agg[0] += hit
                    agg[1] += n
    model.train()
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "samples": total,
        "accuracy_by_verb": {k: v[0] / v[1] for k, v in by_verb.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/default.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg = json.loads(args.config.read_text())
    train_set = ShardDataset(args.data, "train")
    val_set = ShardDataset(args.data, "val")
    if len(train_set) == 0 or len(val_set) == 0:
        parser.error("empty train or val split")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512)
    device = torch.device(args.device)
    policy = make_policy(cfg["hidden"]).to(device)
    model = policy.module
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.lr)
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    start = time.monotonic()
    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = correct = 0
        loss_sum = 0.0
        updates = 0
        for obs, mask, action in train_loader:
            obs, mask, action = obs.to(device), mask.to(device), action.to(device)
            logits, _ = model(obs)
            logits = logits.masked_fill(~mask, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
            seen += len(action)
            loss_sum += float(loss) * len(action)
            correct += int((logits.argmax(dim=-1) == action).sum())
        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "gradient_updates": updates,
            "val": val,
            "elapsed_seconds": time.monotonic() - start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["loss"] < best_val:
            best_val = val["loss"]
            rl_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["ppo"]["learning_rate"])
            state = {
                "policy": policy.state_dict(),
                "optimizer": rl_optimizer.state_dict(),
                "config": cfg,
                "iteration": 0,
                "counters": {
                    "env_steps": 0,
                    "agent_steps": 0,
                    "gradient_updates": 0,
                    "completed_episodes": 0,
                },
                "episode_cursor": 0,
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "bc": {
                    "sources": [str(p) for p in args.data],
                    "train_samples": len(train_set),
                    "val_samples": len(val_set),
                    "epochs": epoch,
                    "val_loss": val["loss"],
                    "val_accuracy": val["accuracy"],
                },
            }
            temporary = args.output / "checkpoint.tmp"
            torch.save(state, temporary)
            temporary.replace(args.output / "checkpoint.pt")
    (args.output / "bc_history.json").write_text(json.dumps(history, indent=2))
    print(
        json.dumps(
            {
                "train_samples": len(train_set),
                "val_samples": len(val_set),
                "best_val_loss": best_val,
                "seconds": time.monotonic() - start,
            }
        )
    )


if __name__ == "__main__":
    main()