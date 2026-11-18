"""Fixed-seed evaluation with seat rotation.

Controlled seat policy: model (checkpoint), teacher (heuristic) or random.
Opponents: random legal actions or the heuristic teacher. Same seed list and
seat rotation across runs make models face identical walls and seats.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .env import Environment
from .policy import make_policy
from .teacher import HeuristicTeacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--policy", choices=("model", "teacher", "random"), default="model")
    parser.add_argument("--opponent", choices=("random", "teacher"), default="random")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.output.exists():
        parser.error("output already exists")
    if args.policy == "model" and args.checkpoint is None:
        parser.error("--checkpoint is required for --policy model")
    torch.set_num_threads(1)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu")
        cfg = state["config"]
    else:
        cfg = json.loads((Path(__file__).parent / "configs/default.json").read_text())
    torch.manual_seed(cfg["seed"])
    policy = make_policy(cfg["hidden"])
    if args.policy == "model" and not args.untrained:
        policy.load_state_dict(state["policy"])
    policy.eval()
    env = Environment(cfg["capacity"])
    controller = HeuristicTeacher() if args.policy == "teacher" else None
    opponent = HeuristicTeacher() if args.opponent == "teacher" else None
    records = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        seat = episode % 4
        controlled = [seat] if __package__ == "mahjong" else [seat, (seat + 2) % 4]
        rng = np.random.RandomState(seed)
        step = env.reset({"seed": seed})
        turns = 0
        while not step.episode_done:
            actions = {}
            for player, decision in step.agents.items():
                i = int(player)
                if i in controlled:
                    if args.policy == "model":
                        with torch.no_grad():
                            actions[player] = policy.act(
                                decision.observation, deterministic=True
                            ).action
                    elif args.policy == "teacher":
                        actions[player] = TensorDict(
                            {"action": torch.tensor(controller.choose(env.views[i]))}, []
                        )
                    else:
                        legal = torch.nonzero(
                            decision.observation["action_mask"], as_tuple=False
                        ).flatten().numpy()
                        actions[player] = TensorDict(
                            {"action": torch.tensor(int(rng.choice(legal)))}, []
                        )
                elif opponent is not None:
                    actions[player] = TensorDict(
                        {"action": torch.tensor(opponent.choose(env.views[i]))}, []
                    )
                else:
                    legal = torch.nonzero(
                        decision.observation["action_mask"], as_tuple=False
                    ).flatten().numpy()
                    actions[player] = TensorDict(
                        {"action": torch.tensor(int(rng.choice(legal)))}, []
                    )
            step = env.step(actions)
            turns += 1
            if turns > 5000:
                raise RuntimeError("Evaluation did not terminate")
        rewards = step.shared["rewards"].tolist()
        records.append(
            {
                "seed": seed,
                "controlled_seats": controlled,
                "turns": turns,
                "rewards": rewards,
                "controlled_mean_reward": float(
                    np.mean([rewards[p] for p in controlled])
                ),
            }
        )
    scores = [x["controlled_mean_reward"] for x in records]
    result = {
        "game": __package__,
        "policy": args.policy,
        "untrained": args.untrained,
        "opponent": args.opponent,
        "episodes": args.episodes,
        "mean_reward": float(np.mean(scores)),
        "reward_std": float(np.std(scores)),
        "wins": sum(x > 0 for x in scores),
        "draws": sum(x == 0 for x in scores),
        "losses": sum(x < 0 for x in scores),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()