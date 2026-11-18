"""Synchronous whole-episode PPO; one learner process and N CPU actors."""

import argparse
import json
import os
import random
import time
import shutil
import hashlib
import subprocess
import sys
from functools import partial
from pathlib import Path
import numpy as np
import torch
from rlframework.collectors.parallel import ParallelEpisodeCollector
from rlframework.collectors.routing import StaticPolicyRouter
from rlframework.collector import PolicyRef
from rlframework.algorithm.ppo import PPOLearner, PPOConfig
from rlframework.processing.gae import GAEProcessor
from rlframework.workflow.on_policy import MultiPolicyOnPolicyWorkflow, OnPolicyStack
from .policy import make_policy, make_collector
from .model import model_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/default.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--actors", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--bc-reference", type=Path)
    parser.add_argument("--bc-kl", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-walls", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=8100000)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    if args.bc_kl < 0 or args.eval_every < 0:
        parser.error("KL and eval period must be nonnegative")
    if args.bc_kl and not args.bc_reference:
        parser.error("KL requires a frozen BC reference")
    if args.eval_every and (not args.resume or not args.split_manifest):
        parser.error("periodic evaluation requires resume and split manifest")
    cfg = json.loads(args.config.read_text())
    for key in ("iterations", "actors"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if any(cfg[k] < 1 for k in ("iterations", "actors", "episodes_per_actor")):
        parser.error("iterations, actors and episodes_per_actor must be positive")
    if cfg["capacity"] < 2 or (args.seconds is not None and args.seconds <= 0):
        parser.error("capacity must be >=2 and seconds must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    policy = make_policy(cfg["hidden"], config=cfg)
    ppo = PPOConfig(**cfg["ppo"])
    learner = PPOLearner(policy, ppo, device=cfg["device"])
    start_iteration = 0
    counters = {
        "env_steps": 0,
        "agent_steps": 0,
        "gradient_updates": 0,
        "completed_episodes": 0,
    }
    episode_cursor = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        if (
            model_spec(state["config"]) != model_spec(cfg)
            or state["config"]["capacity"] != cfg["capacity"]
        ):
            raise ValueError("Checkpoint feature/model configuration mismatch")
        policy.load_state_dict(state["policy"])
        learner.optimizer.load_state_dict(state["optimizer"])
        start_iteration = state["iteration"]
        counters = state["counters"]
        episode_cursor = state["episode_cursor"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
    if args.bc_reference:
        from .ppo_anchor import BCReferenceLoss
        learner.auxiliary_loss = BCReferenceLoss(policy, args.bc_reference, cfg, cfg['device'], args.bc_kl)
    provenance = dict(resume=str(args.resume), bc_reference=str(args.bc_reference), bc_kl=args.bc_kl,
                      eval_every=args.eval_every, eval_walls=args.eval_walls, eval_seed=args.eval_seed,
                      resume_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None)
    (args.output/'experiment.json').write_text(json.dumps(provenance,indent=2))
    best_eval = float('-inf')
    best_ppo_eval = float('-inf')
    eval_history = []

    def evaluate_checkpoint(path, iteration):
        directory=args.output/('eval_%04d'%iteration)
        subprocess.run([sys.executable,'-m','mahjong.arena','--candidate','candidate='+str(path),
                        '--opponent','teacher','--walls',str(args.eval_walls),'--seed',str(args.eval_seed),
                        '--workers',str(cfg['actors']),'--manifest',str(args.split_manifest),
                        '--output',str(directory),'--purpose','development'],check=True,stdout=subprocess.DEVNULL)
        result=json.loads((directory/'results.json').read_text())
        reward=result['summary']['candidate vs teacher']['reward']
        eval_history.append(dict(iteration=iteration,reward=reward,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (args.output/'eval_history.json').write_text(json.dumps(eval_history,indent=2))
        return reward['mean']

    if args.eval_every:
        best_eval=evaluate_checkpoint(args.resume,0)
        shutil.copy2(args.resume,args.output/'eval_best.pt')
    router = StaticPolicyRouter({str(i): PolicyRef("shared", policy) for i in range(4)})
    factory = partial(
        make_collector,
        hidden=cfg["hidden"],
        config=cfg,
        capacity=cfg["capacity"],
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        seed=cfg["seed"],
    )
    start = time.monotonic()
    with ParallelEpisodeCollector(
        factory,
        router,
        actors=cfg["actors"],
        episodes_per_actor=cfg["episodes_per_actor"],
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        seed=cfg["seed"],
        timeout=cfg["actor_timeout"],
    ) as collector:
        collector.episode_cursor = episode_cursor
        workflow = MultiPolicyOnPolicyWorkflow(
            collector,
            {
                "shared": OnPolicyStack(
                    policy, GAEProcessor(ppo.gamma, ppo.gae_lambda), learner
                )
            },
        )
        with (args.output / "metrics.jsonl").open("w") as log:
            for iteration in range(
                start_iteration + 1, start_iteration + 1 + cfg["iterations"]
            ):
                before = {
                    k: v.detach().cpu().clone() for k, v in policy.state_dict().items()
                }
                metrics = workflow.step()
                changed = any(
                    not torch.equal(before[k], v.detach().cpu())
                    for k, v in policy.state_dict().items()
                )
                if not changed:
                    raise RuntimeError("Training completed without a parameter update")
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError("Non-finite training metric")
                for k in counters:
                    counters[k] += int(metrics[k])
                row = dict(
                    metrics,
                    iteration=iteration,
                    elapsed_seconds=time.monotonic() - start,
                    actor_stats=collector.last_actor_stats,
                    actor_pids=collector.actor_pids,
                    learner_pid=os.getpid(),
                    parameters_changed=changed,
                    terminal_rewards=[
                        float(t.data.get("reward").sum())
                        for t in workflow.last_collection.trajectories
                    ],
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "episodes": counters["completed_episodes"],
                            "seconds": row["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
                state = {
                    "policy": policy.state_dict(),
                    "optimizer": learner.optimizer.state_dict(),
                    "config": cfg,
                    "iteration": iteration,
                    "counters": counters,
                    "episode_cursor": collector.episode_cursor,
                    "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                }
                temporary = args.output / "checkpoint.tmp"
                torch.save(state, temporary)
                temporary.replace(args.output / "checkpoint.pt")
                if args.eval_every and (iteration % args.eval_every == 0 or iteration == start_iteration + cfg['iterations']):
                    checkpoint=args.output/'checkpoint.pt'
                    score=evaluate_checkpoint(checkpoint,iteration)
                    shutil.copy2(checkpoint,args.output/('iteration_%04d.pt'%iteration))
                    if score > best_ppo_eval:
                        best_ppo_eval=score
                        shutil.copy2(checkpoint,args.output/'ppo_best.pt')
                    if score > best_eval:
                        best_eval=score
                        shutil.copy2(checkpoint,args.output/'eval_best.pt')
                if args.seconds and time.monotonic() - start >= args.seconds:
                    break
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(counters, iteration=iteration, seconds=time.monotonic() - start),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
