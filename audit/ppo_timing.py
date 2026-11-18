"""Audit: PPO iteration time breakdown (collection vs GAE vs update).

Small-scale timing run (3 iterations, 8 actors). Does NOT save checkpoints
or modify any existing artifact. Outputs audit/ppo_timing.json.
"""

import json
import resource
import time
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
from mahjong.policy import make_policy, make_collector
from rlframework.collectors.parallel import ParallelEpisodeCollector
from rlframework.collectors.routing import StaticPolicyRouter
from rlframework.collector import PolicyRef
from rlframework.algorithm.ppo import PPOLearner, PPOConfig
from rlframework.processing.gae import GAEProcessor
from rlframework.workflow.on_policy import MultiPolicyOnPolicyWorkflow, OnPolicyStack


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    cfg = json.loads((ROOT / "mahjong/configs/default.json").read_text())
    policy = make_policy(cfg["hidden"])
    ppo = PPOConfig(**cfg["ppo"])
    learner = PPOLearner(policy, ppo, device="cpu")
    router = StaticPolicyRouter({str(i): PolicyRef("shared", policy) for i in range(4)})
    factory = partial(
        make_collector,
        hidden=cfg["hidden"],
        capacity=cfg["capacity"],
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        seed=cfg["seed"],
    )
    rows = []
    with ParallelEpisodeCollector(
        factory, router, actors=8, episodes_per_actor=1,
        gamma=ppo.gamma, gae_lambda=ppo.gae_lambda, seed=cfg["seed"], timeout=300,
    ) as collector:
        workflow = MultiPolicyOnPolicyWorkflow(
            collector,
            {"shared": OnPolicyStack(policy, GAEProcessor(ppo.gamma, ppo.gae_lambda), learner)},
        )
        for it in range(4):
            ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
            t0 = time.perf_counter()
            collected = collector.collect_episode()
            t1 = time.perf_counter()
            grouped = [
                workflow.stacks["shared"].processor.process(tr.data, policy)
                for tr in collected.trajectories
            ]
            t2 = time.perf_counter()
            batch = torch.cat(grouped, 0)
            metrics = learner.update(batch)
            t3 = time.perf_counter()
            ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            child_cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
            workflow._publish("shared", workflow.stacks["shared"])
            rows.append(
                {
                    "iteration": it,
                    "collect_s": round(t1 - t0, 4),
                    "gae_s": round(t2 - t1, 4),
                    "update_s": round(t3 - t2, 4),
                    "total_s": round(t3 - t0, 4),
                    "agent_steps": collected.agent_steps,
                    "episodes": collected.completed_episodes,
                    "gradient_updates": metrics["gradient_updates"],
                    "child_cpu_s": round(child_cpu, 3),
                }
            )
    out = {
        "iterations": rows,
        "note": "iteration 0 includes actor startup; child_cpu_s = actor processes user+sys CPU during the iteration (8 actors, 15 cores)",
        "steady_collect_frac": round(
            sum(r["collect_s"] for r in rows[1:]) / sum(r["total_s"] for r in rows[1:]), 3
        ),
    }
    (ROOT / "audit/ppo_timing.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()