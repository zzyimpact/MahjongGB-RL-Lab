"""Audit: exact model structure, parameter counts, forward microbenchmarks.

Read-only (loads checkpoints, never writes them). Outputs audit/model_bench.json
and audit/model_structure.txt.
"""

import json
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
from mahjong.model import Model
from mahjong.features import encode
from mahjong.observation import SeatView
from mahjong.gamecore.engine import GameCore


def layer_table(model):
    rows = []
    x = torch.zeros(1, 6, 4, 9)
    hooks = []
    shapes = {}

    def hook(name):
        def fn(module, inp, out):
            shapes[name] = {
                "in": [tuple(t.shape) for t in inp if isinstance(t, torch.Tensor)],
                "out": tuple(out.shape) if isinstance(out, torch.Tensor) else None,
            }
        return fn

    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(hook(name)))
    model(x)
    for h in hooks:
        h.remove()
    for name, module in model.named_modules():
        if name and not list(module.children()):
            params = sum(p.numel() for p in module.parameters(recurse=False))
            rows.append(
                {
                    "layer": name,
                    "type": type(module).__name__,
                    "detail": str(module).replace("\n", " "),
                    "params": params,
                    "io_shapes": shapes.get(name),
                }
            )
    return rows


def bench_forward(model, device, batch, reps=50, warmup=10):
    model = model.to(device)
    x = torch.randn(batch, 6, 4, 9, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "mps":
            torch.mps.synchronize()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            if device == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {
        "batch": batch,
        "device": device,
        "median_ms": round(med * 1000, 4),
        "p90_ms": round(sorted(times)[int(0.9 * reps)] * 1000, 4),
        "states_per_sec": round(batch / med, 1),
    }


def main():
    out = {}
    model = Model(64)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out["params"] = {
        "total": total,
        "trainable": trainable,
        "fp32_theoretical_bytes": total * 4,
        "by_module": {
            name: sum(p.numel() for p in mod.parameters())
            for name, mod in [
                ("conv", model.conv),
                ("hidden", model.hidden),
                ("actor", model.actor),
                ("critic", model.critic),
            ]
        },
    }
    out["layers"] = layer_table(model)
    structure = repr(model) + "\n\n" + "\n".join(
        "%-14s %-12s params=%-7d io=%s" % (r["layer"], r["type"], r["params"], r["io_shapes"])
        for r in out["layers"]
    )
    (ROOT / "audit/model_structure.txt").write_text(structure)

    # Load time of deployment weights (cold-ish: file cache warm).
    weights_path = ROOT / "submissions/mahjong_v1/mahjong_model.pt"
    t0 = time.perf_counter()
    state = torch.load(weights_path, map_location="cpu")
    m2 = Model(64)
    m2.load_state_dict(state)
    m2.eval()
    out["load_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Forward benchmarks. CPU single-thread matches deployment (set_num_threads(1)).
    torch.set_num_threads(1)
    benches = []
    for device in ("cpu", "mps"):
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        for batch in (1, 64, 256, 1024):
            benches.append(bench_forward(Model(64), device, batch))
    out["forward_bench"] = benches

    # Full bot-style decision latency on CPU: encode + forward + masked argmax.
    view = SeatView(0)
    view.request2obs("Wind 0")
    view.request2obs("Deal W1 W2 W3 W4 W5 B1 B2 B3 T1 T2 T3 J1 J2")
    view.request2obs("Draw J3")
    arrays = encode(view)
    obs = torch.from_numpy(arrays["obs"]).unsqueeze(0)
    mask = torch.from_numpy(arrays["action_mask"]).unsqueeze(0)
    with torch.no_grad():
        for _ in range(20):
            logits, _ = m2(obs)
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            a = encode(view)
            o = torch.from_numpy(a["obs"]).unsqueeze(0)
            mk = torch.from_numpy(a["action_mask"]).unsqueeze(0)
            lg, _ = m2(o)
            int(lg.masked_fill(~mk, -1e9).argmax(dim=-1).item())
            times.append(time.perf_counter() - t0)
    out["decision_pipeline_cpu_ms"] = {
        "median": round(statistics.median(times) * 1000, 4),
        "includes": "features.encode + torch.from_numpy + forward + masked argmax",
    }

    # Environment-side cost: judge step + event replay + encode, random policy.
    core = GameCore()
    core.r.seed(123)
    core.reset()
    views = [SeatView(i) for i in range(4)]
    offsets = [0] * 4
    rng = np.random.RandomState(0)
    env_times, fwd_times = [], []
    steps = 0
    while not core.done and steps < 400:
        t0 = time.perf_counter()
        for i in range(4):
            ev = core.agents[i].events
            for e in ev[offsets[i]:]:
                views[i].request2obs(e)
            offsets[i] = len(ev)
        actions = {}
        arrays_by_seat = {}
        for i in core.obs:
            arrays_by_seat[i] = encode(views[i])
        env_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        with torch.no_grad():
            for i, a in arrays_by_seat.items():
                o = torch.from_numpy(a["obs"]).unsqueeze(0)
                mk = torch.from_numpy(a["action_mask"]).unsqueeze(0)
                lg, _ = m2(o)
                actions[core.agent_names[i]] = views[i].action2response(
                    int(lg.masked_fill(~mk, -1e9).argmax(dim=-1).item())
                )
        fwd_times.append(time.perf_counter() - t0)
        core.step(actions)
        steps += 1
    out["env_vs_forward"] = {
        "joint_steps": steps,
        "env_median_ms": round(statistics.median(env_times) * 1000, 4),
        "forward_median_ms": round(statistics.median(fwd_times) * 1000, 4),
        "note": "env = event replay + fan-calculator legality checks + encode for all acting seats; forward = model decisions for the same seats",
    }
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1)
    out["rss_note"] = "ru_maxrss is process-lifetime peak (macOS bytes); includes imports and benchmarks, not inference-only"
    (ROOT / "audit/model_bench.json").write_text(json.dumps(out, indent=2))
    print(structure)
    print(json.dumps({k: out[k] for k in ("params", "load_ms", "decision_pipeline_cpu_ms", "env_vs_forward", "peak_rss_mb")}, indent=2))
    for b in benches:
        print(b)


if __name__ == "__main__":
    main()