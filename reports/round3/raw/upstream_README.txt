---
license: mit
task_categories:
  - other
tags:
  - mahjong
  - imitation-learning
  - ijcai-2026
pretty_name: IJCAI-2026 MCR Mahjong Training Corpora
---

# IJCAI-2026 MCR Mahjong — Training Corpora

Behavior-cloning / distillation corpora for the IJCAI-2026 MCR Mahjong campaign
(entry **kdens3**). Code: https://github.com/SuuTTT/IJCAI-mahjong.
Companion repos: models `Dannibal/ijcai-mahjong-ckpts-2026`, eval testset
`Dannibal/mcr-final2026-testset`.

## Files

| File | Size | Description |
|---|---|---|
| `cooked_single.npz` | 184M | Base BC training corpus cooked from the 2025 agents pool (features + move labels). |
| `cooked_quarter.npz` | 59M | Quarter-size subset of `cooked_single` for fast iteration / ablations. |
| `final2_cai_corpus.npz` | 23M | Final-2 CAI (champion-imitation) corpus harvested from final-stage games. |
| `final2_bc_corpus.npz` | 30M | Final-2 behavior-cloning corpus. |
| `final2_all.jsonl.gz` | 46M | Raw final-2 game replays (one game per line). |
| `final2_games_summary.jsonl.gz` | 0.2M | Per-game summary rows for the final-2 harvest. |

The `.npz` files are numpy archives (load with `numpy.load(..., allow_pickle=False)`);
the harvest/build scripts (`harvest.py`, `build_corpus_cai.py`, `analyze.py`) are in
`campaign/final2_harvest/` of the code repo.

## Not included (regenerable)
Self-play shard corpora used for the opponent-belief and deal-in predictors
(`oppbelief` ~4.3G, `oppdealin` ~1.8G) are regenerable from the self-play
generators in `train/caiest_repro` and are not stored here.
