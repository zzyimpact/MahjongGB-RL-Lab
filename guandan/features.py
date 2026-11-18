"""Student extension point: private state and candidate features (NumPy only)."""

import numpy as np
from .actions import candidates

CAPACITY = 128
OBS_SIZE = 54 * 6 + 4 + 13 + 5
ACTION_SIZE = 108


def counts(cards):
    out = np.zeros(54, dtype=np.float32)
    for c in cards:
        out[c % 54] += 0.5
    return out


def encode(view, capacity=CAPACITY):
    options, total = candidates(view, capacity)
    seat = view["id"]
    level = np.zeros(13, dtype=np.float32)
    level["A234567890JQK".index(view["level"])] = 1
    major = np.zeros(5, dtype=np.float32)
    major["shcdn".index(view.get("major", "n"))] = 1
    played = [view["played"][(seat + i) % 4] for i in range(4)]
    last = view.get("last_claim") or (view.get("history") or [[]])[0]
    obs = np.concatenate(
        [
            counts(view["hand"]),
            *[counts(x) for x in played],
            counts(last),
            np.array(
                [view["counts"][(seat + i) % 4] / 27.0 for i in range(4)],
                dtype=np.float32,
            ),
            level,
            major,
        ]
    )
    actions = np.zeros((capacity, ACTION_SIZE), dtype=np.float32)
    for i, option in enumerate(options):
        actions[i] = np.concatenate([counts(option[0]), counts(option[1])])
    mask = np.arange(capacity) < len(options)
    return {"obs": obs, "candidates": actions, "action_mask": mask}, options, total
