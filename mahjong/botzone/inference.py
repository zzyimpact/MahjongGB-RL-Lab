"""Deployment-only model loader; no TensorDict, framework or training imports."""

import json
import hashlib
import os
import pkgutil
import numpy as np
import torch
from ..model import Model, make_model, input_keys


def load():
    game = __package__.split(".")[0]
    config = json.loads(pkgutil.get_data(game, "bot_config.json").decode("utf-8"))
    torch.set_num_threads(1)
    if config.get("feature_set") == "v3":
        from .shanten_backend import BACKEND
        config["shanten_backend"] = BACKEND
        if config.get("activation", "gelu") == "gelu" and not hasattr(torch.nn, "GELU"):
            raise RuntimeError("V3 GELU requires a compatible PyTorch runtime; tested with 2.3.1")
    model = make_model(config)
    path = os.environ.get("BOT_MODEL_PATH", config["weight_path"])
    if config.get("weight_sha256"):
        digest = hashlib.sha256()
        if not os.path.isfile(path):
            raise RuntimeError("Upload the matching weight file to Botzone user storage: " + path)
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != config["weight_sha256"]:
            raise RuntimeError("Uploaded weights do not match this code file (SHA-256 mismatch)")
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    model.eval()

    def choose(arrays):
        with torch.no_grad():
            inputs = [
                torch.from_numpy(np.asarray(arrays[k])).float().unsqueeze(0)
                for k in input_keys(config)
            ]
            logits, _ = model(*inputs)
            mask = torch.from_numpy(arrays["action_mask"]).bool().unsqueeze(0)
            if not mask.any().item():
                raise ValueError("No legal action")
            return int(logits.masked_fill(~mask, -1e9).argmax(dim=-1).item())

    return choose, config
