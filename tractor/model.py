"""Portable candidate-scoring actor and state-value critic; customize here."""

import torch
from torch import nn
from .features import OBS_SIZE, ACTION_SIZE


class Model(nn.Module):
    def __init__(self, hidden=64):
        super(Model, self).__init__()
        self.state = nn.Sequential(nn.Linear(OBS_SIZE, hidden), nn.Tanh())
        self.action = nn.Sequential(nn.Linear(ACTION_SIZE, hidden), nn.Tanh())
        self.score = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs, candidates):
        state = self.state(obs)
        features = torch.tanh(state.unsqueeze(-2) + self.action(candidates))
        return self.score(features).squeeze(-1), self.value(state).squeeze(-1)
