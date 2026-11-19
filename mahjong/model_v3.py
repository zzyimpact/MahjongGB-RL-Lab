"""Compact tile ResNet with ordered public context and shared discard scoring."""
import torch
from torch import nn


def activation(name):
    if name == 'gelu':
        return nn.GELU()
    if name == 'relu':
        return nn.ReLU()
    raise ValueError('Unknown activation: ' + name)


class V3Block(nn.Module):
    def __init__(self, width, act):
        super(V3Block, self).__init__()
        self.layers = nn.Sequential(nn.GroupNorm(8, width), activation(act),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), activation(act),
            nn.Conv2d(width, width, 3, padding=1, bias=False))

    def forward(self, x):
        return x + self.layers(x)


class V3Model(nn.Module):
    def __init__(self, config):
        super(V3Model, self).__init__()
        w, act = config.get('width', 64), config.get('activation', 'gelu')
        self.use_candidate = config.get('use_candidate', True)
        self.use_context = config.get('use_context', True)
        self.detach_value = config.get('detach_value', False)
        self.conv = nn.Sequential(nn.Conv2d(20, w, 3, padding=1),
            *[V3Block(w, act) for _ in range(config.get('blocks', 4))],
            nn.GroupNorm(8, w), activation(act))
        # Categorical tokens keep global/temporal information out of tile planes.
        self.event_seat = nn.Embedding(5, 8, padding_idx=0)
        self.event_type = nn.Embedding(8, 8, padding_idx=0)
        self.tile = nn.Embedding(35, 16, padding_idx=0)
        self.meld_type = nn.Embedding(5, 8, padding_idx=0)
        self.context = nn.Sequential(nn.Linear(16*32 + 16*32, 128), activation(act))
        self.hidden = nn.Sequential(nn.Linear(w*36 + 14 + 128, 256), activation(act))
        self.actor = nn.Linear(256, 235)
        self.discard_context = nn.Linear(256, 64)
        self.discard = nn.Sequential(nn.Linear(w + 58 + 64, 128), activation(act), nn.Linear(128, 1))
        self.critic = nn.Sequential(nn.Linear(256, 128), activation(act), nn.Linear(128, 1))

    def forward(self, obs, history, melds, candidate):
        b = obs.shape[0]
        tiles = self.conv(obs[:, :20])
        scalar = obs[:, 20:].reshape(b, 14, 36)[:, :, :34].mean(-1)
        h, m = history.long(), melds.long().reshape(b, 16, 3)
        he = torch.cat((self.event_seat(h[:, :, 0]), self.event_type(h[:, :, 1]), self.tile(h[:, :, 2])), -1)
        me = torch.cat((self.meld_type(m[:, :, 0]), self.tile(m[:, :, 1]), self.event_seat(m[:, :, 2])), -1)
        context = self.context(torch.cat((he.reshape(b, -1), me.reshape(b, -1)), -1))
        if not self.use_context:
            context = context * 0
        state = self.hidden(torch.cat((tiles.reshape(b, -1), scalar, context), -1))
        quality = candidate if self.use_candidate else candidate * 0
        local = tiles.reshape(b, tiles.shape[1], 36)[:, :, :34].transpose(1, 2)
        shared = self.discard_context(state).unsqueeze(1).expand(-1, 34, -1)
        scores = self.discard(torch.cat((local, quality, shared), -1)).squeeze(-1)
        logits = self.actor(state)
        logits = torch.cat((logits[:, :2], logits[:, 2:36] + scores, logits[:, 36:]), -1)
        value = self.critic(state.detach() if self.detach_value else state).squeeze(-1)
        return logits, value
