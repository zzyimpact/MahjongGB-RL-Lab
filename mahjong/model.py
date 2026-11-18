"""Small portable CNN. The deployment process imports no RL framework."""

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, hidden=64, channels=6):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.hidden = nn.Sequential(nn.Linear(16 * 36, hidden), nn.ReLU())
        self.actor, self.critic = nn.Linear(hidden, 235), nn.Linear(hidden, 1)

    def forward(self, obs):
        x = self.hidden(self.conv(obs).reshape(obs.shape[0], -1))
        return self.actor(x), self.critic(x).squeeze(-1)



class PreActBlock(nn.Module):
    def __init__(self, channels=64):
        super(PreActBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False))

    def forward(self, x):
        return x + self.layers(x)


class SmallResNet(nn.Module):
    def __init__(self, channels=6, hidden=256):
        super(SmallResNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, 64, 3, padding=1),
                                  *[PreActBlock() for _ in range(4)],
                                  nn.GroupNorm(8,64), nn.ReLU())
        self.hidden = nn.Sequential(nn.Linear(64*36, hidden), nn.ReLU())
        self.actor, self.critic = nn.Linear(hidden,235), nn.Linear(hidden,1)

    def forward(self, obs):
        x = self.hidden(self.conv(obs).reshape(obs.shape[0], -1))
        return self.actor(x), self.critic(x).squeeze(-1)


def model_spec(config):
    feature = config.get('feature_set', 'basic')
    if feature not in ('basic', 'enhanced'):
        raise ValueError('Unknown features: ' + feature)
    version = 'v1' if feature == 'basic' else 'v2'
    if config.get('feature_version', version) != version:
        raise ValueError('Feature version mismatch')
    architecture = config.get('architecture', 'tiny_cnn')
    if architecture not in ('tiny_cnn', 'small_resnet'):
        raise ValueError('Unknown architecture: ' + architecture)
    return dict(feature_set=feature, feature_version=version, architecture=architecture,
                hidden=config.get('hidden',64), resnet_hidden=config.get('resnet_hidden',256))


def make_model(config):
    spec = model_spec(config)
    channels = 6 if spec['feature_set']=='basic' else 34
    if spec['architecture']=='tiny_cnn':
        return Model(spec['hidden'], channels)
    return SmallResNet(channels, spec['resnet_hidden'])
