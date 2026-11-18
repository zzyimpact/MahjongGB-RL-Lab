"""Sparse four-seat environment; finished players receive the final team reward."""

import torch
from tensordict import TensorDict
from rlframework.environment import MultiEnv, MultiEnvStep, AgentStep
from .gamecore.engine import GameCore
from .features import encode, CAPACITY


class Environment(MultiEnv):
    def __init__(self, capacity=CAPACITY):
        self.core, self.capacity = GameCore(), capacity
        self.last_obs = {}
        self.candidate_overflows = 0

    def reset(self, config=None):
        config = config or {}
        seed = int(config.get("seed", 0))
        self.last_obs = {}
        self.core.reset(seed=seed, level="234567890JQKA"[seed % 13])
        return self._result()

    def _result(self):
        rewards = TensorDict({"rewards": torch.tensor(self.core.rewards)}, [])
        if self.core.done:
            return MultiEnvStep(
                {
                    p: AgentStep(obs, terminated=True)
                    for p, obs in self.last_obs.items()
                },
                True,
                rewards,
            )
        player = str(self.core.current)
        arrays, self.options, total = encode(
            self.core.view(self.core.current), self.capacity
        )
        self.candidate_overflows += int(total > self.capacity)
        obs = TensorDict(
            {key: torch.from_numpy(value) for key, value in arrays.items()}, []
        )
        self.last_obs[player] = obs
        return MultiEnvStep({player: AgentStep(obs)}, False, rewards)

    def step(self, actions):
        if set(actions) != {str(self.core.current)}:
            raise ValueError("Exactly the current seat must act")
        index = int(actions[str(self.core.current)].get("action"))
        if not 0 <= index < len(self.options):
            raise ValueError("Invalid candidate index")
        self.core.step(self.options[index])
        return self._result()
