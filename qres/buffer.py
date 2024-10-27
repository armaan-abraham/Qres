from typing import Dict

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from qres.config import config


class Buffer:
    def __init__(self, device: str):
        self.device = device
        self.pos = torch.tensor(0, device=device)
        self.size = torch.tensor(0, device=device)

        self.states = torch.zeros(
            (config.max_buffer_size, config.state_dim), device=self.device
        )
        self.actions = torch.zeros(
            (config.max_buffer_size, config.action_dim),
            dtype=torch.bool,
            device=self.device,
        )
        self.next_states = torch.zeros(
            (config.max_buffer_size, config.state_dim), device=self.device
        )
        self.rewards = torch.zeros((config.max_buffer_size, 1), device=self.device)

    def add_single(self, state, action, next_state, reward):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.next_states[self.pos] = next_state
        self.rewards[self.pos] = reward

        self.pos = (self.pos + 1) % config.max_buffer_size
        self.size = min(self.size + 1, config.max_buffer_size)

    def add(
        self,
        states: Float[Tensor, "batch state_dim"],
        actions: Bool[Tensor, "batch action_dim"],
        next_states: Float[Tensor, "batch state_dim"],
        rewards: Float[Tensor, "batch 1"],
    ):
        for state, action, next_state, reward in zip(
            states, actions, next_states, rewards
        ):
            self.add_single(state, action, next_state, reward)

    def sample(self, batch_size):
        max_pos = (
            config.max_buffer_size if self.size == config.max_buffer_size else self.pos
        )
        indices = torch.randint(0, max_pos, (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices],
            self.rewards[indices],
        )

    def save(self, path):
        torch.save(
            {
                "states": self.states,
                "actions": self.actions,
                "next_states": self.next_states,
                "rewards": self.rewards,
                "pos": self.pos,
                "size": self.size,
            },
            path,
        )

    def to(self, device):
        self.device = device
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), torch.Tensor):
                setattr(self, attr, getattr(self, attr).to(device))
        return self

    # Multiprocessing functions

    def share_memory(self):
        assert self.device == "cpu"
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), torch.Tensor):
                getattr(self, attr).share_memory_()

    def copy_(self, other):
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), torch.Tensor):
                getattr(self, attr).copy_(getattr(other, attr))
            else:
                assert attr == "device"

    def clone(self):
        clone = Buffer(self.device)
        clone.copy_(self)
        return clone

    def get_devices(self) -> Dict[str, str]:
        results = {}
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), torch.Tensor):
                results[attr] = getattr(self, attr).device
        return results
