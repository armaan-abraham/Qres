from multiprocessing import Value
from typing import Dict

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from qres.config import config


class Buffer:
    """
    Shared memory buffer for multiprocessing.
    """

    def __init__(self, device: str):
        assert device == "cpu"
        self.should_share_memory = False
        self.device = device
        self.pos = 0
        self.size = 0

        self.states = torch.zeros(
            (config.max_buffer_size, config.state_dim),
            dtype=config.state_dtype,
            device=self.device,
        )
        self.actions = torch.zeros(
            (config.max_buffer_size, config.action_dim),
            dtype=torch.bool,
            device=self.device,
        )
        self.next_states = torch.zeros(
            (config.max_buffer_size, config.state_dim),
            dtype=config.state_dtype,
            device=self.device,
        )
        self.rewards = torch.zeros((config.max_buffer_size, 1), device=self.device)

    def get_size(self):
        if self.should_share_memory:
            return self.size.value
        else:
            return self.size

    def get_pos(self):
        if self.should_share_memory:
            return self.pos.value
        else:
            return self.pos

    def set_pos(self, value):
        if self.should_share_memory:
            self.pos.value = value
        else:
            self.pos = value

    def set_size(self, value):
        if self.should_share_memory:
            self.size.value = value
        else:
            self.size = value

    def add_single(self, state, action, next_state, reward):
        self.states[self.get_pos()] = state
        self.actions[self.get_pos()] = action
        self.next_states[self.get_pos()] = next_state
        self.rewards[self.get_pos()] = reward

        self.set_pos((self.get_pos() + 1) % config.max_buffer_size)
        self.set_size(min(self.get_size() + 1, config.max_buffer_size))

    def add(
        self,
        states: torch.Tensor,
        actions: Bool[Tensor, "batch action_dim"],
        next_states: torch.Tensor,
        rewards: Float[Tensor, "batch 1"],
    ):
        for state, action, next_state, reward in zip(
            states, actions, next_states, rewards
        ):
            self.add_single(state, action, next_state, reward)

    def sample(self, batch_size):
        max_pos = (
            config.max_buffer_size
            if self.get_size() == config.max_buffer_size
            else self.get_pos()
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
                "pos": self.get_pos(),
                "size": self.get_size(),
            },
            path,
        )

    def to(self, device):
        inst = self.clone()
        inst.device = device
        for attr in inst.__dict__:
            if isinstance(getattr(inst, attr), torch.Tensor):
                setattr(inst, attr, getattr(inst, attr).to(device))
        return inst

    # Multiprocessing functions

    def copy_(self, other):
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), torch.Tensor):
                getattr(self, attr).copy_(getattr(other, attr))
        self.set_pos(other.get_pos())
        self.set_size(other.get_size())

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

    def share_memory(self):
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), torch.Tensor):
                getattr(self, attr).share_memory_()
        self.pos = Value("i", self.get_pos())
        self.size = Value("i", self.get_size())
        self.should_share_memory = True

    def print_memory_usage(self):
        total_bytes = 0
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), torch.Tensor):
                tensor = getattr(self, attr)
                bytes = tensor.element_size() * tensor.nelement()
                total_bytes += bytes
                print(f"{attr}: {bytes / 1024 / 1024:.2f} MB")
        print(f"Total: {total_bytes / 1024 / 1024:.2f} MB")
