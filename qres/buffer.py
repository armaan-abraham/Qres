import torch
from pathlib import Path
from qres.config import config


class Buffer:
    def __init__(self, device: str):
        self.device = device
        self.pos = 0
        self.size = 0

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

    def add(self, states, actions, next_states, rewards):
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
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.next_states = self.next_states.to(device)
        self.rewards = self.rewards.to(device)
        return self

    def clone(self):
        new_buffer = Buffer(self.device, self.convert_new_to_device)
        new_buffer.pos = self.pos
        new_buffer.size = self.size
        new_buffer.states = self.states.clone()
        new_buffer.actions = self.actions.clone()
        new_buffer.next_states = self.next_states.clone()
        new_buffer.rewards = self.rewards.clone()
        return new_buffer
