import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from qres.structure_prediction import N_AMINO_ACIDS
from qres.config import config
from jaxtyping import Float, Bool
from torch import Tensor
from qres.buffer import Buffer
from qres.logger import logger


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        hidden_size = 256

        self.model = nn.Sequential(
            nn.Linear(config.state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, config.action_dim),
        )

    def forward(self, x):
        return self.model(x)


class Agent:
    def __init__(self, device: str):
        self.device = device
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=config.lr, amsgrad=True
        )

        self.steps_done = 0
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.gamma = config.gamma
        self.tau = config.tau

        # Initialize target network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def validate_actions(self, actions: Bool[Tensor, "batch action_dim"]):
        assert actions.shape == (
            config.train_batch_size,
            config.action_dim,
        )
        assert actions.dtype == torch.bool, f"Expected bool, got {actions.dtype}"
        assert (actions.sum(dim=1) == 1).all(), "Actions must be one-hot encoded"

    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )

    def select_actions(
        self, states: Float[Tensor, "batch state_dim"]
    ) -> Bool[Tensor, "batch action_dim"]:
        eps_threshold = self.get_epsilon()
        self.steps_done += 1
        sample = torch.rand(states.shape[0], device=self.device)

        actions = torch.zeros(
            states.shape[0],
            config.action_dim,
            device=self.device,
            dtype=torch.bool,
        )
        greedy_mask = sample > eps_threshold

        # Get actions for greedy choices
        if greedy_mask.any():
            with torch.no_grad():
                q_values = self.policy_net(states[greedy_mask])
                max_indices = torch.argmax(q_values, dim=1)
                actions[greedy_mask, max_indices] = True

        # Get random actions for exploration
        random_mask = ~greedy_mask
        if random_mask.any():
            random_actions = torch.randint(
                0,
                config.action_dim,
                (random_mask.sum().item(),),
                device=self.device,
            )
            actions[random_mask, random_actions] = True

        self.validate_actions(actions)

        return actions

    def update_target_net(self):
        # Soft update of the target network's weights
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def save_model(self, path):
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def train_step(
        self,
        states: Float[Tensor, "batch state_dim"],
        actions: Bool[Tensor, "batch action_dim"],
        next_states: Float[Tensor, "batch state_dim"],
        rewards: Float[Tensor, "batch 1"],
    ) -> float:
        """Single optimization step"""
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(states)[actions]
        assert state_action_values.shape == (config.train_batch_size,)

        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
        assert next_state_values.shape == (config.train_batch_size,)

        # Compute expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + rewards.squeeze(1)

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def train(self, buffer: Buffer):
        """Train for specified number of iterations, then update target network"""
        assert buffer.size >= config.train_batch_size
        assert config.train_batch_size <= config.structure_predictor_batch_size
        total_loss = 0
        for _ in range(config.train_iter):
            states, actions, next_states, rewards = buffer.sample(config.train_batch_size)
            loss = self.train_step(states, actions, next_states, rewards)
            total_loss += loss

        # Update target network after all iterations
        self.update_target_net()
        avg_loss = total_loss / config.train_iter
        logger.put(Loss=avg_loss)

    def to(self, device: str):
        self.device = device
        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)
        return self

    def clone(self):
        new_agent = Agent(self.device)
        new_agent.policy_net.load_state_dict(self.policy_net.state_dict())
        new_agent.target_net.load_state_dict(self.target_net.state_dict())
        new_agent.optimizer = optim.AdamW(
            new_agent.policy_net.parameters(), 
            lr=config.lr,
            amsgrad=True
        )
        new_agent.steps_done = self.steps_done
        return new_agent

