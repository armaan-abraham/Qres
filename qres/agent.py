import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Bool, Float
from torch import Tensor

from qres.buffer import Buffer
from qres.config import config
from qres.environment import validate_states


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_size = 256
        self.n_hidden = 5

        layers = []

        # Input layer
        layers.extend(
            [
                nn.Linear(config.state_dim, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
            ]
        )

        # Hidden layers
        for _ in range(self.n_hidden - 1):
            layers.extend(
                [
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.LayerNorm(self.hidden_size),
                    nn.ReLU(),
                ]
            )

        # Output layer (no normalization needed for output)
        layers.append(nn.Linear(self.hidden_size, config.action_dim))

        self.model = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        return self.model(x)


def validate_actions(actions: Bool[Tensor, "batch action_dim"]):
    assert (
        actions.shape[1] == config.action_dim
    ), f"Expected actions.shape[1] == {config.action_dim}, got {actions.shape[1]}"
    assert actions.dtype == torch.bool, f"Expected bool, got {actions.dtype}"
    assert (actions.sum(dim=1) == 1).all(), "Actions must be one-hot encoded"


class Agent(torch.nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.steps_done = 0

        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=config.lr,
            weight_decay=config.l2_weight_decay,
            amsgrad=True,
        )

        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.gamma = config.gamma
        self.tau = config.tau

        # Initialize target network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )

    def select_actions(
        self, states: Bool[Tensor, "batch state_dim"], greedy: bool = False
    ) -> Bool[Tensor, "batch action_dim"]:
        if states.dtype != torch.float:
            states = states.to(dtype=torch.float)
        eps_threshold = self.get_epsilon()
        sample = torch.rand(states.shape[0], device=self.device)

        actions = torch.zeros(
            states.shape[0],
            config.action_dim,
            device=self.device,
            dtype=torch.bool,
        )
        greedy_mask = (
            sample > eps_threshold
            if not greedy
            else torch.ones_like(sample, dtype=torch.bool)
        )

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

        validate_actions(actions)

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
        states: Bool[Tensor, "batch state_dim"],
        actions: Bool[Tensor, "batch action_dim"],
        next_states: Bool[Tensor, "batch state_dim"],
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

        self.steps_done += 1

        return loss.item(), state_action_values.mean().item()

    def train(self, buffer: Buffer):
        """Train for specified number of iterations, updating target network periodically"""
        assert (
            buffer.size >= config.train_batch_size
        ), f"Buffer has only {buffer.size} samples, but {config.train_batch_size} are required"
        total_loss = 0
        total_reward = 0
        total_state_action_values_mean = 0

        for step in range(config.train_iter):
            states, actions, next_states, rewards = buffer.sample(
                config.train_batch_size
            )
            validate_actions(actions)
            validate_states(states)
            validate_states(next_states)

            states = states.to(dtype=torch.float)
            next_states = next_states.to(dtype=torch.float)

            loss, state_action_value = self.train_step(
                states, actions, next_states, rewards
            )
            total_loss += loss
            total_reward += torch.mean(rewards).item()
            total_state_action_value += state_action_value
            # Only update target network periodically
            if (step + 1) % config.update_target_every == 0:
                self.update_target_net()

        loss_avg = total_loss / config.train_iter
        reward_avg = total_reward / config.train_iter
        state_action_value_avg = total_state_action_value / config.train_iter
        return loss_avg, reward_avg, state_action_value_avg

    def to(self, device: str):
        inst = self.clone()
        inst.device = device
        inst.policy_net = inst.policy_net.to(device)
        inst.target_net = inst.target_net.to(device)
        for state in inst.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        return inst

    def clone(self):
        clone = Agent(self.device)
        clone.copy_(self)
        return clone

    def copy_(self, other):
        policy_state = {k: v.clone() for k, v in other.policy_net.state_dict().items()}
        target_state = {k: v.clone() for k, v in other.target_net.state_dict().items()}
        optimizer_state = {
            k: {
                k2: v2.clone() if isinstance(v2, torch.Tensor) else v2
                for k2, v2 in v.items()
            }
            if isinstance(v, dict)
            else v
            for k, v in other.optimizer.state_dict().items()
        }
        self.policy_net.load_state_dict(policy_state)
        self.target_net.load_state_dict(target_state)
        self.optimizer.load_state_dict(optimizer_state)
        self.steps_done = other.steps_done
