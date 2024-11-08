import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor

from qres.buffer import Buffer
from qres.config import N_AMINO_ACIDS, config
from qres.environment import parse_seqs_from_states, validate_states

"""
- embedding
    - token embedding
    - positional embedding
    - d_model big enough to accommodate both init and current state
- MLP
- self-attention
    - query, key, value
    - output
- residual connection
- layer norm
- final output (on entire sequence)
"""


class ContentEmbedding(nn.Module):
    """
    Instantiated separately for init and current state
    """

    def __init__(self):
        super().__init__()
        self.W_embed = nn.Parameter(
            nn.init.uniform_(
                torch.empty(N_AMINO_ACIDS, config.d_model, dtype=torch.float), -1, 1
            )
        )

    def forward(
        self, seqs: Int[Tensor, "sequence residue"]
    ) -> Float[Tensor, "sequence residue d_model"]:
        assert seqs.ndim == 2
        assert seqs.dtype in [torch.int8, torch.int16, torch.int32]
        seqs = seqs.to(dtype=torch.int32)
        # We just have one embedding vector for each amino acid
        result = self.W_embed[seqs]
        assert result.shape == (seqs.shape[0], seqs.shape[1], config.d_model)
        return result


class PositionalEmbedding(nn.Module):
    """
    Only applied once for init and current state
    """

    def __init__(self):
        super().__init__()
        self.W_pos = nn.Parameter(
            nn.init.uniform_(
                torch.empty(config.seq_len, config.d_model, dtype=torch.float), -1, 1
            )
        )

    def forward(
        self, seqs: Int[Tensor, "sequence residue"]
    ) -> Float[Tensor, "sequence residue d_model"]:
        assert seqs.ndim == 2
        # we have a positional embedding vector for each sequence position
        result = einops.repeat(
            self.W_pos,
            "residue d_model -> sequence residue d_model",
            sequence=seqs.shape[0],
        )
        assert result.shape == (seqs.shape[0], seqs.shape[1], config.d_model)
        return result


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_init = ContentEmbedding()
        self.content_current = ContentEmbedding()
        self.positional = PositionalEmbedding()

    def forward(
        self, states: Int[Tensor, "batch state_dim"]
    ) -> Float[Tensor, "batch residue d_model"]:
        """
        Each output is just seq_len length as we combine the init and current seqs
        """
        init_seqs, current_seqs = parse_seqs_from_states(states)
        init_embeds = self.content_init(init_seqs)
        current_embeds = self.content_current(current_seqs)
        pos_embeds = self.positional(current_seqs)
        result = init_embeds + pos_embeds + current_embeds
        assert result.shape == (states.shape[0], config.seq_len, config.d_model)
        return result


class LayerNorm(nn.Module):
    def __init__(self, scale=True):
        super().__init__()
        self.w = nn.Parameter(torch.ones(config.d_model, dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(config.d_model, dtype=torch.float))

    def forward(
        self, x: Float[Tensor, "seq seq_len d_model"]
    ) -> Float[Tensor, "seq seq_len d_model"]:
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        result = self.w * (x - mean) / torch.sqrt(var + config.layer_norm_eps) + self.b
        assert result.shape == x.shape
        return result


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_in = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(config.d_model, config.d_mlp, dtype=torch.float).T
            ).T
        )
        self.b_in = nn.Parameter(torch.zeros(config.d_mlp, dtype=torch.float))
        self.W_out = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(config.d_mlp, config.d_model, dtype=torch.float).T
            ).T
        )
        self.b_out = nn.Parameter(torch.zeros(config.d_model, dtype=torch.float))

    def forward(
        self, x: Float[Tensor, "seq seq_len d_model"]
    ) -> Float[Tensor, "seq seq_len d_model"]:
        wx1 = (
            einsum(
                x, self.W_in, "seq seq_len d_model, d_model d_mlp -> seq seq_len d_mlp"
            )
            + self.b_in
        )
        a1 = F.relu(wx1)
        wx2 = (
            einsum(
                a1,
                self.W_out,
                "seq seq_len d_mlp, d_mlp d_model -> seq seq_len d_model",
            )
            + self.b_out
        )
        return wx2


class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(config.d_model, config.d_head, dtype=torch.float).T
            ).T
        )
        self.W_K = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(config.d_model, config.d_head, dtype=torch.float).T
            ).T
        )
        self.W_V = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(config.d_model, config.d_head, dtype=torch.float).T
            ).T
        )
        self.W_O = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(config.d_head, config.d_model, dtype=torch.float).T
            ).T
        )

        self.b_Q = nn.Parameter(torch.zeros(config.d_head, dtype=torch.float))
        self.b_K = nn.Parameter(torch.zeros(config.d_head, dtype=torch.float))
        self.b_V = nn.Parameter(torch.zeros(config.d_head, dtype=torch.float))
        self.b_O = nn.Parameter(torch.zeros(config.d_model, dtype=torch.float))

    def forward(
        self, x: Float[Tensor, "seq residue d_model"]
    ) -> Float[Tensor, "seq residue d_model"]:
        Q = (
            einsum(
                x, self.W_Q, "seq residue d_model, d_model d_head -> seq residue d_head"
            )
            + self.b_Q
        )
        K = (
            einsum(
                x, self.W_K, "seq residue d_model, d_model d_head -> seq residue d_head"
            )
            + self.b_K
        )
        V = (
            einsum(
                x, self.W_V, "seq residue d_model, d_model d_head -> seq residue d_head"
            )
            + self.b_V
        )

        dot_prods = einsum(
            Q,
            K,
            "seq residue_Q d_head, seq residue_K d_head -> seq residue_Q residue_K",
        ) / np.sqrt(config.d_head)

        attn = F.softmax(dot_prods, dim=-1)

        Z = einsum(
            attn,
            V,
            "seq residue_Q residue_K, seq residue_K d_head -> seq residue_Q d_head",
        )

        out = (
            einsum(
                Z,
                self.W_O,
                "seq residue_Q d_head, d_head d_model -> seq residue_Q d_model",
            )
            + self.b_O
        )
        assert out.shape == (x.shape[0], x.shape[1], config.d_model)
        return out


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead() for _ in range(config.n_heads)])

    def forward(
        self, x: Float[Tensor, "seq residue d_model"]
    ) -> Float[Tensor, "seq residue d_model"]:
        result_expanded = torch.stack([head(x) for head in self.heads])
        assert result_expanded.shape == (
            config.n_heads,
            x.shape[0],
            x.shape[1],
            config.d_model,
        )
        result = einops.reduce(
            result_expanded, "n_heads seq residue d_model -> seq residue d_model", "sum"
        )
        assert result.shape == (x.shape[0], x.shape[1], config.d_model)
        return result


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_attn = LayerNorm()
        self.attn_block = Attention()
        self.layer_norm_mlp = LayerNorm()
        self.mlp = MLP()

    def forward(
        self, x: Float[Tensor, "seq residue d_model"]
    ) -> Float[Tensor, "seq residue d_model"]:
        # self attention
        x_attn = self.attn_block(x)
        # residual connection
        x = x + x_attn
        x = self.layer_norm_attn(x)
        # mlp
        x_mlp = self.mlp(x)
        # residual connection
        x = x + x_mlp
        x = self.layer_norm_mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(config.n_layers)]
        )

    def forward(
        self, x: Float[Tensor, "seq residue d_model"]
    ) -> Float[Tensor, "seq residue d_model"]:
        for block in self.blocks:
            x = block(x)
        return x


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Embedding()
        self.transformer = Transformer()
        self.W_act = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(config.d_model, config.action_dim, dtype=torch.float).T
            ).T
        )
        self.b_act = nn.Parameter(torch.zeros(config.action_dim, dtype=torch.float))

    def forward(
        self, states: Int[Tensor, "batch state_dim"]
    ) -> Float[Tensor, "batch action_dim"]:
        embeds = self.embed(states)  # batch seq_len d_model
        transformed = self.transformer(embeds)  # batch seq_len d_model
        result = (
            einsum(
                transformed,
                self.W_act,
                "batch seq_len d_model, d_model action_dim -> batch action_dim",
            )
            + self.b_act
        )
        assert result.shape == (states.shape[0], config.action_dim)
        return result


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
        eps_threshold = self.get_epsilon()
        sample = torch.rand(states.shape[0], device=self.device)

        actions = torch.zeros(
            states.shape[0],
            config.action_dim,
            device=self.device,
            dtype=torch.bool,
        )

        with torch.no_grad():
            if greedy:
                q_values = self.policy_net(states)
                print(f"q_values mean: {q_values.mean().item()}")
                max_indices = torch.argmax(q_values, dim=1)
                actions[range(states.shape[0]), max_indices] = True

            else:
                greedy_mask = (
                    sample > eps_threshold
                )

                # Get actions for greedy choices
                if greedy_mask.any():
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
        states: Int[Tensor, "batch state_dim"],
        actions: Bool[Tensor, "batch action_dim"],
        next_states: Int[Tensor, "batch state_dim"],
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
        total_state_action_value = 0

        for step in range(config.train_iter):
            states, actions, next_states, rewards = buffer.sample(
                config.train_batch_size
            )
            validate_actions(actions)
            validate_states(states)
            validate_states(next_states)

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

    def get_n_params(self):
        return sum(p.numel() for p in self.policy_net.parameters())
