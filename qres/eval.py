# %%
from pathlib import Path

import numpy as np
import torch
import json
import datetime

from qres.agent import Agent
from qres.config import config
import qres.environment
import importlib
importlib.reload(qres.environment)
from qres.environment import Environment, parse_seqs_from_states

# %%

data_path = Path(__file__).parent / "data"

# Specify model path
model_path = (
    data_path / "run_overly-keen-whale" / "model_final.pt"
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize agent
agent = Agent.load_model(model_path, device=device)
print(f"Steps done: {agent.get_steps_done()}")

# Initialize environment with multiple states
batch_size = 50

# Check if env is already defined to avoid reinitialization
if 'env' in locals():
    print("Environment already initialized")
    env = Environment(device=device, batch_size=batch_size, structure_predictor=env.structure_predictor)
else:
    env = Environment(device=device, batch_size=batch_size)

# %%

# Number of steps to run
num_steps = 20

rewards = []
sequences = []

for step in range(num_steps):
    print(f"step: {step}")
    states = env.get_states()  # Now shape: (batch_size, state_dim)

    with torch.no_grad():
        actions = agent.select_actions(
            states, greedy=False
        )  # Shape: (batch_size, action_dim)
    # print(f"actions: {torch.argmax(actions.to(dtype=torch.float), dim=1)}")
    # print(f"states: {states}")

    next_states, reward = env.step(
        states, actions
    )  # next_states: (batch_size, state_dim), reward: (batch_size, 1)
    # print(f"next_states: {next_states}")


    # Decode sequences from states
    _, seqs = parse_seqs_from_states(states)  # seqs: (batch_size, seq_len)
    seq_strings = [env.decode_seq(seq) for seq in seqs]  # Decode each sequence
    sequences.append(seq_strings)

    confidences = env.infer_confidence(seq_strings).cpu().numpy().squeeze()
    rewards.append(confidences)

    if step > 0:
        assert next_seq_strings == seq_strings

    _, next_seqs = parse_seqs_from_states(next_states)
    next_seq_strings = [env.decode_seq(seq) for seq in next_seqs]
    print(f"last reward: {env.last_reward}")
    print(f"last confidence: {env.last_confidence}")
    print(f"last distance penalty: {env.last_distance_penalty}")

rewards = np.array(rewards)
assert rewards.shape == (num_steps, batch_size)
print("Avg reward by step:")
print(rewards.mean(axis=1))

print("Avg reward by sequence:")
print(rewards.mean(axis=0))

print(f"Average reward: {np.mean(rewards)}")

# transpose sequences
sequences = np.array(sequences).T
rewards = np.array(rewards).T # batch_size x num_steps

for traj_idx, traj in enumerate(sequences):
    print(f"\nTrajectory {traj_idx + 1}:")
    for step_idx, step_seq in enumerate(traj):
        if step_idx < num_steps - 1:
            diff = [c2 for c1, c2 in zip(step_seq, traj[step_idx + 1]) if c1 != c2]
            diff = diff[0] if diff else ""
        print(f"  Step {step_idx + 1:<4d}: Sequence: {step_seq}, Confidence: {rewards[traj_idx, step_idx]}, Diff (next): {diff}")
    print("[")
    for seq in traj:
        print(f'        "{seq}",')
    print("],")

print(f"Average reward: {np.mean(rewards)}")