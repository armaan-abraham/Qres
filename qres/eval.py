from pathlib import Path

import numpy as np
import torch

from qres.agent import Agent
from qres.config import config
from qres.environment import Environment, parse_seqs_from_states


def main():
    # Specify model path
    model_path = (
        Path(__file__).parent / "data" / "run_truly-picked-snipe" / "model_3499.pt"
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize agent
    agent = Agent(device=device)

    checkpoint = torch.load(model_path, map_location=device)

    # Verify all parameters are present in state dicts
    policy_missing = set(agent.policy_net.state_dict().keys()) - set(
        checkpoint["policy_net_state_dict"].keys()
    )
    target_missing = set(agent.target_net.state_dict().keys()) - set(
        checkpoint["target_net_state_dict"].keys()
    )
    if policy_missing or target_missing:
        raise ValueError(
            f"Missing parameters in checkpoint: Policy net: {policy_missing}, Target net: {target_missing}"
        )

    agent.policy_net.load_state_dict(
        checkpoint["policy_net_state_dict"], strict=True, assign=True
    )
    agent.target_net.load_state_dict(
        checkpoint["target_net_state_dict"], strict=True, assign=True
    )
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # agent.steps_done = checkpoint.get("steps_done", 0)
    agent = agent.to(device)
    # print(f"steps_done: {agent.steps_done}")

    # Ensure the optimizer's state is on the correct device
    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Set the agent to evaluation mode
    agent.policy_net.eval()
    agent.target_net.eval()
    agent.steps_done = 500000

    # Initialize environment with multiple states
    batch_size = 500
    env = Environment(device=device, batch_size=batch_size)

    # Number of steps to run
    num_steps = 50

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

        # Save rewards and sequences
        rewards.append(reward.tolist())  # Convert tensor to list for multiple rewards

        # Decode sequences from states
        _, seqs = parse_seqs_from_states(states)  # seqs: (batch_size, seq_len)
        seq_strings = [env.decode_seq(seq) for seq in seqs]  # Decode each sequence
        sequences.append(seq_strings)

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

    # Print amino acid sequences along the trajectory
    # print("\nSequence Evolution:")
    for seq_idx in range(batch_size):
        print(f"\nSequence {seq_idx + 1}:")
        print(f"Average reward: {np.mean(rewards[:, seq_idx])}")
        try:
            for step_idx, (step_seqs, step_rewards) in enumerate(
                zip(sequences, rewards)
            ):
                print(
                    f"  Step {step_idx + 1:<4d}: Sequence: {step_seqs[seq_idx]}, Reward: {step_rewards[seq_idx]}"
                )
        except Exception as e:
            print(e)

    print(f"Average reward: {np.mean(rewards)}")


if __name__ == "__main__":
    main()
