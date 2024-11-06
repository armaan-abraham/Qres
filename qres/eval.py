from pathlib import Path

import torch

from qres.agent import Agent
from qres.config import config
from qres.environment import Environment, parse_seqs_from_states


def main():
    # Specify model path
    model_path = Path(__file__).parent / "data" / "run_24" / "model_final.pt"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize agent
    agent = Agent(device=device)

    # Load the saved model
    checkpoint = torch.load(model_path, map_location=device)
    agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.steps_done = checkpoint.get("steps_done", 0)

    # Ensure the optimizer's state is on the correct device
    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Set the agent to evaluation mode
    agent.policy_net.eval()
    agent.target_net.eval()

    # Initialize environment with multiple states
    batch_size = 5  # Adjust this number to evaluate more sequences simultaneously
    env = Environment(device=device, batch_size=batch_size)

    # Number of steps to run
    num_steps = 10  # Adjust this as needed

    rewards = []
    sequences = []

    for step in range(num_steps):
        states = env.get_states()  # Now shape: (batch_size, state_dim)

        with torch.no_grad():
            actions = agent.select_actions(
                states, greedy=True
            )  # Shape: (batch_size, action_dim)
        print(f"actions: {torch.argmax(actions.to(dtype=torch.float), dim=1)}")
        print(f"states: {states}")

        next_states, reward = env.step(
            states, actions
        )  # next_states: (batch_size, state_dim), reward: (batch_size, 1)
        print(f"next_states: {next_states}")

        # Save rewards and sequences
        rewards.append(reward.tolist())  # Convert tensor to list for multiple rewards

        # Decode sequences from states
        _, seqs = parse_seqs_from_states(states)  # seqs: (batch_size, seq_len)
        seq_strings = [env.decode_seq(seq) for seq in seqs]  # Decode each sequence
        sequences.append(seq_strings)

    # Print amino acid sequences along the trajectory
    print("\nSequence Evolution:")
    for seq_idx in range(batch_size):
        print(f"\nSequence {seq_idx + 1}:")
        for step_idx, (step_seqs, step_rewards) in enumerate(zip(sequences, rewards)):
            print(
                f"  Step {step_idx + 1:<4d}: Sequence: {step_seqs[seq_idx]}, Reward: {step_rewards[seq_idx]}"
            )


if __name__ == "__main__":
    main()
