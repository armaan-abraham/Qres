import torch
from qres.buffer import Buffer

def print_buffer_stats(buffer_path: str):
    # Load the saved buffer data
    buffer_data = torch.load(buffer_path)
    
    # Create a new buffer instance and set its device to CPU
    buffer = Buffer(device="cpu")
    print(f"buffer: {buffer.size}")
    
    # Copy the saved data to the buffer
    buffer.states = buffer_data["states"]
    buffer.actions = buffer_data["actions"]
    buffer.next_states = buffer_data["next_states"]
    buffer.rewards = buffer_data["rewards"]
    buffer.set_pos(buffer_data["pos"])
    buffer.set_size(buffer_data["size"])
    
    # Calculate and print mean reward
    valid_rewards = buffer.rewards[:buffer.get_size()]
    mean_reward = valid_rewards.mean().item()
    print(f"Buffer size: {buffer.get_size()}")
    print(f"Mean reward: {mean_reward:.4f}")

if __name__ == "__main__":
    print_buffer_stats("/root/Qres/qres/data/run_truly-picked-snipe/buffer_3599.pth")
