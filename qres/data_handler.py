from collections import deque, namedtuple
import random
import torch
from pathlib import Path
import pickle

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DataHandler:
    def __init__(self, config, capacity=10000):
        self.config = config
        self.memory = deque([], maxlen=capacity)
        
        # Create directories if they don't exist
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        self.config.data_dir.mkdir(exist_ok=True)
        
    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    def save_model(self, model, optimizer, episode):
        """Save model checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_{episode}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode
        }, checkpoint_path)
        
    def save_transitions(self):
        """Save all transitions to disk."""
        transition_path = self.config.data_dir / "transitions.pkl"
        with open(transition_path, "wb") as f:
            pickle.dump(list(self.memory), f)
    
    def __len__(self):
        return len(self.memory)
