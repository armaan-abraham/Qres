import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class DQN(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.sequence_length * len(AMINO_ACIDS)
        output_dim = config.sequence_length * len(AMINO_ACIDS)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(config).to(self.device)
        self.target_net = DQN(config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps_done = 0
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * \
            math.exp(-1. * self.steps_done / self.config.eps_decay)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.config.sequence_length * len(AMINO_ACIDS))]], 
                              device=self.device, dtype=torch.long)
