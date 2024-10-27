import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from qres.agent import Agent
from qres.buffer import Buffer
from qres.config import config
from qres.environment import Environment
from qres.logger import logger


class SingleTrainer:
    def __init__(self, save_dir: Path = None):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        # Set number of GPUs to use for structure prediction
        self.device = "cuda"

        self.agent = Agent(device=self.device)
        self.env = Environment(device=self.device)
        self.buffer = Buffer(device=self.device)

    def collect_experience(self):
        states = self.env.get_states()

        # Select batch of actions from the agent
        actions = self.agent.select_actions(states)  # Shape: [batch_size, action_dim]

        # Apply actions in the environment
        next_states, rewards = self.env.step(states, actions)  # Shapes: [batch_size, *]

        # Store transitions in the buffer
        self.buffer.add(states, actions, next_states, rewards)
        logger.put(BufferSize=self.buffer.size)

    def run(self):
        for epoch in tqdm(range(config.n_epochs)):
            self.collect_experience()
            self.agent.train(self.buffer)

            logger.put(Epoch=epoch, Epsilon=self.agent.get_epsilon())
            logger.push_attrs()
