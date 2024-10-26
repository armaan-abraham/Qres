import torch
import random
import numpy as np
from qres.config import config
from qres.agent import Agent
from qres.environment import Environment
from qres.buffer import Buffer
from tqdm import tqdm
from qres.logger import logger
from pathlib import Path

save_dir = Path(__file__).parent / "data"
save_dir.mkdir(parents=True, exist_ok=True)


def get_curr_save_dir():
    iterations = [int(f.stem.split("_")[-1]) for f in save_dir.glob("*/")]
    curr_save_dir = save_dir / f"run_{max(iterations, default=0) + 1}"
    return curr_save_dir


def collect_experience(agent: Agent, env: Environment, buffer: Buffer):
    states = env.get_states()

    # Select batch of actions from the agent
    actions = agent.select_actions(states)  # Shape: [batch_size, action_dim]

    # Apply actions in the environment
    next_states, rewards = env.step(states, actions)  # Shapes: [batch_size, *]
    logger.put(Reward=rewards.mean().item())

    # Store transitions in the buffer
    buffer.add(states, actions, next_states, rewards)
    logger.put(BufferSize=buffer.size)


def train():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    curr_save_dir = get_curr_save_dir()

    agent = Agent()
    env = Environment()
    buffer = Buffer()

    try:
        for epoch in tqdm(range(config.n_epochs)):
            collect_experience(agent, env, buffer)

            agent.train(buffer)

            logger.put(Epoch=epoch, Epsilon=agent.get_epsilon())
            logger.log()

    finally:
        print(f"Saving to {curr_save_dir}")
        if not curr_save_dir.exists():
            curr_save_dir.mkdir(parents=True, exist_ok=True)
        agent.save_model(curr_save_dir / "model.pt")
        buffer.save(curr_save_dir / "buffer.pth")
        config.save(curr_save_dir / "config.yaml")
        logger.finish()


if __name__ == "__main__":
    train()
