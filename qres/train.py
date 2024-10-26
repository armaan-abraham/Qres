import wandb
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

from qres.config import Config
from qres.environment import ProteinEnv
from qres.agent import Agent
from qres.data_handler import DataHandler

def train():
    # Initialize config
    config = Config()
    
    # Initialize wandb
    wandb.init(project="protein-folding", config=config.__dict__)
    
    # Initialize components
    env = ProteinEnv(config)
    agent = Agent(config)
    data_handler = DataHandler(config)
    optimizer = optim.AdamW(agent.policy_net.parameters(), lr=config.learning_rate)
    
    try:
        for episode in tqdm(range(config.num_episodes)):
            state = env.reset()
            total_reward = 0
            
            for t in range(1000):  # Max steps per episode
                action = agent.select_action(state)
                next_state, reward, done = env.step(state, action.item())
                total_reward += reward
                
                # Store transition
                data_handler.push(state, action, next_state, reward)
                
                # Move to next state
                state = next_state
                
                # Perform optimization
                if len(data_handler) >= config.batch_size:
                    optimize_model(agent, data_handler, optimizer, config)
                
                if done:
                    break
            
            # Log metrics
            wandb.log({
                "episode": episode,
                "total_reward": total_reward,
                "epsilon": agent.eps_threshold
            })
            
            # Save checkpoint
            if episode % config.save_interval == 0:
                data_handler.save_model(agent.policy_net, optimizer, episode)
                
    finally:
        # Save final model and transitions
        data_handler.save_model(agent.policy_net, optimizer, "final")
        data_handler.save_transitions()
        wandb.finish()

def optimize_model(agent, data_handler, optimizer, config):
    if len(data_handler) < config.batch_size:
        return
    
    transitions = data_handler.sample(config.batch_size)
    batch = Transition(*zip(*transitions))
    
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.stack(batch.next_state)
    
    state_action_values = agent.policy_net(state_batch).gather(1, action_batch)
    
    with torch.no_grad():
        next_state_values = agent.target_net(next_state_batch).max(1)[0]
    
    expected_state_action_values = (next_state_values * config.gamma) + reward_batch
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100)
    optimizer.step()

if __name__ == "__main__":
    train()
