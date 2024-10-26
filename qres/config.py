from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    # Model parameters
    sequence_length: int = 30
    hidden_dim: int = 256
    
    # Training parameters
    batch_size: int = 256
    gamma: float = 0.99
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: int = 1000
    tau: float = 0.005
    learning_rate: float = 1e-4
    num_episodes: int = 1500
    
    # Save/logging parameters
    save_interval: int = 1000
    checkpoint_dir: Path = Path("checkpoints")
    data_dir: Path = Path("data")
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, path: Path):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)
