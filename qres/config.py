from dataclasses import dataclass, field
from pathlib import Path
import yaml
from qres.structure_prediction import N_AMINO_ACIDS


@dataclass
class Config:
    sequence_length: int = 30
    batch_size: int = 256
    max_buffer_size: int = int(1e6)
    train_iterations_per_batch: int = 20
    gamma: float = 0.99
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: int = 1000
    tau: float = 0.005
    lr: float = 1e-3
    save_interval: int = 100
    n_epochs: int = int(1e4)
    max_episode_length: int = 200
    device: str = "cuda"
    project_name: str = "qres_stability"
    wandb_enabled: bool = True
    fake_structure_prediction: bool = False

    @property
    def state_dim(self):
        return N_AMINO_ACIDS * self.sequence_length

    @property
    def action_dim(self):
        return N_AMINO_ACIDS * self.sequence_length

    def save(self, path: Path):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


config = Config()
