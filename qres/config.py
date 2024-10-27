from dataclasses import dataclass
from pathlib import Path

import yaml

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
N_AMINO_ACIDS = len(AMINO_ACIDS)


@dataclass
class Config:
    sequence_length: int = 30

    structure_predictor_batch_size: int = int(1e3)

    max_buffer_size: int = int(1e7)
    n_epochs: int = int(5e3)

    train_iter: int = int(1e2)
    train_batch_size: int = int(1e4)

    @property
    def train_interval(self):
        return self.train_batch_size / self.structure_predictor_batch_size

    gamma: float = 0.99
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: int = 1000
    tau: float = 0.005
    lr: float = 1e-3
    distance_penalty_coeff: float = 5e-3

    save_interval: int = 100

    max_episode_length: int = 200

    project_name: str = "qres_stability"
    run_name: str = "4H"
    wandb_enabled: bool = True
    fake_structure_prediction: bool = False
    save_enabled: bool = True

    train_type: str = "multi-gpu"

    @property
    def state_dim(self):
        # Updated state dimension to include initial and current sequences
        return 2 * N_AMINO_ACIDS * self.sequence_length

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
