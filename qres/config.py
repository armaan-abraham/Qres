from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
N_AMINO_ACIDS = len(AMINO_ACIDS)


@dataclass
class Config:
    sequence_length: int = 30

    structure_predictor_batch_size: int = int(500)

    max_buffer_size: int = int(1e7)
    n_epochs: int = int(5e3)

    train_iter: int = int(100)
    train_batch_size: int = int(1500)

    @property
    def train_interval(self):
        return self.train_batch_size / self.structure_predictor_batch_size

    gamma: float = 0.99
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: int = 1000
    tau: float = 0.005
    lr: float = 1e-3
    distance_penalty_coeff: float = 1e-2

    save_interval: int = 50

    max_episode_length: int = 200

    project_name: str = "qres_stability"
    run_name: str = "4H"
    wandb_enabled: bool = True
    fake_structure_prediction: bool = False
    save_enabled: bool = True

    train_type: str = "multi-gpu"

    state_dtype: torch.dtype = torch.int8  # Updated state data type

    @property
    def state_dim(self):
        # Updated state dimension to account for indices instead of one-hot encoding
        return 2 * self.sequence_length

    @property
    def action_dim(self):
        return N_AMINO_ACIDS * self.sequence_length

    def save(self, path: Path):
        # Get all non-property attributes that aren't builtins
        attrs = {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("__")
            and not isinstance(getattr(type(self), k, None), property)
        }

        # Handle torch.dtype specially
        attrs["state_dtype"] = str(self.state_dtype)

        with open(path, "w") as f:
            yaml.dump(attrs, f)

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert state_dtype back to torch.dtype
        if "state_dtype" in config_dict:
            dtype_str = config_dict["state_dtype"]
            config_dict["state_dtype"] = getattr(torch, dtype_str.split(".")[-1])

        return cls(**config_dict)


config = Config()
