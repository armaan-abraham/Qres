from dataclasses import dataclass
from multiprocessing import Manager
from pathlib import Path
from typing import Optional

import torch
import yaml

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
N_AMINO_ACIDS = len(AMINO_ACIDS)

save_dir = Path(__file__).parent / "data"
save_dir.mkdir(parents=True, exist_ok=True)


def get_curr_run_iter():
    iterations = [int(f.stem.split("_")[-1]) for f in save_dir.glob("*/")]
    return max(iterations, default=0) + 1


def get_curr_save_dir():
    curr_save_dir = save_dir / f"run_{get_curr_run_iter()}"
    if not curr_save_dir.exists():
        curr_save_dir.mkdir(parents=True, exist_ok=True)
    return curr_save_dir


@dataclass
class Config:
    # training objective
    seq_len: int = 30
    distance_penalty_coeff: float = 5e-4

    # training scale/duration
    max_buffer_size: int = int(5e6)
    n_epochs: int = int(1e3)
    train_iter: int = int(300)
    max_episode_length: int = 50

    # batch size
    structure_predictor_batch_size: int = int(500)
    train_batch_size: int = int(5000)

    @property
    def train_interval(self):
        return self.train_batch_size / self.structure_predictor_batch_size

    # DQN
    gamma: float = 0.99
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: int = 1000
    tau: float = 0.1
    update_target_every: int = 10
    lr: float = 3e-4
    l2_weight_decay: float = 1e-4
    d_model: int = 128
    d_mlp: int = 256
    d_head: int = 32
    n_heads: int = 4
    n_layers: int = 3
    layer_norm_eps: float = 1e-5

    # save
    save_interval: int = 100
    save_enabled: bool = True

    wandb_enabled: bool = True

    fake_structure_prediction: bool = False

    train_type: str = "multi-gpu"

    state_dtype: torch.dtype = torch.int8

    @property
    def state_dim(self):
        # Updated state dimension to account for indices instead of one-hot encoding
        return 2 * self.seq_len

    @property
    def action_dim(self):
        return N_AMINO_ACIDS * self.seq_len

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
