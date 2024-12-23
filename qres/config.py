from dataclasses import dataclass
from multiprocessing import Manager
from pathlib import Path
from typing import Optional

import petname
import torch
import yaml

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
N_AMINO_ACIDS = len(AMINO_ACIDS)

save_dir = Path(__file__).parent / "data"
save_dir.mkdir(parents=True, exist_ok=True)


def generate_run_id():
    return petname.generate(words=3, separator="-")


def get_curr_save_dir(run_id: str):
    """Should only be called once!"""
    curr_save_dir = save_dir / f"run_{run_id}"
    if not curr_save_dir.exists():
        curr_save_dir.mkdir(parents=True, exist_ok=True)
    return curr_save_dir


@dataclass
class Config:
    # training objective
    seq_len: int = 20
    distance_penalty_coeff: float = 5e-4

    # training scale/duration
    n_epochs: int = int(1e4)
    train_iter: int = int(150)
    max_episode_length: int = 20
    buffer_size_rel_to_total_experience: float = 0.2
    epochs_per_eval: int = 20

    # batch size
    structure_predictor_batch_size: int = int(500)
    train_batch_size: int = int(2500)
    eval_batch_size: int = int(50)

    @property
    def total_train_samples(self):
        return self.n_epochs * self.train_batch_size

    @property
    def max_buffer_size(self):
        return int(self.total_train_samples * self.buffer_size_rel_to_total_experience)

    @property
    def train_interval(self):
        return self.train_batch_size / self.structure_predictor_batch_size

    @property
    def eval_interval(self):
        return self.epochs_per_eval * self.train_interval

    # save
    save_interval: int = 100
    save_enabled: bool = True

    wandb_enabled: bool = True

    # DQN
    gamma: float = 0.99
    epsilon_start: float = 0.95
    epsilon_end: float = 0.05
    epsilon_decay: int = int(2e4)
    tau: float = 0.1
    update_target_every: int = 10
    lr: float = 3e-4
    l2_weight_decay: float = 1e-4
    d_model: int = 128
    d_mlp: int = 256
    d_head: int = 32
    n_heads: int = 4
    n_layers: int = 2
    layer_norm_eps: float = 1e-5

    fake_structure_prediction: bool = False

    train_type: str = "multi-gpu"

    state_dtype: torch.dtype = torch.int8

    @property
    def state_dim(self):
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

    def __str__(self):
        # Get regular attributes
        attrs = {k: v for k, v in vars(self).items() if not k.startswith("__")}

        # Add properties
        for name in dir(self.__class__):
            if isinstance(getattr(self.__class__, name), property):
                attrs[name] = getattr(self, name)

        return "\n".join([f"{k}: {v}" for k, v in attrs.items()])


config = Config()
