import time

import wandb
from qres.config import config


class Logger:
    """
    Handles multithreading
    """

    def __init__(self):
        self.attrs = {}
        self.tables = {}
        self.step = None

    def log_attrs(self, **kwargs):
        if config.wandb_enabled:
            wandb.log(kwargs, step=self.step)
        else:
            self.log(**kwargs)

    def log(self, **kwargs):
        message = {"time": time.time(), **kwargs}
        print(message)

    def log_str(self, s: str):
        message = {"Msg": s, "time": time.time()}
        print(message)

logger = Logger()
