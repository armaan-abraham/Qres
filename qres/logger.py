import wandb
from qres.config import config
from pprint import pprint


class Logger:
    def __init__(self):
        if config.wandb_enabled:
            self.wandb = wandb.init(
                project=config.project_name,
                config=config.__dict__,
            )
        self.attrs = {}

    def put(self, **kwargs):
        self.attrs.update(kwargs)

    def log(self):
        if config.wandb_enabled:
            self.wandb.log(self.attrs)
        else:
            pprint(self.attrs)
        self.attrs = {}

    def finish(self):
        if config.wandb_enabled:
            self.wandb.finish()


logger = Logger()
