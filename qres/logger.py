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

    def push_attrs(self):
        self.log(**self.attrs)
        self.attrs = {}

    def log(self, **kwargs):
        if config.wandb_enabled:
            self.wandb.log(kwargs)
        else:
            pprint(kwargs)

    def finish(self):
        if config.wandb_enabled:
            self.wandb.finish()


logger = Logger()
