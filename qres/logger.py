from pprint import pprint
from threading import get_ident

import wandb
from qres.config import config


class Logger:
    """
    Handles multithreading
    """

    def __init__(self):
        self.attrs = {}

    def put(self, **kwargs):
        thread_id = get_ident()
        self.attrs[thread_id] = self.attrs.get(thread_id, {})
        self.attrs[thread_id].update(kwargs)

    def push_attrs(self):
        thread_id = get_ident()
        self.log(**self.attrs[thread_id])
        self.attrs[thread_id] = {}

    def log(self, **kwargs):
        if config.wandb_enabled:
            wandb.log(kwargs)
        else:
            pprint(kwargs)

    def log_str(self, s: str):
        if config.wandb_enabled:
            wandb.log({"log": s})
        else:
            print(s)


logger = Logger()
