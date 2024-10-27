import wandb
from qres.config import config
from pprint import pprint
import torch.multiprocessing as mp
from threading import get_ident


class Logger:
    """
    Handles multithreading
    """

    def __init__(self):
        if config.wandb_enabled:
            self.wandb = wandb.init(
                project=config.project_name,
                config=config.__dict__,
            )
        self.attrs = {}
        self._lock = mp.Lock()

    def put(self, **kwargs):
        thread_id = get_ident()
        # with self._lock:
        self.attrs[thread_id] = self.attrs.get(thread_id, {})
        self.attrs[thread_id].update(kwargs)

    def push_attrs(self):
        thread_id = get_ident()
        self.log(**self.attrs[thread_id])
        # with self._lock:
        self.attrs[thread_id] = {}

    def log(self, **kwargs):
        if config.wandb_enabled:
            self.wandb.log(kwargs)
        else:
            pprint(kwargs)
    
    def log_str(self, s: str):
        if config.wandb_enabled:
            self.wandb.log({"log": s})
        else:
            print(s)

    def finish(self):
        if config.wandb_enabled:
            self.wandb.finish()


logger = Logger()
