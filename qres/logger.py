import time
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
        self.tables = {}
        self.step = None

    def log_to_table(self, **kwargs):
        kwargs["time"] = time.time()
        columns = sorted(list(kwargs.keys()))
        table = None

        # Create table name by concatenating sorted keys
        table_name = "_".join(columns)

        # Get existing table or create new one
        if table_name in self.tables:
            table = self.tables[table_name]
        else:
            # Create new table with sorted columns
            self.tables[table_name] = wandb.Table(columns=columns)
            table = self.tables[table_name]

        table.add_data(kwargs["time"], *[kwargs[col] for col in columns[1:]])

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
            wandb.log(kwargs, step=self.step)
        else:
            pprint(kwargs)

    def log_str(self, s: str):
        if config.wandb_enabled:
            self.log_to_table(Msg=s)
        else:
            print(s)


logger = Logger()
