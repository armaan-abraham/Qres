import time

import datetime
import time
from qres.config import config

print(f"Local timezone: {datetime.datetime.now().astimezone().tzinfo}")

def print_time():
    current_time = time.localtime(time.time())
    current_time_with_ms = datetime.datetime.now()
    formatted_time = current_time_with_ms.strftime("%Y-%m-%d %I:%M:%S.%f")[:-4] + " " + time.strftime("%p", current_time)
    return formatted_time

class Logger:
    def __init__(self):
        self.attrs = {}
        self.tables = {}
        self.log_dir = None

    def log(self, **kwargs):
        message = {"Time": print_time(), **kwargs}
        self._log(message)

    def log_str(self, s: str):
        message = {"Msg": s, "Time": print_time()}
        self._log(message)

    def _log(self, message):
        print(message)

logger = Logger()
