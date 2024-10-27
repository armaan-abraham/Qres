import torch.multiprocessing as mp
from typing import List
from tqdm import tqdm
from qres.config import config
from enum import Enum
from qres.agent import Agent
from qres.environment import Environment
from qres.buffer import Buffer
from qres.logger import logger
import torch
from threading import Lock
"""
TODO: add logging

tasks:
1. choose actions / predict structure
    - holds individual environment instance
2. train agent

task queue:
1. inputs: agent
    - choose actions based on current states
    - predict structure
    - produce experiences
2. inputs: agent
    - add condition: certain number of prediction tasks have been completed
    since last train task

result queue:
1. outputs: experiences
    - add experiences to buffer
2. outputs: agent
    - update agent
"""


class TaskType(Enum):
    PREDICT_STRUCTURE = 0
    TRAIN_AGENT = 1
    ERROR = 2



class MultiTrainer:
    def __init__(self):
        self.device_ids = list(range(torch.cuda.device_count()))
        self.num_workers = len(self.device_ids)
        self.tasks = mp.Queue()
        self.results = mp.Queue()

        # Start worker processes
        self.workers = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=self._worker, args=(self.tasks, self.results, self.device_ids[i])
            )
            p.start()
            self.workers.append(p)
    
    def _worker(self, tasks: mp.Queue, results: mp.Queue, device_id: int):
        """
        Handles both structure prediction and training
        """
        device = f"cuda:{device_id}"
        env = Environment(device=device)

        try:
            while True:
                task = tasks.get()

                if task is None:
                    break

                if task == TaskType.PREDICT_STRUCTURE:
                    with self.agent_lock:
                        agent = self.agent.clone().to(device)
                    states = env.get_states()
                    with torch.no_grad():
                        actions = agent.select_actions(states)
                        next_states, rewards = env.step(states, actions)
                    results.put(
                        (
                            TaskType.PREDICT_STRUCTURE,
                            {
                                "states": states.clone().cpu(),
                                "actions": actions.clone().cpu(),
                                "next_states": next_states.clone().cpu(),
                                "rewards": rewards.clone().cpu(),
                            },
                        )
                    )
                    del agent, states, actions, next_states, rewards
                elif task == TaskType.TRAIN_AGENT:
                    with self.agent_lock:
                        agent = self.agent.clone().to(device)
                    with self.buffer_lock:
                        buffer = self.buffer.clone().to(device)
                    agent.train(buffer)
                    results.put((TaskType.TRAIN_AGENT, {"agent": agent.clone().cpu()}))
                    del agent, buffer
                else:
                    raise ValueError(f"Unknown task type: {task}")
                torch.cuda.empty_cache()
                logger.push_attrs()
        except Exception as e:
            results.put((TaskType.ERROR, {"error": e}))
            raise

    def run(self):
        self.agent = Agent(device="cpu")
        self.agent_lock = Lock()
        self.buffer = Buffer(device="cpu")
        self.buffer_lock = Lock()


        tasks_since_last_train = 0
        n_epochs = 0
        n_active_workers = 0
        with tqdm(total=config.n_epochs) as pbar:
            while n_epochs < config.n_epochs:
                # Choose task type and add to task queue
                if tasks_since_last_train >= config.train_interval:
                    self.tasks.put(TaskType.TRAIN_AGENT)
                    tasks_since_last_train = 0
                    n_epochs += 1
                    pbar.update(1)
                else:
                    self.tasks.put(TaskType.PREDICT_STRUCTURE)
                    tasks_since_last_train += 1
                n_active_workers += 1

                if n_active_workers == self.num_workers:
                    # Take result from queue
                    task_type, result = self.results.get()
                    if task_type == TaskType.ERROR:
                        raise result["error"]
                    elif task_type == TaskType.TRAIN_AGENT:
                        with self.agent_lock:
                            self.agent = result["agent"]
                    elif task_type == TaskType.PREDICT_STRUCTURE:
                        self.buffer.add(
                            result["states"],
                            result["actions"],
                            result["next_states"],
                            result["rewards"],
                        )
                    n_active_workers -= 1

    def shutdown(self):
        for _ in self.workers:
            self.tasks.put(None)
        for p in self.workers:
            p.join()
