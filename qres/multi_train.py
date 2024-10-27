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
from copy import deepcopy

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

multi buffer:
- buffer in shared memory, passed to each worker on worker init
- when worker gets structure prediction task, it moves the agent to the local
device, and then adds to the buffer while holding the lock
- when worker gets train task, it moves the agent and buffer to local device,
trains agent, and then copies agent back to global memory
"""


# TODO: steps done in agent to shared memory


class TaskType(Enum):
    PREDICT_STRUCTURE = 0
    TRAIN_AGENT = 1
    ERROR = 2

def worker(
    tasks: mp.Queue,
    results: mp.Queue,
    buffer: Buffer,
    buffer_lock,
    device_id: int,
):
    """
    Handles both structure prediction and training
    """
    try:
        device = f"cuda"
        env = Environment(device=device)
        logger_kwargs = {"Worker": device_id}
    except Exception as e:
        results.put((TaskType.ERROR, {"error": e}))
        raise

    try:
        while True:
            task, agent = tasks.get()

            if task is None:
                break

            logger_kwargs_local = deepcopy(logger_kwargs)
            if task == TaskType.PREDICT_STRUCTURE:
                logger_kwargs_local["Task"] = "predict_structure"
                logger_kwargs_local["Msg"] = "started"
                logger.log(**logger_kwargs_local)


                states = env.get_states()
                logger_kwargs_local["Msg"] = "got states"
                logger.log(**logger_kwargs_local)

                assert agent.device == "cpu"
                agent = agent.to(device)

                # Select actions and step environment
                with torch.no_grad():
                    actions = agent.select_actions(states)
                    next_states, rewards = env.step(states, actions)

                logger_kwargs_local["Msg"] = "selected actions"
                logger.log(**logger_kwargs_local)

                # Add experiences to buffer
                with buffer_lock:
                    buffer_local = buffer.clone().to(device)
                    buffer_local.add(
                        states,
                        actions,
                        next_states,
                        rewards,
                    )
                    buffer.copy_(buffer_local.to("cpu"))

                logger_kwargs_local["Msg"] = "added experiences to buffer"
                logger.log(**logger_kwargs_local)

                results.put((TaskType.PREDICT_STRUCTURE, None))
                del agent, buffer_local, states, actions, next_states, rewards, logger_kwargs_local

            elif task == TaskType.TRAIN_AGENT:
                logger_kwargs_local["Task"] = "train_agent"
                logger_kwargs_local["Msg"] = "started"
                logger.log(**logger_kwargs_local)

                assert agent.device == "cpu"
                agent = agent.to(device)

                logger_kwargs_local["Msg"] = "cloned agent"
                logger.log(**logger_kwargs_local)

                with buffer_lock:
                    assert buffer.device == "cpu"
                    buffer_local = buffer.clone().to(device)

                logger_kwargs_local["Msg"] = "cloned buffer"
                logger.log(**logger_kwargs_local)

                agent.train(buffer_local)

                logger_kwargs_local["Msg"] = "trained agent"
                logger.log(**logger_kwargs_local)

                logger_kwargs_local["Msg"] = "updated global agent"
                logger.log(**logger_kwargs_local)

                results.put((TaskType.TRAIN_AGENT, agent.cpu()))
                del agent, buffer_local, logger_kwargs_local
            else:
                raise ValueError(f"Unknown task type: {task}")
            torch.cuda.empty_cache()
            logger.push_attrs()
    except Exception as e:
        logger_kwargs_local["Task"] = "error"
        logger_kwargs_local["Msg"] = str(e)
        logger.log(**logger_kwargs_local)
        results.put((TaskType.ERROR, {"error": e}))
        raise

class MultiTrainer:
    def __init__(self):
        self.device_ids = list(range(torch.cuda.device_count()))
        self.num_workers = len(self.device_ids)

    def run(self):
        mp.set_start_method("spawn", force=True)
        tasks = mp.Queue()
        results = mp.Queue()

        agent = Agent(device="cpu")

        buffer = Buffer(device="cpu")
        buffer.share_memory()
        buffer_lock = mp.Lock()

        workers = []
        for i in range(self.num_workers):
            logger.log_str(f"Starting worker {i}")
            p = mp.Process(
                target=worker,
                args=(
                    tasks,
                    results,
                    buffer,
                    buffer_lock,
                    self.device_ids[i],
                ),
            )
            p.start()
            workers.append(p)

        tasks_since_last_train = 0
        # We only want one training task to be in progress at a time
        prev_train_completed = True
        n_epochs = 0
        n_active_workers = 0
        with tqdm(total=config.n_epochs) as pbar:
            while n_epochs < config.n_epochs:
                assert n_active_workers <= self.num_workers and n_active_workers >= 0
                assert (
                    tasks_since_last_train >= 0
                    and tasks_since_last_train <= config.train_interval
                )

                # Choose task type and add to task queue
                if (
                    tasks_since_last_train >= config.train_interval
                    and prev_train_completed
                ):
                    logger.log_str("Adding train agent task to queue")
                    self.tasks.put((TaskType.TRAIN_AGENT, agent))
                    prev_train_completed = False
                    tasks_since_last_train = 0
                    pbar.update(1)
                    logger.log(Epoch=n_epochs)
                else:
                    logger.log_str("Adding predict structure task to queue")
                    self.tasks.put((TaskType.PREDICT_STRUCTURE, agent))
                    tasks_since_last_train += 1
                n_active_workers += 1

                if n_active_workers == self.num_workers:
                    # Take result from queue
                    logger.log_str(f"Waiting for result from queue")
                    task_type, result = self.results.get()
                    if task_type == TaskType.ERROR:
                        raise result["error"]
                    elif task_type == TaskType.TRAIN_AGENT:
                        assert result["agent"].device == "cpu"
                        agent = result["agent"]
                        n_epochs += 1
                        prev_train_completed = True
                        logger.log_str(f"Train agent task completed")
                    elif task_type == TaskType.PREDICT_STRUCTURE:
                        logger.log_str(f"Predict structure task completed")
                    n_active_workers -= 1

    def shutdown(self):
        for _ in self.workers:
            self.tasks.put(None)
        for p in self.workers:
            p.join()
