from copy import deepcopy
from enum import Enum

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from qres.agent import Agent
from qres.buffer import Buffer
from qres.config import config
from qres.environment import Environment
from qres.logger import logger
from pathlib import Path
import wandb

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
    wandb_run_id: str = None,  # Add wandb run ID parameter
):
    """
    Handles both structure prediction and training
    """
    device = f"cuda:{device_id}"
    try:
        # Initialize wandb in worker process if enabled
        if config.wandb_enabled:
            wandb.init(
                project=config.project_name,
                id=wandb_run_id,  # Use the same run ID
                resume="allow",
                name=config.run_name,
            )
        env = Environment(device=device)
        logger_kwargs = {"Worker": device_id}
    except Exception as e:
        results.put((TaskType.ERROR, {"error": e}))
        raise

    try:
        while True:
            task = tasks.get()

            logger_kwargs_local = deepcopy(logger_kwargs)
            logger_kwargs_local["MemoryMB"] = torch.cuda.memory_allocated(device) / 1e6
            logger.log(**logger_kwargs_local)

            if task is None:
                break

            task_type = task["type"]
            agent = task["agent"]
            task_id = task["task_id"]
            logger_kwargs_local["TaskID"] = task_id

            if task_type == TaskType.PREDICT_STRUCTURE:
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
    
                logger_kwargs_local["Msg"] = "stepped environment"
                logger.log(**logger_kwargs_local)

                # Add experiences to buffer
                with buffer_lock:
                    buffer.add(
                        states.clone().to("cpu"),
                        actions.clone().to("cpu"),
                        next_states.clone().to("cpu"),
                        rewards.clone().to("cpu"),
                    )

                logger_kwargs_local["Msg"] = "added experiences to buffer"
                logger.log(**logger_kwargs_local)

                results.put((TaskType.PREDICT_STRUCTURE, {}))
                del (
                    agent,
                    states,
                    actions,
                    next_states,
                    rewards,
                    logger_kwargs_local,
                )

            elif task_type == TaskType.TRAIN_AGENT:
                logger_kwargs_local["Task"] = "train_agent"
                logger_kwargs_local["Msg"] = "started"
                logger.log(**logger_kwargs_local)

                assert agent.device == "cpu"
                agent = agent.to(device)

                with buffer_lock:
                    logger_kwargs_local = deepcopy(logger_kwargs)
                    logger_kwargs_local["MemoryMB"] = torch.cuda.memory_allocated(device) / 1e6
                    logger.log(**logger_kwargs_local)
                    
                    assert buffer.device == "cpu"
                    buffer_local = buffer.to(device)
                    assert buffer.device == "cpu"
                    assert buffer_local.device == device

                agent.train(buffer_local)

                logger_kwargs_local["Msg"] = "trained agent"
                logger.log(**logger_kwargs_local)

                results.put((TaskType.TRAIN_AGENT, {"agent": agent.to("cpu")}))
                del agent, buffer_local, logger_kwargs_local
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            torch.cuda.empty_cache()
            logger.push_attrs()
    except Exception as e:
        logger_kwargs_local["Task"] = "error"
        logger_kwargs_local["Msg"] = str(e)
        logger.log(**logger_kwargs_local)
        results.put((TaskType.ERROR, {"error": e}))
        raise


class MultiTrainer:
    def __init__(self, save_dir: Path = None):
        self.save_dir = save_dir
        self.device_ids = list(range(torch.cuda.device_count()))
        self.num_workers = len(self.device_ids)
        self.wandb_run_id = wandb.run.id if config.wandb_enabled else None  # Store wandb run ID

    def run(self):
        mp.set_start_method("spawn", force=True)
        tasks = mp.Queue()
        results = mp.Queue()

        agent = Agent(device="cpu")

        buffer = Buffer(device="cpu")
        buffer.share_memory()
        buffer.print_memory_usage()
        buffer_lock = mp.Lock()

        try:
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
                        self.wandb_run_id,  # Pass wandb run ID to workers
                    ),
                )
                p.start()
                workers.append(p)

            tasks_since_last_train = 0
            # We only want one training task to be in progress at a time
            prev_train_completed = True
            n_epochs = 0
            n_active_workers = 0
            task_id = 0
            with tqdm(total=config.n_epochs) as pbar:
                while n_epochs < config.n_epochs:
                    assert (
                        n_active_workers <= self.num_workers and n_active_workers >= 0
                    )
                    assert (
                        tasks_since_last_train >= 0
                        # and tasks_since_last_train <= config.train_interval
                    )

                    # Choose task type and add to task queue
                    if (
                        tasks_since_last_train >= config.train_interval
                        and prev_train_completed and buffer.get_size() >= config.train_batch_size
                    ):
                        logger.log_str("Adding train agent task to queue")
                        tasks.put(({"type": TaskType.TRAIN_AGENT, "agent": agent, "task_id": task_id}))
                        prev_train_completed = False
                        tasks_since_last_train = 0
                        pbar.update(1)
                        logger.log(Epoch=n_epochs)
                    else:
                        logger.log_str("Adding predict structure task to queue")
                        tasks.put(({"type": TaskType.PREDICT_STRUCTURE, "agent": agent, "task_id": task_id}))
                        tasks_since_last_train += 1
                    task_id += 1
                    n_active_workers += 1

                    if n_active_workers == self.num_workers:
                        # Take result from queue
                        logger.log_str("Waiting for result from queue")
                        task_type, result = results.get()
                        if task_type == TaskType.ERROR:
                            raise result["error"]
                        elif task_type == TaskType.TRAIN_AGENT:
                            assert result["agent"].device == "cpu", result[
                                "agent"
                            ].device
                            agent.copy_(result["agent"])
                            n_epochs += 1
                            prev_train_completed = True
                            logger.log_str("Train agent task completed")

                            if config.save_enabled and (n_epochs+1) % config.save_interval == 0:
                                self.save(agent, buffer, n_epochs)

                        elif task_type == TaskType.PREDICT_STRUCTURE:
                            logger.log_str("Predict structure task completed")
                            logger.log(BufferSize=buffer.get_size())
                        n_active_workers -= 1

        finally:
            for _ in workers:
                tasks.put(None)
            for p in workers:
                p.join()


    def save(self, agent: Agent, buffer: Buffer, n_epochs: int):
        agent.save_model(self.save_dir / f"model_{n_epochs}.pt")
        buffer.save(self.save_dir / f"buffer_{n_epochs}.pth")
        config.save(self.save_dir / f"config_{n_epochs}.yaml")

        if config.wandb_enabled:
            wandb.save(self.save_dir / f"model_{n_epochs}.pt")
            wandb.save(self.save_dir / f"buffer_{n_epochs}.pth")
            wandb.save(self.save_dir / f"config_{n_epochs}.yaml")
