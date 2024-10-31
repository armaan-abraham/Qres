from enum import Enum
from pathlib import Path

import torch
import traceback
import torch.multiprocessing as mp

from qres.agent import Agent
from qres.buffer import Buffer
from qres.config import config
from qres.environment import Environment
from qres.logger import logger

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
    device = f"cuda:{device_id}"
    try:
        env = Environment(device=device)
    except Exception as e:
        results.put((TaskType.ERROR, {"error": e}))
        raise

    try:
        while True:
            task = tasks.get()


            if task is None:
                break

            task_type = task["type"]
            agent = task["agent"]

            if task_type == TaskType.PREDICT_STRUCTURE:

                states = env.get_states()

                assert agent.device == "cpu"
                agent = agent.to(device)

                # Select actions and step environment
                with torch.no_grad():
                    actions = agent.select_actions(states)
                    next_states, rewards = env.step(states, actions)


                # Add experiences to buffer
                with buffer_lock:
                    buffer.add(
                        states.clone().to("cpu"),
                        actions.clone().to("cpu"),
                        next_states.clone().to("cpu"),
                        rewards.clone().to("cpu"),
                    )

                results.put(
                    (
                        TaskType.PREDICT_STRUCTURE,
                        {
                            "reward": env.last_reward,
                            "confidence": env.last_confidence,
                            "distance_penalty": env.last_distance_penalty,
                            "device_id": device_id,
                            "task_id": task["task_id"],
                            "gpu_memory_allocated": torch.cuda.memory_allocated(device) / 1e9,
                            "gpu_memory_reserved": torch.cuda.memory_reserved(device) / 1e9,
                        },
                    )
                )
                del (
                    agent,
                    states,
                    actions,
                    next_states,
                    rewards,
                )
            elif task_type == TaskType.TRAIN_AGENT:
                assert agent.device == "cpu"

                agent = agent.to(device)

                with buffer_lock:

                    assert buffer.device == "cpu"
                    buffer_local = buffer.to(device)
                    assert buffer.device == "cpu"
                    assert buffer_local.device == device

                agent.train(buffer_local)

                results.put(
                    (
                        TaskType.TRAIN_AGENT,
                        {
                            "agent": agent.to("cpu"),
                            "device_id": device_id,
                            "task_id": task["task_id"],
                            "gpu_memory_allocated": torch.cuda.memory_allocated(device) / 1e9,
                            "gpu_memory_reserved": torch.cuda.memory_reserved(device) / 1e9,
                        },
                    )
                )
                del agent, buffer_local
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            torch.cuda.empty_cache()
    except Exception as e:
        error_msg = traceback.format_exc()
        results.put((TaskType.ERROR, {"error": error_msg}))
        raise


class MultiTrainer:
    def __init__(self, save_dir: Path = None):
        self.save_dir = save_dir
        self.device_ids = list(range(torch.cuda.device_count()))
        self.num_workers = len(self.device_ids)

    def run(self):
        mp.set_start_method("spawn", force=True)
        tasks = mp.Queue()
        results = mp.Queue()

        self.agent = Agent(device="cpu")

        self.buffer = Buffer(device="cpu")
        self.buffer.share_memory()
        self.buffer.print_memory_usage()
        buffer_lock = mp.Lock()

        workers = []
        for i in range(self.num_workers):
            logger.log_str(f"Starting worker {i}")
            p = mp.Process(
                target=worker,
                args=(
                    tasks,
                    results,
                    self.buffer,
                    buffer_lock,
                    i,
                ),
            )
            p.start()
            workers.append(p)


        try:
            tasks_since_last_train = 0
            # We only want one training task to be in progress at a time
            prev_train_completed = True
            n_active_workers = 0
            task_id = 0
            epoch = 0
            while epoch < config.n_epochs:
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
                    and prev_train_completed
                    and self.buffer.get_size() >= config.train_batch_size
                ):
                    logger.log(
                        Msg="Task created",
                        TaskType="Train",
                        TaskID=task_id,
                    )
                    tasks.put(
                        (
                            {
                                "type": TaskType.TRAIN_AGENT,
                                "agent": self.agent,
                                "task_id": task_id,
                            }
                        )
                    )
                    prev_train_completed = False
                    tasks_since_last_train = 0
                else:
                    logger.log(
                        Msg="Task created",
                        TaskType="Structure",
                        TaskID=task_id,
                    )
                    tasks.put(
                        (
                            {
                                "type": TaskType.PREDICT_STRUCTURE,
                                "agent": self.agent,
                                "task_id": task_id,

                            }
                        )
                    )
                    tasks_since_last_train += 1
                task_id += 1
                logger.step = task_id
                n_active_workers += 1

                if n_active_workers == self.num_workers:
                    # Take result from queue
                    logger.log_str("Waiting for result from queue")
                    task_type, result = results.get()

                    if task_type == TaskType.ERROR:
                        logger.log(Error=result["error"])
                        raise result["error"]

                    elif task_type == TaskType.TRAIN_AGENT:
                        assert result["agent"].device == "cpu", result[
                            "agent"
                        ].device
                        self.agent.copy_(result["agent"])
                        epoch += 1
                        prev_train_completed = True

                        logger.log(
                            Msg="Task completed",
                            TaskType="Train",
                            Epoch=epoch,
                            DeviceID=result["device_id"],
                            TaskID=result["task_id"],
                            GpuMemoryAllocated=result["gpu_memory_allocated"],
                            GpuMemoryReserved=result["gpu_memory_reserved"],
                        )

                        if (
                            config.save_enabled
                            and (epoch + 1) % config.save_interval == 0
                        ):
                            self.save(epoch)

                    elif task_type == TaskType.PREDICT_STRUCTURE:
                        logger.log(
                            Msg="Task completed",
                            TaskType="Structure",
                            TaskID=result["task_id"],
                            DeviceID=result["device_id"],
                            BufferSize=self.buffer.get_size(),
                            Reward=result["reward"],
                            Confidence=result["confidence"],
                            DistancePenalty=result["distance_penalty"],
                            GpuMemoryAllocated=result["gpu_memory_allocated"],
                            GpuMemoryReserved=result["gpu_memory_reserved"],
                        )

                    n_active_workers -= 1

                # Print GPU memory usage for each device
                for device_id in self.device_ids:
                    allocated = torch.cuda.memory_allocated(device_id) / 1e9  # Convert to GB
                    reserved = torch.cuda.memory_reserved(device_id) / 1e9    # Convert to GB
                    logger.log(
                        Msg="GPU Memory Usage",
                        DeviceID=device_id,
                        AllocatedGB=allocated,
                        ReservedGB=reserved
                    )

        finally:
            for _ in workers:
                tasks.put(None)
            for p in workers:
                p.join()

    def save(self, n_epochs: int | None = None):
        if n_epochs is None:
            n_epochs = "final"
        logger.log(
            Msg="Saving model and buffer",
            Epoch=n_epochs,
            SaveDir=self.save_dir,
        )
        self.agent.save_model(self.save_dir / f"model_{n_epochs}.pt")
        self.buffer.save(self.save_dir / f"buffer_{n_epochs}.pth")
