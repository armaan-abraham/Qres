import logging
import os
import traceback
from enum import Enum
from pathlib import Path

import torch
import torch.multiprocessing as mp

import wandb
from qres.agent import Agent
from qres.buffer import Buffer
from qres.config import config
from qres.environment import Environment

logger = logging.getLogger()


class TaskType(Enum):
    PREDICT_STRUCTURE = 0
    TRAIN_AGENT = 1
    ERROR = 2
    EVAL = 3


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

                # For logging env reset
                env_reset = 0
                if torch.any(env.steps_done == 0):
                    assert torch.all(
                        env.steps_done == 0
                    ), f"Steps done: {env.steps_done}"
                    env_reset = 1

                results.put(
                    (
                        TaskType.PREDICT_STRUCTURE,
                        {
                            "reward": env.last_reward,
                            "confidence": env.last_confidence,
                            "distance_penalty": env.last_distance_penalty,
                            "device_id": device_id,
                            "task_id": task["task_id"],
                            "gpu_memory_allocated": torch.cuda.memory_allocated(device)
                            / 1e9,
                            "gpu_memory_reserved": torch.cuda.memory_reserved(device)
                            / 1e9,
                            "env_reset": env_reset,
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

                avg_loss, avg_reward, avg_state_action_value = agent.train(buffer_local)

                results.put(
                    (
                        TaskType.TRAIN_AGENT,
                        {
                            "avg_loss": avg_loss,
                            "avg_reward": avg_reward,
                            "avg_state_action_value": avg_state_action_value,
                            "agent": agent.to("cpu"),
                            "device_id": device_id,
                            "task_id": task["task_id"],
                            "gpu_memory_allocated": torch.cuda.memory_allocated(device)
                            / 1e9,
                            "gpu_memory_reserved": torch.cuda.memory_reserved(device)
                            / 1e9,
                        },
                    )
                )
                del agent, buffer_local

            elif task_type == TaskType.EVAL:
                # run the agent through an entire episode to see how it performs
                # when it can't learn from previous experiences on the same episode
                eval_env = Environment(
                    device=device,
                    structure_predictor=env.structure_predictor,
                    batch_size=config.eval_batch_size,
                )

                assert agent.device == "cpu"
                agent = agent.to(device)

                all_rewards = []
                states = eval_env.get_states()
                for i in range(config.max_episode_length):
                    with torch.no_grad():
                        actions = agent.select_actions(states)
                        states, rewards = eval_env.step(states, actions)
                    all_rewards.append(rewards)
                assert torch.all(
                    eval_env.steps_done == 0
                ), f"Steps done: {eval_env.steps_done}"
                rewards = torch.stack(all_rewards)

                results.put(
                    (
                        TaskType.EVAL,
                        {
                            "rewards_mean": torch.mean(rewards).item(),
                            "rewards_std": torch.std(rewards).item(),
                            "rewards_min": torch.min(rewards).item(),
                            "rewards_max": torch.max(rewards).item(),
                            "device_id": device_id,
                            "task_id": task["task_id"],
                        },
                    )
                )
                del agent, states, rewards, eval_env

            else:
                raise ValueError(f"Unknown task type: {task_type}")
            torch.cuda.empty_cache()

    except BaseException as e:
        msg = "Could not format error"
        try:
            exc = traceback.format_exc()
            msg = exc
        except:
            try:
                exc = str(e)
                msg = exc
            except:
                pass
        results.put((TaskType.ERROR, {"error": msg}))
        raise


class MultiTrainer:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.device_ids = list(range(torch.cuda.device_count()))
        self.num_workers = len(self.device_ids)

    def run(self):
        mp.set_start_method("spawn", force=True)
        tasks = mp.Queue()
        results = mp.Queue()

        self.agent = Agent(device="cpu")
        logger.info(f"Number of agent parameters: {self.agent.get_n_params()}")

        self.buffer = Buffer(device="cpu")
        self.buffer.share_memory()
        self.buffer.print_memory_usage()
        buffer_lock = mp.Lock()

        workers = []
        for i in range(self.num_workers):
            logger.info({"Msg": f"Starting worker {i}"})
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
            tasks_since_last_eval = 0

            prev_train_completed = True
            prev_eval_completed = True

            n_active_workers = 0
            task_id = 0
            epoch = 0
            while epoch < config.n_epochs:
                assert n_active_workers <= self.num_workers and n_active_workers >= 0
                assert tasks_since_last_train >= 0 and tasks_since_last_eval >= 0

                # Choose task type and add to task queue
                if (
                    tasks_since_last_train >= config.train_interval
                    and prev_train_completed
                    and self.buffer.get_size() >= config.train_batch_size
                ):
                    logger.info(
                        {
                            "Msg": "Task created",
                            "TaskType": "Train",
                            "TaskID": task_id,
                        }
                    )
                    task_type = TaskType.TRAIN_AGENT
                    tasks.put(
                        (
                            {
                                "type": task_type,
                                "agent": self.agent,
                                "task_id": task_id,
                            }
                        )
                    )
                    prev_train_completed = False
                    tasks_since_last_train = 0
                elif (
                    tasks_since_last_eval >= config.eval_interval
                    and prev_eval_completed
                ):
                    logger.info(
                        {
                            "Msg": "Task created",
                            "TaskType": "Eval",
                            "TaskID": task_id,
                        }
                    )
                    task_type = TaskType.EVAL
                    tasks.put(
                        (
                            {
                                "type": task_type,
                                "agent": self.agent,
                                "task_id": task_id,
                            }
                        )
                    )
                    prev_eval_completed = False
                    tasks_since_last_eval = 0
                else:
                    logger.info(
                        {
                            "Msg": "Task created",
                            "TaskType": "Structure",
                            "TaskID": task_id,
                        }
                    )
                    task_type = TaskType.PREDICT_STRUCTURE
                    tasks.put(
                        (
                            {
                                "type": task_type,
                                "agent": self.agent,
                                "task_id": task_id,
                            }
                        )
                    )

                if task_type != TaskType.EVAL:
                    tasks_since_last_eval += 1
                if task_type != TaskType.TRAIN_AGENT:
                    tasks_since_last_train += 1

                task_id += 1
                n_active_workers += 1

                if n_active_workers == self.num_workers:
                    # Take result from queue
                    logger.info({"Msg": "Waiting for result from queue"})
                    task_type, result = results.get()

                    if task_type == TaskType.ERROR:
                        # try to get the next result with message
                        logger.error({"Msg": "Error in subprocess"})
                        task_type, result = results.get()
                        logger.error({"Error": result["error"]})
                        raise Exception(result["error"])

                    elif task_type == TaskType.TRAIN_AGENT:
                        assert result["agent"].device == "cpu", result["agent"].device
                        self.agent.copy_(result["agent"])
                        epoch += 1
                        prev_train_completed = True

                        logger.info(
                            {
                                "Msg": "Task completed",
                                "TaskType": "Train",
                                "Epoch": epoch,
                                "DeviceID": result["device_id"],
                                "TaskID": result["task_id"],
                                "GpuMemoryAllocated": result["gpu_memory_allocated"],
                                "GpuMemoryReserved": result["gpu_memory_reserved"],
                                "AvgLoss": result["avg_loss"],
                                "AvgReward": result["avg_reward"],
                                "AvgStateActionValue": result["avg_state_action_value"],
                                "Epsilon": self.agent.get_epsilon().item(),
                            }
                        )

                        if config.wandb_enabled:
                            wandb.log(
                                {
                                    "avg_loss": result["avg_loss"],
                                    "avg_reward": result["avg_reward"],
                                    "avg_state_action_value": result[
                                        "avg_state_action_value"
                                    ],
                                    "epsilon": self.agent.get_epsilon().item(),
                                    "epoch": epoch,
                                }
                            )

                        if (
                            config.save_enabled
                            and (epoch + 1) % config.save_interval == 0
                        ):
                            self.save(epoch)

                    elif task_type == TaskType.PREDICT_STRUCTURE:
                        logger.info(
                            {
                                "Msg": "Task completed",
                                "TaskType": "Structure",
                                "TaskID": result["task_id"],
                                "DeviceID": result["device_id"],
                                "BufferSize": self.buffer.get_size(),
                                "BufferPos": self.buffer.get_pos(),
                                "Reward": result["reward"],
                                "Confidence": result["confidence"],
                                "DistancePenalty": result["distance_penalty"],
                                "GpuMemoryAllocated": result["gpu_memory_allocated"],
                                "GpuMemoryReserved": result["gpu_memory_reserved"],
                                "EnvReset": result["env_reset"],
                            }
                        )

                        if config.wandb_enabled:
                            wandb.log(
                                {
                                    "reward": result["reward"],
                                    "confidence": result["confidence"],
                                    "distance_penalty": result["distance_penalty"],
                                    "gpu_memory_allocated": result[
                                        "gpu_memory_allocated"
                                    ],
                                    "gpu_memory_reserved": result[
                                        "gpu_memory_reserved"
                                    ],
                                    "task_id": result["task_id"],
                                    "device_id": result["device_id"],
                                    "buffer_size": self.buffer.get_size(),
                                    "buffer_pos": self.buffer.get_pos(),
                                    "env_reset": result["env_reset"],
                                }
                            )
                    elif task_type == TaskType.EVAL:
                        prev_eval_completed = True
                        logger.info(
                            {
                                "Msg": "Task completed",
                                "TaskType": "Eval",
                                "TaskID": result["task_id"],
                                "DeviceID": result["device_id"],
                                "RewardsMean": result["rewards_mean"],
                                "RewardsStd": result["rewards_std"],
                                "RewardsMin": result["rewards_min"],
                                "RewardsMax": result["rewards_max"],
                            }
                        )
                        if config.wandb_enabled:
                            wandb.log(
                                {
                                    "rewards_mean": result["rewards_mean"],
                                    "rewards_std": result["rewards_std"],
                                    "rewards_min": result["rewards_min"],
                                    "rewards_max": result["rewards_max"],
                                    "task_id": result["task_id"],
                                    "device_id": result["device_id"],
                                }
                            )
                    n_active_workers -= 1

        except BaseException as e:
            logger.error({"Msg": "Error in main process"})
            logger.error({"Error": e})
            raise e

        finally:
            for _ in workers:
                tasks.put(None)
            for p in workers:
                p.join()

    def save(self, n_epochs: int | None = None):
        if n_epochs is None:
            n_epochs = "final"
        logger.info(
            {
                "Msg": "Saving model and buffer",
                "Epoch": n_epochs,
                "SaveDir": self.save_dir,
            }
        )
        self.agent.save_model(self.save_dir / f"model_{n_epochs}.pt")
        try:
            self.buffer.save(self.save_dir / f"buffer_temp.pth")
        except Exception as e:
            os.remove(self.save_dir / f"buffer_temp.pth")
            logger.error({"Msg": "Error saving buffer", "Error": e})
            raise e
        # move temp file to final file
        os.rename(self.save_dir / f"buffer_temp.pth", self.save_dir / f"buffer.pth")

        # test model save
        agent_test = Agent.load_model(self.save_dir / f"model_{n_epochs}.pt", "cpu")
        assert self.agent == agent_test, "Model is not equal to saved model"
