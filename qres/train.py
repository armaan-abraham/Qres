import logging
import os

import wandb
from qres.config import config, get_curr_run_iter, get_curr_save_dir
from qres.logger import setup_logging
from qres.multi_train import MultiTrainer

logger = logging.getLogger()


if __name__ == "__main__":
    if config.wandb_enabled:
        wandb.init(
            project="qres",
            config=config,
            name=f"run_{get_curr_run_iter()}",
        )

    save_dir = None
    if config.save_enabled:
        save_dir = get_curr_save_dir()
        config.save(save_dir / "config.yaml")

    setup_logging(save_dir)

    logger.info(f"Config: {config}")

    try:
        trainer = MultiTrainer(save_dir)
        trainer.run()

    except BaseException as e:
        logger.error({"Msg": "Uncaught exception in train.py", "Error": e})

    finally:
        if config.save_enabled:
            trainer.save()
