from pathlib import Path

import wandb
from qres.config import config
from qres.logger import logger
from qres.multi_train import MultiTrainer
from qres.single_train import SingleTrainer

save_dir = Path(__file__).parent / "data"
save_dir.mkdir(parents=True, exist_ok=True)


def get_curr_save_dir():
    iterations = [int(f.stem.split("_")[-1]) for f in save_dir.glob("*/")]
    curr_save_dir = save_dir / f"run_{max(iterations, default=0) + 1}"
    return curr_save_dir


if __name__ == "__main__":
    if config.wandb_enabled:
        wandb.init(
            project=config.project_name,
            config=config.__dict__,
            name=config.run_name,
        )
    try:
        if config.train_type == "multi-gpu":
            trainer = MultiTrainer
        elif config.train_type == "single-gpu":
            trainer = SingleTrainer
        else:
            raise ValueError(f"Invalid train type: {config.train_type}")

        if config.save_enabled:
            curr_save_dir = get_curr_save_dir()
            if not curr_save_dir.exists():
                curr_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            curr_save_dir = None

        trainer = trainer(curr_save_dir)

        trainer.run()
    except Exception as e:
        logger.log(Error=e)
        raise e
    finally:
        if config.save_enabled:
            trainer.save()
            config.save(curr_save_dir / "config.yaml")
        if config.wandb_enabled:
            wandb.finish()
