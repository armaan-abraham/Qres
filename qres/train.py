from pathlib import Path

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
    try:
        if config.train_type == "multi-gpu":
            trainer = MultiTrainer()
        elif config.train_type == "single-gpu":
            trainer = SingleTrainer()
        else:
            raise ValueError(f"Invalid train type: {config.train_type}")
        trainer.run()
    finally:
        if config.save_enabled:
            curr_save_dir = get_curr_save_dir()
            print(f"Saving to {curr_save_dir}")
            if not curr_save_dir.exists():
                curr_save_dir.mkdir(parents=True, exist_ok=True)
            trainer.agent.save_model(curr_save_dir / "model.pt")
            trainer.buffer.save(curr_save_dir / "buffer.pth")
            config.save(curr_save_dir / "config.yaml")
        logger.finish()
