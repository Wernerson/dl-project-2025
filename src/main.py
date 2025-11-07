import hydra
import lightning as L
from hydra.utils import instantiate

import sys
from pathlib import Path

@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg):
    # add external libraries to import path
    libs_dir = Path(__file__).resolve().parent / "libs"
    for lib in cfg.libs:
        sys.path.insert(0, str(libs_dir / lib))

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    logger = instantiate(cfg.logger)
    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    trainer = L.Trainer(logger=logger)
    trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":
    main()
