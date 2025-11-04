import hydra
import lightning as L
from hydra.utils import instantiate, call


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg):

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    logger = instantiate(cfg.logger)
    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    trainer = L.Trainer(logger=logger)
    trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":
    main()
