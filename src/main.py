import hydra
import lightning as L
from hydra.utils import instantiate, call


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    logger = instantiate(cfg.logger)
    train, val = call(cfg.dataset)
    model = instantiate(cfg.model)
    trainer = L.Trainer(logger=logger)
    trainer.fit(model, train, val)


if __name__ == "__main__":
    main()
