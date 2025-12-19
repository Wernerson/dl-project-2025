import hydra
import lightning as L
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.trainer)

    print("\nTraining...")
    trainer.fit(model, datamodule=dataset)

    print("\nTesting...")
    trainer.test(model, datamodule=dataset)


if __name__ == "__main__":
    main()
