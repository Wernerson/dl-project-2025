import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import OmegaConf

from config import conf_expr

OmegaConf.register_new_resolver(
    "eval",
    lambda expr, **vars: conf_expr(expr, vars),
)


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    logger = instantiate(cfg.logger)
    logger.log_hyperparams(cfg)

    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.trainer, logger=logger)
    evaluator = instantiate(cfg.evaluator)

    print("\nTraining...")
    trainer.fit(model, datamodule=dataset)

    print("\nTesting...")
    trainer.test(model, datamodule=dataset)

    print("\nEvaluating...")
    evaluator.evaluate(model)


if __name__ == "__main__":
    main()
