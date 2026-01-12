import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import OmegaConf

from config import conf_expr
from model.musicbert_diffusion import MusicBertDiffusion

OmegaConf.register_new_resolver(
    "eval",
    lambda expr, **vars: conf_expr(expr, vars),
)


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg):
    # Print configuration
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)


    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    logger = instantiate(cfg.logger)
    logger.log_hyperparams(cfg)

    dataset = instantiate(cfg.dataset)
    dataset.prepare_data()

    model = MusicBertDiffusion.load_from_checkpoint(
        cfg.checkpoint,
        net=instantiate(cfg.model.net),
        optimizer=instantiate(cfg.model.optimizer),
        lr_scheduler=instantiate(cfg.model.lr_scheduler),
        offsets=instantiate(cfg.model.offsets),
        mask_strategy=instantiate(cfg.model.mask_strategy)
    )

    evaluator = instantiate(cfg.evaluator)
    print("\nEvaluating...")
    evaluator.evaluate(model)

if __name__ == "__main__":
    main()
