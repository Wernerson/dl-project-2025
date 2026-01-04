import os
import random
import shutil
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from config import conf_expr
from metrics.fss import FSS

OmegaConf.register_new_resolver(
    "eval",
    lambda expr, **vars: conf_expr(expr, vars),
)

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

    print("\nEvaluating...")
    # create directories
    eval_dir = cfg.paths.eval_dir
    ref_dir = os.path.join(eval_dir, "ref")
    sample_dir = os.path.join(eval_dir, "samples")
    os.mkdirs(ref_dir)
    os.mkdirs(sample_dir)

    # copy references from dataset
    ref_files = list(Path(os.path.join(cfg.data_dir, "MIDITok", "processed")).glob("**/*.mid"))
    ref_files = random.sample(ref_files, k=cfg.eval.num_references)
    for file in ref_files:
        shutil.copy(file, ref_dir)

    # create samples
    converter = instantiate(cfg.eval.converter)
    model.eval()
    for i in range(cfg.eval.num_samples):
        try:
            with torch.no_grad():
                tokens = model.sample()
            midi_file = os.paths.join(sample_dir, f"sample_{i}.mid")
            abc_file = os.paths.join(sample_dir, f"sample_{i}.abc")
            converter.to_midi(tokens, midi_file)
            converter.to_abc(tokens, abc_file)
        except Exception as e:
            print(f"[Generation] Skipped eval sample {i} due to rendering error: {e}")
            import traceback
            traceback.print_exc()

    # calculate metrics
    fss = FSS(ref_dir, sample_dir)
    print(fss.evaluate())


if __name__ == "__main__":
    main()
