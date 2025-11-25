import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from torch.utils.data import TensorDataset


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg):
    # add external libraries to import path
    libs_dir = Path(__file__).resolve().parent / "libs"
    for lib in cfg.libs:
        sys.path.insert(0, str(libs_dir / lib))

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, datamodule=dataset)

    seq_len = 10
    pred = trainer.predict(model, TensorDataset(torch.tensor([seq_len])))
    tokenizer = instantiate(cfg.dataset.tokenizer)
    try:
        midi = tokenizer(pred[0])
        midi.dump_midi("outputs/pred.mid")
    except:
        print("Failed to generate midi.")

if __name__ == "__main__":
    main()
