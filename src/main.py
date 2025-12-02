import random
import sys
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from hydra.utils import instantiate
from symusic import Synthesizer, BuiltInSF3


def to_audio(tokenizer, predictions, sample_rate):
    synthesizer = Synthesizer(
        sf_path=BuiltInSF3.MuseScoreGeneral().path(download=True),
        sample_rate=sample_rate,
        quality=4
    )
    audios = []
    for pred in predictions:
        try:
            midi = tokenizer(pred.unsqueeze(0))
        except:
            print("Failed to generate midi.")
            continue
        audio = synthesizer.render(midi, stereo=True)
        audio = np.ravel(np.array(audio))
        audios.append(audio)
    return audios


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
    trainer = instantiate(cfg.trainer, logger=logger)
    trainer.fit(model, datamodule=dataset)

    print("Generating some samples... (hopefully...)")
    model.eval()
    model.freeze()
    predictions = []
    with torch.no_grad():
        for i in range(10):
            t = torch.tensor(random.randint(5, 20))
            predictions += model.predict_step(t)
    tokenizer = instantiate(cfg.dataset.tokenizer)
    sample_rate = 44100
    audios = to_audio(tokenizer, predictions, sample_rate)
    logger.log_audio("pred/samples", audios, sample_rate=[sample_rate] * len(audios))


if __name__ == "__main__":
    main()
