import hydra
import lightning as L
import numpy as np
from hydra.utils import instantiate
from symusic import Synthesizer, BuiltInSF3


def to_audio(tokenizer, predictions, sample_rate):
    """Helper for final manual generation"""
    synthesizer = Synthesizer(
        sf_path=BuiltInSF3.MuseScoreGeneral().path(download=True),
        sample_rate=sample_rate,
        quality=4
    )
    audios = []
    for pred in predictions:
        try:
            if pred.dim() == 3: pred = pred[0]
            midi = tokenizer.decode(pred.cpu().numpy())
        except Exception as e:
            print(f"Failed to generate midi: {e}")
            continue
        try:
            audio = synthesizer.render(midi, stereo=True)
        except Exception as e:
            print(f"Failed to convert midi to audio: {e}")
            continue
        audio = np.ravel(np.array(audio))
        audios.append(audio)
    return audios

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