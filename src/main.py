import sys
from pathlib import Path
import hydra
import lightning as L
import numpy as np
import soundfile as sf
from hydra.utils import instantiate
from omegaconf import OmegaConf  # <--- NEW IMPORT
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
    # 1. Setup Environment
    libs_dir = Path(__file__).resolve().parent / "libs"
    for lib in cfg.libs:
        sys.path.insert(0, str(libs_dir / lib))

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # 2. Instantiate Components
    tokenizer = instantiate(cfg.dataset.tokenizer)
    logger = instantiate(cfg.logger)
    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)

    # 3. Parse Callbacks (Dict -> List)
    callbacks = []
    if cfg.trainer.get("callbacks"):
        for cb_name, cb_conf in cfg.trainer.callbacks.items():
            if cb_conf is not None:
                print(f"Instantiating callback: {cb_name}")
                callbacks.append(instantiate(cb_conf))

    # 4. Prepare Trainer Config (CRITICAL FIX)
    # We convert the Hydra config to a standard Python dictionary.
    # This allows us to delete the 'callbacks' key so it doesn't conflict with our list.
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    
    if "callbacks" in trainer_cfg:
        del trainer_cfg["callbacks"]

    # 5. Instantiate Trainer
    # Now 'trainer_cfg' has no 'callbacks' key, so passing 'callbacks=callbacks' works perfectly.
    trainer = instantiate(trainer_cfg, logger=logger, callbacks=callbacks)

    print("\nTraining...")
    trainer.fit(model, datamodule=dataset)

    print("\nTesting...")
    trainer.test(model, datamodule=dataset)

    # --- 6. Final Manual Generation Block ---
    print("Generating final manual samples...")
    predictions = []
    for _ in range(5):
        res = trainer.predict(model, datamodule=dataset)
        if res:
             predictions.append(res[0]) 

    print(f"Generated {len(predictions)} final samples.")
    
    sample_rate = 44100
    audios = to_audio(tokenizer, predictions, sample_rate)
    
    if audios:
        logger.log_audio("pred/final_samples", audios, sample_rate=[sample_rate] * len(audios))

if __name__ == "__main__":
    main()