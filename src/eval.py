import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from symusic import Synthesizer, dump_wav, BuiltInSF3

from config import conf_expr
from model.musicbert_diffusion import MusicBertDiffusion

OmegaConf.register_new_resolver(
    "eval",
    lambda expr, **vars: conf_expr(expr, vars),
)


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg):
    print(cfg)
    model = MusicBertDiffusion.load_from_checkpoint(
        cfg.checkpoint,
        net=instantiate(cfg.model.net),
        optimizer=instantiate(cfg.model.optimizer),
        lr_scheduler=instantiate(cfg.model.lr_scheduler),
        offsets=instantiate(cfg.model.offsets),
        mask_strategy=instantiate(cfg.model.mask_strategy)
    )

    print("Start sampling...")
    model.eval()
    tokens = model.sample()
    tokens_np = tokens[0].cpu().numpy()
    print("Sampling one.")

    print("Rendering sample...")
    tokenizer = instantiate(cfg.dataset.tokenizer)
    score = tokenizer.decode(tokens_np)
    synth = Synthesizer(BuiltInSF3.MuseScoreGeneral().path())
    audio = synth.render(score)
    sample_dir = Path(cfg.paths.sample_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    file = sample_dir / "sample.wav"
    dump_wav(str(file), audio, sample_rate=44100)
    print(f"Rendering done. Check {file}")


if __name__ == "__main__":
    main()
