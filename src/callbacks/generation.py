import os
from pathlib import Path

import torch
from lightning import Callback
from utils import AudioConverter


class SampleGeneration(Callback):
    def __init__(self, converter: AudioConverter, sample_dir, num_samples=3):
        super(Callback, self).__init__()
        self.sample_dir = sample_dir
        self.num_samples = num_samples
        self.converter = converter

    def on_train_epoch_end(self, trainer, model):
        model.eval()
        for i in range(self.num_samples):
            try:
                with torch.no_grad():
                    tokens = model.sample()

                sample_dir = Path(self.sample_dir)
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)
                file = sample_dir / f"epoch={trainer.current_epoch}_sample_{i}.wav"
                self.converter.to_wav(tokens, str(file))

                # log to WandB
                trainer.logger.log_audio(
                    "val/samples", [file],
                    sample_rate=[self.converter.sample_rate],
                    step=trainer.global_step
                )

            except Exception as e:
                print(f"[Generation] Skipped sample {i} due to rendering error: {e}")
