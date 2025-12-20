import numpy as np
import torch
from lightning import Callback
from symusic import Synthesizer, BuiltInSF3


class SampleGeneration(Callback):
    def __init__(self, tokenizer, sample_rate=44100, num_samples=3, quality=4):
        super(Callback, self).__init__()
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.num_samples = num_samples

        # Initialize Synthesizer
        self.synthesizer = Synthesizer(
            sf_path=BuiltInSF3.MuseScoreGeneral().path(download=True),
            sample_rate=sample_rate,
            quality=quality
        )

    def on_train_epoch_end(self, trainer, model):
        model.eval()
        for i in range(self.num_samples):
            try:
                with torch.no_grad():
                    generated_tokens = model.sample()

                if generated_tokens.dim() == 3:
                    generated_tokens = generated_tokens[0]
                tokens_np = generated_tokens.cpu().numpy()

                # Decode to MIDI object
                midi_obj = self.tokenizer.decode(tokens_np)
                # We render. If it fails due to size, the try-except below catches it.
                audio_data = self.synthesizer.render(midi_obj, stereo=True)

                # Check if audio is empty or too huge before processing
                if len(audio_data) > self.sample_rate * 60:  # Limit to 60 seconds
                    print(f"[Generation] Sample {i} too long ({len(audio_data) / self.sample_rate:.1f}s), truncating.")
                    audio_data = audio_data[:self.sample_rate * 60]

                audio_np = np.ravel(np.array(audio_data))
                trainer.logger.log_audio(
                    "val/samples", [audio_np],
                    sample_rate=[self.sample_rate],
                    step=trainer.current_epoch
                )

            except Exception as e:
                # This catches the bad_alloc (MemoryError in Python)
                print(f"[Generation] Skipped sample {i} due to rendering error: {e}")
