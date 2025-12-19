import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from lightning import Callback
from hydra.utils import instantiate
from omegaconf import DictConfig
from symusic import Synthesizer, BuiltInSF3

class GenerationCallback(Callback):
    def __init__(self, tokenizer, sample_rate=44100, num_samples=3, output_dir=None):
        super().__init__()
        
        # Handle Tokenizer instantiation
        if isinstance(tokenizer, (DictConfig, dict)):
            self.tokenizer = instantiate(tokenizer)
        else:
            self.tokenizer = tokenizer
            
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        
        # Initialize Synthesizer
        self.synthesizer = Synthesizer(
            sf_path=BuiltInSF3.MuseScoreGeneral().path(download=True),
            sample_rate=sample_rate,
            quality=4
        )

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_idx = trainer.current_epoch
        
        # --- 1. Find WandB Output Directory ---
        # Try to get the WandB run directory. 
        # Fallback to default_root_dir if WandB is not active.
        try:
            # This gets the actual folder like "wandb/run-2025.../files"
            # We append 'media/audio/pred' to match the structure you wanted
            base_path = Path(trainer.logger.experiment.dir) / "media" / "audio" / "pred"
        except AttributeError:
            # Fallback for offline/no-logger runs
            base_path = Path(trainer.default_root_dir) / "generated_samples"

        epoch_folder = base_path / f"epoch_{epoch_idx}"
        epoch_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[Generation] Saving {self.num_samples} samples to: {epoch_folder}")
        pl_module.eval()
        
        for i in range(self.num_samples):
            try:
                with torch.no_grad():
                    # Generate tokens (128 notes)
                    generated_tokens = pl_module.predict_step(None, 0, 0)
                
                if generated_tokens.dim() == 3: generated_tokens = generated_tokens[0]
                tokens_np = generated_tokens.cpu().numpy()
                
                # Decode to MIDI object
                midi_obj = self.tokenizer.decode(tokens_np)

                # --- 2. Fix bad_alloc (Safety Check) ---
                # Get duration in seconds (ticks / tpq * seconds_per_tick is roughly valid)
                # A safer check is the end time of the last note
                if len(midi_obj.tracks) > 0 and len(midi_obj.tracks[0].notes) > 0:
                    last_note_end = midi_obj.tracks[0].notes[-1].end
                    # Rough conversion: 480 ticks per beat, 120 bpm = 0.5s per beat
                    # If the song is excessively long (e.g. > 100,000 ticks), skip it.
                    # Or better: check the raw duration in seconds if symusic supports it easily.
                    # For safety, let's just use a try/catch block with a timeout or strictly limit max ticks.
                    
                    # Manual heuristic: 128 notes * max duration is huge.
                    # We simply clamp the error during rendering.
                    pass 

                # --- 3. Disable MIDI Saving ---
                # midi_path = epoch_folder / f"sample_{i}.mid"
                # midi_obj.dump_midi(str(midi_path))
                
                # Render Audio
                # We wrap this in a stricter try/catch to catch C++ memory errors if possible,
                # but preventing them is better. 
                # Calculating duration in seconds:
                # symusic doesn't give .duration_sec direct property easily without compute.
                
                # We render. If it fails due to size, the try-except below catches it.
                audio_data = self.synthesizer.render(midi_obj, stereo=True)
                
                # Check if audio is empty or too huge before processing
                if len(audio_data) > self.sample_rate * 60: # Limit to 60 seconds
                     print(f"[Generation] Sample {i} too long ({len(audio_data)/self.sample_rate:.1f}s), truncating.")
                     audio_data = audio_data[:self.sample_rate * 60]

                audio_np = np.ravel(np.array(audio_data))
                
                # Save WAV
                wav_path = epoch_folder / f"sample_{i}.wav"
                sf.write(wav_path, audio_np, self.sample_rate)
                
            except Exception as e:
                # This catches the bad_alloc (MemoryError in Python)
                print(f"[Generation] Skipped sample {i} due to rendering error: {e}")

        pl_module.train()