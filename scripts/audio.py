from pathlib import Path

import numpy as np
import soundfile as sf
from symusic import Synthesizer, BuiltInSF3, Score

if __name__ == "__main__":
    sample_rate = 44100
    # MIDI_PATH = Path("data/MIDITok/processed")
    MIDI_PATH = Path("data/MIDITok/raw")
    MIDI_FILE = "2.mid"

    # read MIDI file
    score = Score(MIDI_PATH / MIDI_FILE)

    # render midi to audio
    synthesizer = Synthesizer(
        sf_path=BuiltInSF3.MuseScoreGeneral().path(download=True),
        sample_rate=sample_rate,
        quality=4
    )

    audio_data = []
    chunk_size = 60
    total_duration = score.end()
    for start in range(0, int(total_duration), chunk_size):
        end = min(start + chunk_size, total_duration)
        chunk = score.clip(start, end)
        chunk_audio = synthesizer.render(chunk, stereo=True)
        audio_data.append(np.array(chunk_audio))

    audio_np = np.ravel(audio_data)

    # safe audio
    file_name, _ = MIDI_FILE.split(".")
    sf.write(MIDI_PATH / f"{file_name}.wav", audio_np, sample_rate)
