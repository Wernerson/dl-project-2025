from symusic import Score, Synthesizer, dump_wav, BuiltInSF3


class AudioConverter:
    def __init__(self, tokenizer, sample_rate=44100, quality=4):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.synthesizer = Synthesizer(
            sf_path=BuiltInSF3.MuseScoreGeneral().path(download=True),
            sample_rate=sample_rate,
            quality=quality
        )

    def to_score(self, tokens):
        if tokens.dim() == 3:
            tokens = tokens[0]
        return self.tokenizer.decode(tokens.cpu().numpy())

    def to_audio(self, tokens):
        # clip midi to 60s max, otherwise we run out of memory
        score = self.to_score(tokens)
        total_duration = score.end()
        if total_duration > 60:
            score = score.clip(0, 60)
        return self.synthesizer.render(score, stereo=True)

    def to_wav(self, tokens, file):
        audio = self.to_audio(tokens)
        dump_wav(str(file), audio, sample_rate=self.sample_rate)

    def to_abc(self, tokens, file):
        score = self.to_score(tokens)
        score.dump_abc(file)

    def to_midi(self, tokens, file):
        score = self.to_score(tokens)
        score.dump_midi(file)

    def midi_to_abc(self, midi_file, abc_file):
        score = Score.from_file(midi_file, fmt="midi")
        score.dump_abc(abc_file)
