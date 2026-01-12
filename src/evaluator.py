import os
import random
import shutil
from pathlib import Path
from typing import Sequence

import torch
from tqdm import tqdm
from tqdm.auto import trange

from utils import AudioConverter
from metrics.common import Metric


class Evaluator:
    def __init__(
            self,
            logger,
            eval_dir: str,
            ref_data_dir: str,
            num_samples: int,
            seq_len: int,
            batch_size: int,
            converter: AudioConverter,
            metrics: Sequence[Metric]
    ):
        self.logger = logger
        self.eval_dir = eval_dir
        self.ref_data_dir = ref_data_dir
        self.ref_dir = os.path.join(eval_dir, "ref")
        self.sample_dir = os.path.join(eval_dir, "samples")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.converter = converter
        self.metrics = metrics

    def prepare(self):
        # create required directories
        if not os.path.exists(self.eval_dir):
            os.mkdir(self.eval_dir)
        if not os.path.exists(self.ref_dir):
            os.mkdir(self.ref_dir)
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

        # copy references from dataset
        ref_files = list(Path(self.ref_data_dir).glob("**/*.mid"))
        ref_files = random.sample(ref_files, k=self.num_samples)
        for file in ref_files:
            shutil.copy(file, self.ref_dir)

        # prepare metrics
        for metric in self.metrics:
            metric.prepare()

    def sample(self, model):
        # create samples
        model.eval()
        failed = 0
        for b in trange(self.num_samples // self.batch_size + 1, desc="Generating samples"):
                with torch.no_grad():
                    token_batch = model.sample(seq_len = self.seq_len, batch_size = self.batch_size)
                for i, tokens in enumerate(token_batch):
                    if b * self.batch_size + i - failed > self.num_samples:
                        break # enough samples created
                    try:
                        midi_file = os.path.join(self.sample_dir, f"sample_{b}_{i}.mid")
                        self.converter.to_midi(tokens, midi_file)
                    except Exception as e:
                        print(f"[Evaluation] Skipped sample from batch {b}, item {i} due to rendering error: {e}")
                        failed += 1

    def evaluate(self, model):
        self.prepare()
        self.sample(model)
        for metric in self.metrics:
            try:
                m = metric.evaluate()
                for name, value in m.items():
                    print(type(metric).__name__, name, value)
                    self.logger.experiment.log({f"eval/{name}": value})
            except Exception as e:
                print(f"[Evaluation] Metric {type(metric).__name__} failed to evaluate: {e}")
