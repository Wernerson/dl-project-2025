import os
import random
import shutil
from pathlib import Path
from typing import Sequence

import torch

from utils import AudioConverter
from metrics.common import Metric


class Evaluator:
    def __init__(
            self,
            eval_dir: str,
            ref_data_dir: str,
            num_references: int,
            num_samples: int,
            converter: AudioConverter,
            metrics: Sequence[Metric]
    ):
        self.eval_dir = eval_dir
        self.ref_dir = os.path.join(eval_dir, "ref")
        self.sample_dir = os.path.join(eval_dir, "samples")
        self.ref_data_dir = ref_data_dir
        self.num_references = num_references
        self.num_samples = num_samples
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
        ref_files = random.sample(ref_files, k=self.num_references)
        for file in ref_files:
            shutil.copy(file, self.ref_dir)

        # prepare metrics
        for metric in self.metrics:
            metric.prepare()

    def sample(self, model):
        # create samples
        model.eval()
        for i in range(self.num_samples):
            try:
                with torch.no_grad():
                    tokens = model.sample()
                midi_file = os.path.join(self.sample_dir, f"sample_{i}.mid")
                self.converter.to_midi(tokens, midi_file)
            except Exception as e:
                print(f"[Evaluation] Skipped sample {i} due to rendering error: {e}")

    def evaluate(self, model):
        self.prepare()
        self.sample(model)
        for metric in self.metrics:
            try:
                metric.evaluate()
            except Exception as e:
                print(f"[Evaluation] Metric {type(metric).__name__} failed to evaluate: {e}")
