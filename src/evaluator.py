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
            metrics: Sequence[Metric],
            use_self_correction: bool = False
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
        self.use_self_correction = use_self_correction

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

        total_gen = 0
        total_corr = 0

        iterator = trange(self.num_samples // self.batch_size + 1, desc="Generating samples")
        for b in iterator:
                with torch.no_grad():
                    token_batch = model.sample(seq_len = self.seq_len, batch_size = self.batch_size)

                    # --- NEW CONDITIONAL BLOCK ---
                    if self.use_self_correction:
                        # 1. Run Correction
                        token_batch, stats = model.self_correct(token_batch, iterations=5)

                        # 2. Update Stats
                        n_unmasked = token_batch.numel()
                        n_corrected = stats["total_corrections"]
                        total_gen += n_unmasked
                        total_corr += n_corrected

                        # 3. Update Progress Bar
                        if n_unmasked > 0:
                            iterator.set_postfix({"CorrRate": f"{(n_corrected / n_unmasked):.1%}"})
                    # -----------------------------

                for i, tokens in enumerate(token_batch):
                    if b * self.batch_size + i - failed > self.num_samples:
                        break # enough samples created
                    try:
                        midi_file = os.path.join(self.sample_dir, f"sample_{b}_{i}.mid")
                        self.converter.to_midi(tokens, midi_file)
                    except Exception as e:
                        print(f"[Evaluation] Skipped sample from batch {b}, item {i} due to rendering error: {e}")
                        failed += 1

        # --- FINAL SUMMARY (Only prints if flag is True) ---
        if self.use_self_correction:
            print("\n" + "=" * 40)
            print("GIDD Correction Statistics")
            print("=" * 40)
            print(f"Total Tokens Generated: {total_gen}")
            print(f"Total Tokens Corrected: {total_corr}")
            if total_gen > 0:
                print(f"Global Correction Rate: {(total_corr / total_gen):.2%}")
            print("=" * 40 + "\n")
        # ---------------------------------------------------

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
