import itertools
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from libs.schmubert.prepare_data import _load_midi_trio
from libs.schmubert.utils.data_utils import SubseqSampler
from torch.utils.data import random_split
from tqdm import tqdm

from src.dataset.midi import MIDIData


def load_lakh_trio(path: str, bars=16, max_tensors_per_ns=5):
    root_dir = Path(path)
    p = Pool(4)
    midis = sorted(root_dir.rglob("*.mid"))
    result = list(
        tqdm(p.imap(
            partial(_load_midi_trio, bars, max_tensors_per_ns), midis
        ), total=len(midis), miniters=1)
    )

    result = list(itertools.chain(*result))
    return np.array(result)


class SchmubertDataSampler(MIDIData):
    def __init__(self, data_dir: str, download_url: str, batch_size: int, splits, seq_len):
        super(SchmubertDataSampler, self).__init__(load_lakh_trio, data_dir, download_url, batch_size, splits)
        self.seq_len = seq_len

    def setup(self, stage: str):
        data = np.load(self.cache_file)
        data = SubseqSampler(data, self.seq_len)
        self.train_set, self.val_set, self.test_set = random_split(data, self.splits)
