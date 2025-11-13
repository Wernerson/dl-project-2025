import itertools
import os
import tarfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import lightning as L
import numpy as np
import torch
from libs.schmubert.prepare_data import _load_midi_trio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_lakh_trio(path: str, cache_path: str, bars=16, max_tensors_per_ns=5):
    root_dir = Path(path)
    p = Pool(4)
    midis = sorted(root_dir.rglob("*.mid"))
    result = list(
        tqdm(p.imap(
            partial(_load_midi_trio, bars, max_tensors_per_ns), midis
        ), total=len(midis), miniters=1)
    )

    result = list(itertools.chain(*result))
    np.save(cache_path, result)
    return np.array(result)


class MIDIDataset(Dataset):
    def __init__(self, dir, no_files, suffix):
        super(MIDIDataset, self).__init__()
        self.dir = dir
        self.no_files = no_files
        self.suffix = suffix

    def __len__(self):
        return self.no_files

    def __getitem__(self, idx):
        file = os.path.join(self.dir, f"{idx}{self.suffix}")
        # todo https://github.com/dvruette/figaro/blob/main/src/datasets.py#L190
        # todo https://github.com/plassma/symbolic-music-discrete-diffusion/blob/master/prepare_data.py
        return file


class MIDIDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, download_url: str, batch_size: int, splits):
        super().__init__()
        self.data_dir = os.path.join(data_dir, "MIDI")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.cache_file = os.path.join(self.data_dir, "cache.npy")
        self.download_url = download_url
        self.batch_size = batch_size
        self.splits = splits

        self.no_files = None
        self.train_set = None
        self.test_set = None
        self.val_set = None

    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # download data
        tar_file = os.path.join(self.data_dir, "data.tar.gz")
        if not os.path.exists(tar_file):
            print("Downloading tar file...")
            torch.hub.download_url_to_file(self.download_url, tar_file)
            print("Download complete.")

        # extracting archive
        if not os.path.exists(self.raw_dir):
            print("Extracting archive...")
            no_files = 0

            def tar_filter(info, _):
                nonlocal no_files
                if not info.isfile():
                    return None
                no_files += 1
                return info.replace(name=f"{no_files}.mid")

            with tarfile.open(tar_file) as file:
                file.extractall(self.raw_dir, filter=tar_filter)

            print(f"{no_files} files extracted.")

        # converting data
        if not os.path.exists(self.cache_file):
            print("Converting data...")
            result = load_lakh_trio(path=self.raw_dir, cache_path=self.cache_file)
            print("Data converted.")
        else:
            print("Data present. Reading it...")
            result = np.load(self.cache_file)
            print("Data read.")
        print(result.shape)
        exit()

    def setup(self, stage: str):
        # full_set = MIDIDataset(self.raw_dir, self.no_files, ".midi")
        # self.train_set, self.val_set, self.test_set = random_split(full_set, self.splits)
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        raise ValueError("Not supported yet.")
