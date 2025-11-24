import os
import tarfile

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split


class MIDIData(L.LightningDataModule):
    def __init__(
            self,
            process_fn,
            data_dir: str,
            download_url: str,
            batch_size: int,
            splits,
    ):
        super(MIDIData, self).__init__()
        self.process_fn = process_fn
        self.data_dir = os.path.join(data_dir, "MIDI")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.cache_file = os.path.join(self.data_dir, "cache.npy")
        self.download_url = download_url
        self.batch_size = batch_size
        self.splits = splits

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
            data = self.process_fn(self.raw_dir)
            np.save(self.cache_file, data)
            print("Data converted.")

    def setup(self, stage: str):
        data = np.load(self.cache_file)
        self.train_set, self.val_set, self.test_set = random_split(data, self.splits)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        raise ValueError("Not supported yet.")
