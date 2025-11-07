import os
import shutil
import tarfile

import lightning as L
import torch
from note_seq import midi_to_note_sequence, MIDIConversionError
from torch.utils.data import random_split, DataLoader, Dataset


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
        self.prep_dir = os.path.join(self.data_dir, "prep")
        self.info_file = os.path.join(self.data_dir, "info.txt")
        self.download_url = download_url
        self.batch_size = batch_size
        self.splits = splits

        self.no_files = None
        self.train_set = None
        self.test_set = None
        self.val_set = None

    def prepare_data(self):
        # download data
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        tar_file = os.path.join(self.data_dir, "data.tar.gz")
        if not os.path.exists(tar_file):
            torch.hub.download_url_to_file(self.download_url, tar_file)

        # extracting archive
        if not os.path.exists(self.info_file):
            print("Deleting corrupted data...")
            shutil.rmtree(self.raw_dir)
            print("Done.")

        if not os.path.exists(self.raw_dir):
            print("Extracting archive...")

            no_files = 0

            def tar_filter(info, _):
                nonlocal no_files
                if not info.isfile():
                    return None
                no_files += 1
                return info.replace(name=f"{no_files}.midi")

            with tarfile.open(tar_file) as file:
                file.extractall(self.raw_dir, filter=tar_filter)

            with open(self.info_file, "wt") as file:
                file.write(f"{no_files}")

            self.no_files = no_files
            print(f"{no_files} files extracted.")
        else:
            with open(self.info_file, "rt") as file:
                self.no_files = int(file.read())
            print(f"{self.no_files} already present.")

        # converting data
        if not os.path.exists(self.prep_dir):
            os.makedirs(self.prep_dir)

        print("Converting data...")
        for i in range(1, self.no_files + 1):
            file = os.path.join(self.raw_dir, f"{i}.midi")
            print("pre", file)
            try:
                ns = midi_to_note_sequence(open(file, 'rb').read())
            except MIDIConversionError:
                pass  # ignore
            print("post")
        print("Data converted.")

    def setup(self, stage: str):
        full_set = MIDIDataset(self.raw_dir, self.no_files, ".midi")
        self.train_set, self.val_set, self.test_set = random_split(full_set, self.splits)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        raise ValueError("Not supported yet.")
