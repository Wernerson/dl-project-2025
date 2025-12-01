import os
import tarfile
from pathlib import Path

import lightning as L
import torch
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader, random_split


class MIDITok(L.LightningDataModule):
    def __init__(self, data_dir: str, download_url: str, batch_size: int, splits, max_seq_len, tokenizer):
        super(MIDITok, self).__init__()
        self.data_dir = os.path.join(data_dir, "MIDITok")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.download_url = download_url
        self.batch_size = batch_size
        self.splits = splits
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

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

        # Converting data
        if not os.path.exists(self.processed_dir):
            print("Tokenizing data...")
            # Split MIDIs into smaller chunks for training
            split_files_for_training(
                files_paths=list(Path(self.raw_dir).absolute().glob("**/*.mid")),
                tokenizer=self.tokenizer,
                save_dir=Path(self.processed_dir),
                max_seq_len=self.max_seq_len,
            )
            print("Data tokenized.")

    def setup(self, stage: str):
        # Create a Dataset, a DataLoader and a collator to train a model
        dataset = DatasetMIDI(
            files_paths=list(Path(self.processed_dir).glob("**/*.mid")),
            tokenizer=self.tokenizer,
            max_seq_len=1024,
            bos_token_id=self.tokenizer["BOS_None"],
            eos_token_id=self.tokenizer["EOS_None"],
        )

        self.train_set, self.val_set, self.test_set = random_split(dataset, self.splits)

    def train_dataloader(self):
        collator = DataCollator(self.tokenizer.pad_token_id, copy_inputs_as_labels=True)
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=collator)

    def val_dataloader(self):
        collator = DataCollator(self.tokenizer.pad_token_id, copy_inputs_as_labels=True)
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=collator)

    def test_dataloader(self):
        collator = DataCollator(self.tokenizer.pad_token_id, copy_inputs_as_labels=True)
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=collator)

    def predict_dataloader(self):
        raise ValueError("Not supported yet.")
