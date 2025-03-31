import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.utils.CrisprBERT import base_pair, off_tar_read


class CrisprDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # Tensor of shape (n_samples, seq_length)
        self.labels = labels  # Tensor of shape (n_samples,)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"input_ids": self.sequences[idx], "labels": self.labels[idx]}


class CrisprDataModule(pl.LightningDataModule):
    def __init__(
        self,
        encoding,
        filepath,
        batch_size=128,
        val_split=0.2,
        test_split=0.1,
        num_workers=4,
    ):
        super().__init__()
        self.encoding = encoding
        self.file_path = filepath
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Create the full dataset
        base_p = base_pair()
        base_list = base_p.create_dict(self.encoding)
        encoder = off_tar_read(self.file_path, base_list)

        encode_matrix, class_labels = encoder.encode(self.encoding)

        # print(encode_matrix, class_labels)
        full_dataset = CrisprDataset(encode_matrix.astype(np.int64), class_labels)

        # Optionally split out test data if desired
        if self.test_split > 0:
            test_size = int(len(full_dataset) * self.test_split)
            train_val_size = len(full_dataset) - test_size
            train_val_dataset, self.test_dataset = random_split(
                full_dataset, [train_val_size, test_size]
            )
        else:
            train_val_dataset = full_dataset
            self.test_dataset = None

        # Split train_val_dataset into training and validation sets
        if self.val_split > 0:
            val_size = int(len(train_val_dataset) * self.val_split)
            train_size = len(train_val_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(
                train_val_dataset, [train_size, val_size]
            )
        else:
            self.train_dataset = train_val_dataset
            self.val_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return None

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return None
