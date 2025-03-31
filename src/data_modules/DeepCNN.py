import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.utils.DeepCNN import preprocess_inputs_batch


class TrainerDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = torch.from_numpy(targets).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float), self.targets[idx]


class CNNDataModule(pl.LightningDataModule):
    def __init__(self, filepath, batch_size=128, val_split=0.2):
        super().__init__()
        self.filepath = filepath
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage=None):
        # Create full dataset
        df = pd.read_csv(self.filepath)

        seq1 = np.array(df["sequence"])
        seq2 = np.array(df["Target sequence"])

        encoded_inputs = np.array(preprocess_inputs_batch(seq1, seq2))

        classes = np.array(df["class"])

        full_dataset = TrainerDataset(encoded_inputs, classes)
        total_samples = len(full_dataset)
        val_size = int(total_samples * self.val_split)
        train_size = total_samples - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
