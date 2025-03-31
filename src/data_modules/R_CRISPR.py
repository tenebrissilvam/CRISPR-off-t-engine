import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.utils.R_CRISPR import Encoder


class TrainerDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        # Ensure targets are torch tensors of type long (for classification)
        self.targets = torch.from_numpy(targets).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Convert inputs to torch.Tensor (float by default)
        return torch.tensor(self.inputs[idx], dtype=torch.float), self.targets[idx]


class RCRISPRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        filepath,
        batch_size=128,
        val_split=0.2,
        rna_column_name="sequence",
        dna_column_name="Target sequence",
        label_column="class",
    ):
        super().__init__()
        self.filepath = filepath
        self.batch_size = batch_size
        self.val_split = val_split

        self.rna_col = rna_column_name
        self.dna_col = dna_column_name
        self.label_col = label_column

    def setup(self, stage=None):
        # Create full dataset
        df = pd.read_csv(self.filepath)

        seq1 = np.array(df[self.rna_col])
        seq2 = np.array(df[self.dna_col])
        classes = np.array(df[self.label_col])

        encoded_inputs = []

        encoded_inputs = np.array(
            [
                Encoder(on_seq=s1, off_seq=s2, with_reg_val=True, value=cl).on_off_code
                for s1, s2, cl in zip(seq1, seq2, classes)
            ]
        )

        encoded_inputs = np.array(encoded_inputs)

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
