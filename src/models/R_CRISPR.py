import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import AUROC, Accuracy, Precision, Recall


class CrisprNetLightning(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.learning_rate = learning_rate

        # Convolution branches:
        # Note: Our input is expected to be of shape (batch, 7, 1, 24)
        # corresponding to (channels, height, width).
        self.conv0 = nn.Conv2d(
            in_channels=7, out_channels=10, kernel_size=(1, 1), stride=1, padding="same"
        )
        self.conv1 = nn.Conv2d(
            in_channels=7, out_channels=10, kernel_size=(1, 2), stride=1, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=7, out_channels=10, kernel_size=(1, 3), stride=1, padding="same"
        )
        self.conv3 = nn.Conv2d(
            in_channels=7, out_channels=10, kernel_size=(1, 5), stride=1, padding="same"
        )
        self.relu = nn.ReLU()

        # Bidirectional LSTM:
        # After concatenation, we have the original input (7 channels)
        # plus four branches (4 x 10 channels) = 47 channels.
        # We want to reshape to (batch, sequence_length, features) where:
        #    sequence_length = 24 (the width dimension)
        #    features = 47 (all channels)
        self.lstm = nn.LSTM(
            input_size=47, hidden_size=15, batch_first=True, bidirectional=True
        )

        # Fully connected layers:
        # The LSTM returns a sequence output of shape (batch, 24, 30)
        # which is then flattened (24 * 30 = 720) before passing through Dense layers.
        self.fc1 = nn.Linear(24 * 30, 80)
        self.fc2 = nn.Linear(80, 20)
        self.dropout = nn.Dropout(0.35)
        self.fc_out = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

        # Metrics for training and validation:
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")

    def forward(self, x):
        """
        Forward pass:
          x: tensor of shape (batch, 7, 1, 24)
        """
        # Compute each branch (applying ReLU right after each conv):
        branch0 = self.relu(self.conv0(x))  # shape: (batch, 10, 1, 24)
        branch1 = self.relu(self.conv1(x))  # shape: (batch, 10, 1, 24)
        branch2 = self.relu(self.conv2(x))  # shape: (batch, 10, 1, 24)
        branch3 = self.relu(self.conv3(x))  # shape: (batch, 10, 1, 24)

        # Concatenate the original input with branch outputs along the channel dimension.
        # Original input: (batch, 7, 1, 24)
        # Branch outputs: each (batch, 10, 1, 24)
        # Total channels after concatenation: 7 + 4*10 = 47.
        x_cat = torch.cat(
            [x, branch0, branch1, branch2, branch3], dim=1
        )  # (batch, 47, 1, 24)

        # Remove the height dimension (which is 1) and transpose so that
        # the sequence length corresponds to the width dimension (24).
        x_cat = x_cat.squeeze(2)  # (batch, 47, 24)
        x_seq = x_cat.transpose(1, 2)  # (batch, 24, 47)

        # Pass the sequence through the bidirectional LSTM.
        lstm_out, _ = self.lstm(x_seq)  # (batch, 24, 30)

        # Flatten the LSTM output.
        lstm_out = lstm_out.reshape(lstm_out.size(0), -1)  # (batch, 24 * 30 = 720)

        # Pass through fully connected layers.
        x = self.relu(self.fc1(lstm_out))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return self.sigmoid(x)

    def training_step(self, batch, batch_idx):
        # Expect batch to be a dict with keys 'inputs' and 'labels'.
        inputs, labels = batch  # input[1, 128, 24, 7]
        # shape: (batch, 7, 1, 24)
        labels = labels.float().unsqueeze(1)  # shape: (batch, 1)
        # print(inputs.size())
        inputs = inputs.unsqueeze(0)
        outputs = self(inputs.permute(1, 3, 0, 2))
        loss = F.binary_cross_entropy(outputs, labels)
        preds = (outputs > 0.5).int()

        # Update training metrics.
        self.train_accuracy(preds, labels.int())
        self.train_precision(preds, labels.int())
        self.train_recall(preds, labels.int())
        self.train_auroc(outputs, labels.int())

        # Log metrics (these logs will appear in wandb if using a WandbLogger).
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train_precision", self.train_precision, on_step=True, on_epoch=True)
        self.log("train_recall", self.train_recall, on_step=True, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        # shape: (batch, 7, 1, 24)
        labels = labels.float().unsqueeze(1)  #
        # print(inputs.size())
        inputs = inputs.unsqueeze(0)
        outputs = self(inputs.permute(1, 3, 0, 2))
        loss = F.binary_cross_entropy(outputs, labels)
        preds = (outputs > 0.5).int()

        # Update validation metrics.
        self.val_accuracy(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_auroc(outputs, labels.int())

        # Log metrics to the progress bar.
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        self.log("val_precision", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
