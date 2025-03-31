import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, Precision, Recall


class CNNModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(1, 1), padding="same")
        self.conv2 = nn.Conv2d(1, 10, kernel_size=(1, 2), padding="same")
        self.conv3 = nn.Conv2d(1, 10, kernel_size=(1, 3), padding="same")
        self.conv4 = nn.Conv2d(1, 10, kernel_size=(1, 5), padding="same")

        self.bn = nn.BatchNorm2d(40)

        self.pool = nn.AdaptiveMaxPool2d((4, 1))

        self.fc1 = nn.Linear(160, 100)
        self.fc2 = nn.Linear(100, 23)
        self.dropout = nn.Dropout(0.15)
        self.fc3 = nn.Linear(23, 2)

        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")

        self.criterion = F.cross_entropy

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x))

        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x_bn = self.bn(x_cat)

        x_pool = self.pool(x_bn)
        x_flat = torch.flatten(x_pool, start_dim=1)

        x_fc1 = F.relu(self.fc1(x_flat))
        x_fc2 = F.relu(self.fc2(x_fc1))
        x_drop = self.dropout(x_fc2)
        out = self.fc3(x_drop)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        self.train_accuracy(preds, y.int())
        self.train_precision(preds, y.int())
        self.train_recall(preds, y.int())
        self.train_auroc(preds, y.int())

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train_precision", self.train_precision, on_step=True, on_epoch=True)
        self.log("train_recall", self.train_recall, on_step=True, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_accuracy(preds, y.int())
        self.val_precision(preds, y.int())
        self.val_recall(preds, y.int())
        self.val_auroc(preds, y.int())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        self.log("val_precision", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = (preds == y).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"preds": preds, "targets": y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
        }
