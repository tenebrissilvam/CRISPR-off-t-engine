import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import AUROC, Accuracy, Precision, Recall
from transformers import BertConfig, BertModel


class CrisprBERTLightning(pl.LightningModule):
    # CrisprBERT model for predicting presense of the off-target effects via returning 0 or 1. Uses encoded embeddings matrix as an input
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        hidden_act,
        num_attention_heads,
        num_hidden_layers,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        learning_rate,
    ):
        super(CrisprBERTLightning, self).__init__()
        self.save_hyperparameters()

        config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=25,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=1,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            num_labels=2,
        )
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.4)
        self.out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")

    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids)
        lstm_out, (h_n, _) = self.lstm(outputs.last_hidden_state)
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        x = torch.relu(self.fc1(h_n))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.out(x)
        return self.sigmoid(x)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"].float().unsqueeze(1)
        outputs = self(input_ids)
        loss = nn.BCELoss()(outputs, labels)
        preds = (outputs > 0.5).int()

        self.train_accuracy(preds, labels.int())
        self.train_precision(preds, labels.int())
        self.train_recall(preds, labels.int())
        self.train_auroc(outputs, labels.int())

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train_precision", self.train_precision, on_step=True, on_epoch=True)
        self.log("train_recall", self.train_recall, on_step=True, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"].float().unsqueeze(1)
        outputs = self(input_ids)
        loss = nn.BCELoss()(outputs, labels)
        preds = (outputs > 0.5).int()

        self.val_accuracy(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_auroc(outputs, labels.int())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        self.log("val_precision", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
