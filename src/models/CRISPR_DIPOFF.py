import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, Precision, Recall


class RNNLightningModel(pl.LightningModule):
    def __init__(self, config, model_type="LSTM"):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model_type = model_type

        self.vocab_size = config["vocab_size"]
        self.emb_size = config["emb_size"]
        self.hidden_size = config["hidden_size"]
        self.lstm_layers = config["lstm_layers"]
        self.bi_lstm = config["bi_lstm"]
        self.reshape = config["reshape"]
        self.number_hidden_layers = config["number_hidder_layers"]
        self.dropout_prob = config["dropout_prob"]

        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")

        self.hidden_shape = self.hidden_size * 2 if self.bi_lstm else self.hidden_size

        # Embedding layer (only if vocab_size > 0)
        self.embedding = (
            nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)
            if self.vocab_size > 0
            else None
        )

        # Choose the recurrent layer type
        if self.model_type == "LSTM":
            self.rnn = nn.LSTM(
                self.emb_size,
                self.hidden_size,
                num_layers=self.lstm_layers,
                batch_first=True,
                bidirectional=self.bi_lstm,
            )
        elif self.model_type == "GRU":
            self.rnn = nn.GRU(
                self.emb_size,
                self.hidden_size,
                num_layers=self.lstm_layers,
                batch_first=True,
                bidirectional=self.bi_lstm,
            )
        else:
            self.rnn = nn.RNN(
                self.emb_size,
                self.hidden_size,
                num_layers=self.lstm_layers,
                batch_first=True,
                bidirectional=self.bi_lstm,
            )

        # Build hidden layers (a series of Linear -> ReLU -> Dropout blocks)
        start_size = self.hidden_shape
        hidden_layers = []
        for _ in range(self.number_hidden_layers):
            hidden_layers.append(
                nn.Sequential(
                    nn.Linear(start_size, start_size // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                )
            )
            start_size = start_size // 2
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output = nn.Linear(start_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # If using embeddings, convert input to long and embed
        if self.embedding is not None:
            x = x.long()
            x = self.embedding(x)
        elif self.reshape:
            x = x.view(x.shape[0], x.shape[1], 1)

        # Pass through the RNN layer
        # (Lightning handles device placement and hidden state initialization automatically)
        if self.model_type == "LSTM":
            x, _ = self.rnn(x)
        else:
            x, _ = self.rnn(x)

        # Use the output from the last time step
        x = x[:, -1, :]

        # Pass through the hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        # acc = (preds == y).float().mean()
        # Log metrics to wandb
        # self.log("train_loss", loss, on_step=False, on_epoch=True)
        # self.log("train_acc", acc, on_step=False, on_epoch=True)

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
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        # acc = (preds == y).float().mean()

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
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"preds": preds, "targets": y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return {
            "optimizer": optimizer,
        }
