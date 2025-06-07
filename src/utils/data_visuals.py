import os
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

BASE_DIR = Path(__file__).resolve().parents[2]


def get_dataset(runs):
    for i, run in enumerate(runs):
        summary_dict = run.summary
        summary_dict["name"] = run.name
        new_data = pd.DataFrame([dict(summary_dict)])
        if i == 0:
            data = new_data

        else:
            data = pd.concat([data, new_data])

    return data


def process_data(runs):
    df_plots = get_dataset(runs)[:-1]

    plot_cols = [
        "train_acc_epoch",
        "train_acc_step",
        "train_auroc_epoch",
        "train_auroc_step",
        "train_loss_epoch",
        "train_loss_step",
        "train_precision_epoch",
        "train_precision_step",
        "train_recall_epoch",
        "train_recall_step",
        "val_acc",
        "val_auroc",
        "val_loss",
        "val_precision",
        "val_recall",
    ]

    df_plots = df_plots.set_index("name")[plot_cols]
    val_cols = ["val_acc", "val_auroc", "val_precision", "val_recall"]
    df_plots_val = df_plots[val_cols]

    return df_plots_val


def bars_plot(df_plots_val):
    plt.style.use("seaborn-v0_8-darkgrid")

    df_plots_val.T.plot(kind="bar", figsize=(15, 4), rot=0)

    plt.title("Model Comparison on Validation Metrics", fontsize=16)
    plt.ylabel("Metric Value", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(
        title="Models", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("plots/bar_plot.png", dpi=300)
    plt.close()


def heatmap_plot(df_plots_val):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_plots_val, annot=True, cmap="viridis", fmt=".3f", linewidths=0.5)

    plt.title("Heatmap of Model Validation Metrics", fontsize=16)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Models", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plots/heatmap_plot.png", dpi=300)
    plt.close()


def plot_loss(runs):
    loss_data = {}

    for run in runs:
        run_name = run.name or run.id
        try:
            history = run.history(pandas=True)[["train_loss_epoch", "val_loss"]]
            if not history.empty:
                loss_data[run_name] = {
                    "train": history["train_loss_epoch"]
                    .dropna()
                    .reset_index(drop=True),
                    "val": history["val_loss"].dropna().reset_index(drop=True),
                }
        except Exception as e:
            print(f"Could not fetch history for run {run_name}: {e}")

    plt.figure(figsize=(12, 6))
    for run_name, losses in loss_data.items():
        if "val" in losses:
            plt.plot(losses["val"], label=f"{run_name} (val)")
        if "train" in losses:
            plt.plot(losses["train"], linestyle="--", label=f"{run_name} (train)")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("plots/wandb_loss_plot.png", dpi=300)
    plt.close()


@hydra.main(
    version_base=None, config_path=os.path.join(BASE_DIR, "conf/"), config_name="config"
)
def main(cfg):
    api = wandb.Api()
    user = cfg.logging.user
    proj_name = cfg.logging.project
    try:
        runs = api.runs(user + "/" + proj_name)
    except Exception as e:
        print(f"Error fetching runs: {e}")
        sys.exit(1)

    df_plots_val = process_data(runs)
    bars_plot(df_plots_val)
    heatmap_plot(df_plots_val)
    plot_loss(runs)

    print("Plots have been saved to plots folder")


if __name__ == "__main__":
    main()
