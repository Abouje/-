import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_eurosat_dataset
from src.engine import evaluate, load_checkpoint, save_checkpoint, step_decay, train_one_epoch
from src.model import MLPClassifier
from src.optim import SGD


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP on EuroSAT")
    parser.add_argument("--data_root", type=str, default="./EuroSAT_RGB")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--decay_every", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="Path to a checkpoint (.npz) to continue training from",
    )
    return parser.parse_args()


def plot_curves(history, output_dir: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_acc"], color="tab:green", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_acc_curve.png"), dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_eurosat_dataset(
        root_dir=args.data_root,
        image_size=(args.image_size, args.image_size),
        seed=args.seed,
    )

    model = MLPClassifier(
        input_dim=data.train.x.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=len(data.class_names),
        activation=args.activation,
        seed=args.seed,
    )
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_ckpt = os.path.join(args.output_dir, "best_model.npz")
    best_val_acc = -1.0
    start_epoch = 0

    if args.resume_checkpoint:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"resume_checkpoint not found: {args.resume_checkpoint}")
        resume_data = load_checkpoint(args.resume_checkpoint)
        model.load_state_dict(resume_data["state_dict"])

        if "history" in resume_data and isinstance(resume_data["history"], dict):
            history = resume_data["history"]
        if "best_val_acc" in resume_data:
            best_val_acc = float(np.array(resume_data["best_val_acc"]).reshape(-1)[0])
        if "epoch" in resume_data:
            start_epoch = int(np.array(resume_data["epoch"]).reshape(-1)[0]) + 1

        print(
            f"Resume training from checkpoint: {args.resume_checkpoint}\n"
            f"Start epoch: {start_epoch + 1}, previous best val acc: {best_val_acc:.4f}"
        )

    end_epoch = start_epoch + args.epochs

    for epoch in range(start_epoch, end_epoch):
        cur_lr = step_decay(args.lr, epoch, args.decay_rate, args.decay_every)
        optimizer.set_lr(cur_lr)

        train_stats = train_one_epoch(
            model,
            optimizer,
            data.train.x,
            data.train.y,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            epoch_seed=args.seed + epoch,
        )
        val_stats = evaluate(model, data.val.x, data.val.y, batch_size=args.batch_size)

        history["train_loss"].append(train_stats.loss)
        history["train_acc"].append(train_stats.acc)
        history["val_loss"].append(val_stats.loss)
        history["val_acc"].append(val_stats.acc)
        history["lr"].append(cur_lr)

        print(
            f"Epoch {epoch + 1:02d}/{end_epoch} "
            f"lr={cur_lr:.5f} "
            f"train_loss={train_stats.loss:.4f} train_acc={train_stats.acc:.4f} "
            f"val_loss={val_stats.loss:.4f} val_acc={val_stats.acc:.4f}"
        )

        if val_stats.acc > best_val_acc:
            best_val_acc = val_stats.acc
            save_checkpoint(
                best_ckpt,
                model,
                {
                    "class_names": np.array(data.class_names, dtype=object),
                    "mean": data.mean,
                    "std": data.std,
                    "image_size": np.array([args.image_size], dtype=np.int64),
                    "hidden_dim": np.array([args.hidden_dim], dtype=np.int64),
                    "epoch": np.array([epoch], dtype=np.int64),
                    "best_val_acc": np.array([best_val_acc], dtype=np.float32),
                    "history": np.array(history, dtype=object),
                },
            )

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plot_curves(history, args.output_dir)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved best checkpoint to: {best_ckpt}")


if __name__ == "__main__":
    main()
