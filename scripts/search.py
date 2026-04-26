import argparse
import csv
import itertools
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_eurosat_dataset
from src.engine import evaluate, step_decay, train_one_epoch
from src.model import MLPClassifier
from src.optim import SGD


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search for EuroSAT MLP")
    parser.add_argument("--data_root", type=str, default="./EuroSAT_RGB")
    parser.add_argument("--output_csv", type=str, default="./results/grid_search.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lrs", type=str, default="0.05,0.02")
    parser.add_argument("--hidden_dims", type=str, default="128,256,512,1024")
    parser.add_argument("--weight_decays", type=str, default="0.0,0.0001")
    parser.add_argument("--activations", type=str, default="relu,tanh")
    return parser.parse_args()


def str_to_list(s, cast):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    data = load_eurosat_dataset(
        root_dir=args.data_root,
        image_size=(args.image_size, args.image_size),
        seed=args.seed,
    )

    lrs = str_to_list(args.lrs, float)
    hidden_dims = str_to_list(args.hidden_dims, int)
    weight_decays = str_to_list(args.weight_decays, float)
    activations = str_to_list(args.activations, str)

    rows = []
    for trial_id, (lr, hd, wd, act) in enumerate(
        itertools.product(lrs, hidden_dims, weight_decays, activations), start=1
    ):
        model = MLPClassifier(
            input_dim=data.train.x.shape[1],
            hidden_dim=hd,
            num_classes=len(data.class_names),
            activation=act,
            seed=args.seed + trial_id,
        )
        optimizer = SGD(model.parameters(), lr=lr)

        best_val_acc = -1.0
        for epoch in range(args.epochs):
            optimizer.set_lr(step_decay(lr, epoch, decay_rate=0.9, decay_every=5))
            train_one_epoch(
                model,
                optimizer,
                data.train.x,
                data.train.y,
                batch_size=args.batch_size,
                weight_decay=wd,
                epoch_seed=args.seed + epoch,
            )
            val_stats = evaluate(model, data.val.x, data.val.y, batch_size=args.batch_size)
            best_val_acc = max(best_val_acc, val_stats.acc)

        row = {
            "trial": trial_id,
            "lr": lr,
            "hidden_dim": hd,
            "weight_decay": wd,
            "activation": act,
            "best_val_acc": best_val_acc,
        }
        rows.append(row)
        print(row)

    rows = sorted(rows, key=lambda x: x["best_val_acc"], reverse=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial", "lr", "hidden_dim", "weight_decay", "activation", "best_val_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved grid search results to {args.output_csv}")
    print(f"Best config: {rows[0]}")


if __name__ == "__main__":
    main()
