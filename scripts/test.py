import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.data import load_eurosat_dataset
from src.engine import load_checkpoint, predict_logits
from src.metrics import accuracy_score, confusion_matrix
from src.model import MLPClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained MLP model")
    parser.add_argument("--data_root", type=str, default="./EuroSAT_RGB")
    parser.add_argument("--checkpoint", type=str, default="./results/best_model.npz")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_cm_figure(cm: np.ndarray, class_names, out_path: str):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint)
    state = ckpt["state_dict"]

    hidden_dim = int(ckpt["hidden_dim"][0]) if "hidden_dim" in ckpt else state["fc1.weight"].shape[1]
    image_size = int(ckpt["image_size"][0]) if "image_size" in ckpt else 32

    data = load_eurosat_dataset(
        root_dir=args.data_root,
        image_size=(image_size, image_size),
        seed=args.seed,
    )

    model = MLPClassifier(
        input_dim=data.test.x.shape[1],
        hidden_dim=hidden_dim,
        num_classes=len(data.class_names),
        activation=str(state["activation"]),
        seed=args.seed,
    )
    model.load_state_dict(state)

    logits = predict_logits(model, data.test.x, batch_size=args.batch_size)
    preds = logits.argmax(axis=1)

    test_acc = accuracy_score(data.test.y, preds)
    cm = confusion_matrix(data.test.y, preds, len(data.class_names))

    print(f"Test Accuracy: {test_acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    np.savetxt(os.path.join(args.output_dir, "confusion_matrix.txt"), cm, fmt="%d")
    save_cm_figure(cm, data.class_names, os.path.join(args.output_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
