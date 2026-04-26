import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image

from src.data import load_eurosat_dataset
from src.engine import load_checkpoint, predict_logits
from src.model import MLPClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Weight visualization and error analysis")
    parser.add_argument("--data_root", type=str, default="./EuroSAT_RGB")
    parser.add_argument("--checkpoint", type=str, default="./results/best_model.npz")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_error_cases", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def visualize_first_layer_weights(weight: np.ndarray, image_size: int, output_path: str):
    num_filters = min(16, weight.shape[1])
    scores = np.linalg.norm(weight, axis=0)
    top_indices = np.argsort(scores)[-num_filters:][::-1]

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(top_indices, start=1):
        patch = weight[:, idx].reshape(image_size, image_size, 3)
        patch = patch - patch.min()
        patch = patch / (patch.max() + 1e-8)
        plt.subplot(4, 4, i)
        plt.imshow(patch)
        plt.axis("off")
        plt.title(f"Neuron {idx}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_errors(paths, y_true, y_pred, class_names, output_path: str, max_cases: int):
    wrong_idx = np.where(y_true != y_pred)[0]
    if wrong_idx.size == 0:
        print("No misclassified samples on test set.")
        return

    pick = wrong_idx[:max_cases]
    cols = 4
    rows = int(np.ceil(len(pick) / cols))

    plt.figure(figsize=(4 * cols, 3 * rows))
    for i, idx in enumerate(pick, start=1):
        img = Image.open(paths[idx]).convert("RGB")
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis("off")
        t = class_names[int(y_true[idx])]
        p = class_names[int(y_pred[idx])]
        plt.title(f"T:{t}\nP:{p}", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    with open(output_path.replace(".png", ".txt"), "w", encoding="utf-8") as f:
        for idx in pick:
            f.write(
                f"{paths[idx]} | true={class_names[int(y_true[idx])]} | pred={class_names[int(y_pred[idx])]}\n"
            )


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

    visualize_first_layer_weights(
        model.fc1.weight.data,
        image_size,
        os.path.join(args.output_dir, "first_layer_weight_viz.png"),
    )

    logits = predict_logits(model, data.test.x, batch_size=args.batch_size)
    pred = logits.argmax(axis=1)
    visualize_errors(
        data.test.paths,
        data.test.y,
        pred,
        data.class_names,
        os.path.join(args.output_dir, "error_cases.png"),
        max_cases=args.num_error_cases,
    )


if __name__ == "__main__":
    main()
