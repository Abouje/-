import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image


@dataclass
class DatasetSplit:
    x: np.ndarray
    y: np.ndarray
    paths: List[str]


@dataclass
class DataBundle:
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    class_names: List[str]
    mean: np.ndarray
    std: np.ndarray


def _discover_classes(root_dir: str) -> List[str]:
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()
    if not classes:
        raise ValueError(f"No class folders found in {root_dir}")
    return classes


def _collect_image_paths(root_dir: str, class_names: List[str]) -> Tuple[List[str], List[int]]:
    all_paths: List[str] = []
    all_labels: List[int] = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        for name in sorted(os.listdir(cls_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                all_paths.append(os.path.join(cls_dir, name))
                all_labels.append(idx)
    if not all_paths:
        raise ValueError(f"No images found under {root_dir}")
    return all_paths, all_labels


def _load_images(paths: List[str], image_size: Tuple[int, int]) -> np.ndarray:
    h, w = image_size
    arr = np.zeros((len(paths), h * w * 3), dtype=np.float32)
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        arr[i] = x.reshape(-1)
    return arr


def _split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _standardize(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-8
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std, mean, std


def load_eurosat_dataset(
    root_dir: str,
    image_size: Tuple[int, int] = (64, 64),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DataBundle:
    class_names = _discover_classes(root_dir)
    paths, labels = _collect_image_paths(root_dir, class_names)

    x = _load_images(paths, image_size)
    y = np.asarray(labels, dtype=np.int64)

    train_idx, val_idx, test_idx = _split_indices(len(y), train_ratio, val_ratio, seed)

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    paths_arr = np.asarray(paths)
    train_paths = paths_arr[train_idx].tolist()
    val_paths = paths_arr[val_idx].tolist()
    test_paths = paths_arr[test_idx].tolist()

    x_train, x_val, x_test, mean, std = _standardize(x_train, x_val, x_test)

    return DataBundle(
        train=DatasetSplit(x_train, y_train, train_paths),
        val=DatasetSplit(x_val, y_val, val_paths),
        test=DatasetSplit(x_test, y_test, test_paths),
        class_names=class_names,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]
