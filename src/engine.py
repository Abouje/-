from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .autograd import Tensor
from .data import iterate_minibatches
from .losses import l2_regularization, softmax_cross_entropy
from .metrics import accuracy_score


@dataclass
class EpochStats:
    loss: float
    acc: float


def predict_logits(model, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    outputs = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size]
        logits = model(Tensor(xb, requires_grad=False))
        outputs.append(logits.data)
    return np.concatenate(outputs, axis=0)


def train_one_epoch(
    model,
    optimizer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    weight_decay: float,
    epoch_seed: int,
) -> EpochStats:
    total_loss = 0.0
    total_correct = 0
    total = 0

    for xb, yb in iterate_minibatches(x_train, y_train, batch_size, shuffle=True, seed=epoch_seed):
        model.zero_grad()
        logits = model(Tensor(xb, requires_grad=True))
        ce = softmax_cross_entropy(logits, yb)
        reg = l2_regularization(model.parameters(), weight_decay)
        loss = ce + reg
        loss.backward()
        optimizer.step()

        preds = logits.data.argmax(axis=1)
        total_loss += float(loss.data) * xb.shape[0]
        total_correct += int((preds == yb).sum())
        total += xb.shape[0]

    return EpochStats(loss=total_loss / total, acc=total_correct / total)


def evaluate(
    model,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> EpochStats:
    logits = predict_logits(model, x, batch_size)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True) + 1e-12

    loss = -np.log(probs[np.arange(y.shape[0]), y] + 1e-12).mean()
    pred = logits.argmax(axis=1)
    acc = accuracy_score(y, pred)
    return EpochStats(loss=float(loss), acc=acc)


def step_decay(initial_lr: float, epoch: int, decay_rate: float, decay_every: int) -> float:
    factor = decay_rate ** (epoch // decay_every)
    return initial_lr * factor


def save_checkpoint(path: str, model, extra: Dict):
    payload = {"state_dict": model.state_dict(), **extra}
    np.savez(path, **payload)


def load_checkpoint(path: str) -> Dict:
    data = np.load(path, allow_pickle=True)
    out = {}
    for key in data.files:
        value = data[key]
        if value.dtype == object and value.size == 1:
            out[key] = value.item()
        else:
            out[key] = value
    return out
