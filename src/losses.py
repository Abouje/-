from __future__ import annotations

import numpy as np

from .autograd import Tensor


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    targets = targets.astype(np.int64)
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)

    n = logits.data.shape[0]
    loss_value = -np.log(probs[np.arange(n), targets] + 1e-12).mean()
    out = Tensor(np.array(loss_value, dtype=np.float32), requires_grad=logits.requires_grad)

    def _backward():
        if out.grad is None or not logits.requires_grad:
            return
        grad = probs.copy()
        grad[np.arange(n), targets] -= 1.0
        grad = grad / n
        grad = grad * out.grad
        if logits.grad is None:
            logits.grad = grad
        else:
            logits.grad = logits.grad + grad

    out._prev = {logits}
    out._backward = _backward
    return out


def l2_regularization(parameters, weight_decay: float) -> Tensor:
    if weight_decay <= 0.0:
        return Tensor(np.array(0.0, dtype=np.float32), requires_grad=False)

    reg = None
    for p in parameters:
        term = (p * p).sum()
        reg = term if reg is None else (reg + term)

    return reg * (0.5 * weight_decay)
