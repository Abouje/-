from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .autograd import Parameter


@dataclass
class SGD:
    params: List[Parameter]
    lr: float
    momentum: float = 0.0

    def __post_init__(self):
        self.velocity = [None for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.momentum > 0.0:
                if self.velocity[i] is None:
                    self.velocity[i] = p.grad.copy()
                else:
                    self.velocity[i] = self.momentum * self.velocity[i] + p.grad
                update = self.velocity[i]
            else:
                update = p.grad
            p.data = p.data - self.lr * update

    def set_lr(self, lr: float):
        self.lr = lr
