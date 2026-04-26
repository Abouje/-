from __future__ import annotations

from typing import Dict, List

import numpy as np

from .autograd import Parameter, Tensor


class Module:
    def parameters(self) -> List[Parameter]:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.weight = Parameter(rng.normal(0.0, scale, size=(in_dim, out_dim)).astype(np.float32))
        self.bias = Parameter(np.zeros((1, out_dim), dtype=np.float32))

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> List[Parameter]:
        return [self.weight, self.bias]


class MLPClassifier(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        activation: str,
        seed: int = 42,
    ):
        self.activation = activation.lower()
        if self.activation not in {"relu", "sigmoid", "tanh"}:
            raise ValueError("activation must be one of: relu/sigmoid/tanh")

        rng = np.random.default_rng(seed)
        self.fc1 = Linear(input_dim, hidden_dim, rng)
        self.fc2 = Linear(hidden_dim, hidden_dim, rng)
        self.fc3 = Linear(hidden_dim, num_classes, rng)

    def _act(self, x: Tensor) -> Tensor:
        if self.activation == "relu":
            return x.relu()
        if self.activation == "sigmoid":
            return x.sigmoid()
        return x.tanh()

    def __call__(self, x: Tensor) -> Tensor:
        h1 = self._act(self.fc1(x))
        h2 = self._act(self.fc2(h1))
        return self.fc3(h2)

    def parameters(self) -> List[Parameter]:
        return self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters()

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "fc1.weight": self.fc1.weight.data.copy(),
            "fc1.bias": self.fc1.bias.data.copy(),
            "fc2.weight": self.fc2.weight.data.copy(),
            "fc2.bias": self.fc2.bias.data.copy(),
            "fc3.weight": self.fc3.weight.data.copy(),
            "fc3.bias": self.fc3.bias.data.copy(),
            "activation": np.array(self.activation),
        }

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        self.fc1.weight.data = state["fc1.weight"].astype(np.float32)
        self.fc1.bias.data = state["fc1.bias"].astype(np.float32)
        self.fc2.weight.data = state["fc2.weight"].astype(np.float32)
        self.fc2.bias.data = state["fc2.bias"].astype(np.float32)
        self.fc3.weight.data = state["fc3.weight"].astype(np.float32)
        self.fc3.bias.data = state["fc3.bias"].astype(np.float32)
