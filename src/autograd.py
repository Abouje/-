from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Set, Tuple

import numpy as np


Array = np.ndarray


def _ensure_array(x) -> Array:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    return np.array(x, dtype=np.float32)


def _unbroadcast(grad: Array, target_shape: Tuple[int, ...]) -> Array:
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


@dataclass
class Tensor:
    data: Array
    requires_grad: bool = False
    grad: Optional[Array] = None

    def __post_init__(self):
        self.data = _ensure_array(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set()

    def __hash__(self):
        return id(self)

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float32)

    def backward(self, grad: Optional[Array] = None):
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be provided for non-scalar tensor")
            grad = np.ones_like(self.data, dtype=np.float32)
        else:
            grad = _ensure_array(grad)

        self.grad = grad
        topo = []
        visited = set()

        def build(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        for node in reversed(topo):
            node._backward()

    def _accumulate_grad(self, g: Array):
        if self.grad is None:
            self.grad = g
        else:
            self.grad = self.grad + g

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(out.grad, other.shape))

        out._prev = {self, other}
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(-out.grad, other.shape))

        out._prev = {self, other}
        out._backward = _backward
        return out

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad * other.data
                self._accumulate_grad(_unbroadcast(g, self.shape))
            if other.requires_grad:
                g = out.grad * self.data
                other._accumulate_grad(_unbroadcast(g, other.shape))

        out._prev = {self, other}
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1)

    def __neg__(self):
        return self * -1.0

    def pow(self, exponent: float):
        out = Tensor(np.power(self.data, exponent), self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad * exponent * np.power(self.data, exponent - 1.0))

        out._prev = {self}
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate_grad(self.data.T @ out.grad)

        out._prev = {self, other}
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad
                if axis is None:
                    g = np.broadcast_to(g, self.shape)
                else:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    if not keepdims:
                        for ax in sorted([(a if a >= 0 else a + self.data.ndim) for a in axes]):
                            g = np.expand_dims(g, ax)
                    g = np.broadcast_to(g, self.shape)
                self._accumulate_grad(g)

        out._prev = {self}
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = float(self.data.size)
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denom = 1.0
            for a in axes:
                denom *= self.data.shape[a]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(denom))

    def log(self):
        out = Tensor(np.log(self.data + 1e-12), self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad / (self.data + 1e-12))

        out._prev = {self}
        out._backward = _backward
        return out

    def exp(self):
        ex = np.exp(self.data)
        out = Tensor(ex, self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad * ex)

        out._prev = {self}
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad * (1.0 - t * t))

        out._prev = {self}
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad * s * (1.0 - s))

        out._prev = {self}
        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(self.data, 0.0)
        out = Tensor(out_data, self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad * (self.data > 0.0))

        out._prev = {self}
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.shape))

        out._prev = {self}
        out._backward = _backward
        return out


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)
