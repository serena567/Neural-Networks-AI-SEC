"""
Dropout implementations for a neural network (assignment-ready):
- Pure NumPy inverted-dropout (forward & backward) + a tiny MLP example
- PyTorch: built-in nn.Dropout usage and a custom dropout module
- Keras (TensorFlow): quick example for reference

Notes
-----
* Inverted dropout: during training we scale activations by 1/(1-p) so that
  E[y] ≈ x. At inference, we DO NOT apply dropout or any scaling.
* For backprop, we multiply upstream gradients by the same mask/(1-p).
* Randomness is controlled via RNG keys for reproducibility where applicable.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np

###############################
# 1) NumPy: Inverted Dropout  #
###############################

def dropout_forward(x: np.ndarray, p: float, training: bool, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, Dict]:
    """
    Inverted dropout forward pass.

    Args:
        x: input activations, any shape.
        p: dropout rate in [0,1). Fraction of units to drop.
        training: if True, apply dropout; else return x unchanged.
        rng: optional numpy Generator for reproducibility.

    Returns:
        out: activations after dropout
        cache: dict containing mask and keep_prob for backward
    """
    assert 0 <= p < 1, "dropout rate p must be in [0,1)"
    keep = 1.0 - p
    cache = {"p": p, "training": training, "mask": None}

    if not training or p == 0.0:
        out = x.copy()
        return out, cache

    if rng is None:
        rng = np.random.default_rng()

    # Bernoulli mask with probability keep; scale by 1/keep (inverted)
    mask = (rng.random(size=x.shape) < keep).astype(x.dtype)
    out = (x * mask) / keep
    cache["mask"] = mask
    cache["keep"] = keep
    return out, cache


def dropout_backward(dout: np.ndarray, cache: Dict) -> np.ndarray:
    """
    Backward pass for inverted dropout.
    
    Args:
        dout: upstream gradient (same shape as forward output)
        cache: dict returned from dropout_forward
    Returns:
        dx: gradient w.r.t. x
    """
    p = cache.get("p", 0.0)
    training = cache.get("training", False)
    mask = cache.get("mask", None)

    if (not training) or p == 0.0 or mask is None:
        # During eval mode, dropout is a no-op; gradient passes through.
        return dout

    keep = cache.get("keep", 1.0 - p)
    dx = (dout * mask) / keep
    return dx


########################################
# 2) NumPy: Tiny MLP with Dropout layer #
########################################

@dataclass
class LinearLayer:
    W: np.ndarray
    b: np.ndarray

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
        out = x @ self.W + self.b
        cache = (x,)
        return out, cache

    def backward(self, dout: np.ndarray, cache: Tuple[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        (x,) = cache
        dW = x.T @ dout
        db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx, dW, db


def relu_forward(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    x = cache
    dx = dout * (x > 0)
    return dx


class MLP2:
    """Two-layer MLP with dropout between layers.
    Architecture: input -> Linear -> ReLU -> Dropout -> Linear -> logits
    """
    def __init__(self, din: int, dh: int, dout: int, p_drop: float = 0.5, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        self.fc1 = LinearLayer(W=rng.normal(0, np.sqrt(2.0/din), size=(din, dh)), b=np.zeros(dh))
        self.fc2 = LinearLayer(W=rng.normal(0, np.sqrt(2.0/dh), size=(dh, dout)), b=np.zeros(dout))
        self.p_drop = p_drop
        self.rng = np.random.default_rng(seed+1 if seed is not None else None)

        # grads storage
        self.grads: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        caches = {}
        h1, c1 = self.fc1.forward(x)
        a1, c_relu = relu_forward(h1)
        do, c_do = dropout_forward(a1, p=self.p_drop, training=training, rng=self.rng)
        out, c2 = self.fc2.forward(do)
        caches.update({"lin1": c1, "relu": c_relu, "drop": c_do, "lin2": c2, "a1": a1})
        return out, caches

    def backward(self, dout: np.ndarray, caches: Dict) -> np.ndarray:
        # Backprop through second linear
        dx2, dW2, db2 = self.fc2.backward(dout, caches["lin2"])
        # Through dropout
        dx2 = dropout_backward(dx2, caches["drop"])  # scaled by mask/keep
        # Through ReLU
        dx_relu = relu_backward(dx2, caches["relu"])
        # Through first linear
        dx1, dW1, db1 = self.fc1.backward(dx_relu, caches["lin1"])
        # Save grads
        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return dx1

    def step_sgd(self, lr: float = 1e-2):
        self.fc1.W -= lr * self.grads["W1"]
        self.fc1.b -= lr * self.grads["b1"]
        self.fc2.W -= lr * self.grads["W2"]
        self.fc2.b -= lr * self.grads["b2"]


###############################
# 3) Quick correctness checks #
###############################

def _check_inference_scaling():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10000,))
    p = 0.3
    y_train, cache = dropout_forward(x, p=p, training=True, rng=rng)
    y_eval, _ = dropout_forward(x, p=p, training=False, rng=rng)
    # The mean magnitude should be close after inverted scaling
    return float(np.mean(y_train) - np.mean(y_eval))


def _check_backward_shape():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 5))
    y, cache = dropout_forward(x, p=0.5, training=True, rng=rng)
    dout = rng.normal(size=x.shape)
    dx = dropout_backward(dout, cache)
    return x.shape, dx.shape


#############################################
# 4) PyTorch: built-in & custom Dropout      #
#############################################

# (Keep these imports local to avoid hard dependency if torch isn't installed.)
try:
    import torch
    from torch import nn
    from torch.nn import functional as F

    class CustomDropout(nn.Module):
        """
        Custom inverted-dropout layer replicating core logic.
        Use only for didactic purposes; in practice, prefer nn.Dropout.
        """
        def __init__(self, p: float = 0.5):
            super().__init__()
            assert 0 <= p < 1
            self.p = p
            self.keep = 1.0 - p

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.training or self.p == 0.0:
                return x
            mask = (torch.rand_like(x) < self.keep).float()
            return x * mask / self.keep

    class TorchMLP(nn.Module):
        def __init__(self, din: int, dh: int, dout: int, p_drop: float = 0.5):
            super().__init__()
            self.fc1 = nn.Linear(din, dh)
            self.fc2 = nn.Linear(dh, dout)
            self.dropout = nn.Dropout(p_drop)  # uses inverted dropout semantics

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)  # active only in training mode
            x = self.fc2(x)
            return x

    # Example usage (commented):
    # model = TorchMLP(784, 256, 10, p_drop=0.3)
    # model.train()  # enable dropout
    # logits = model(torch.randn(32, 784))
    # model.eval()   # disable dropout

except Exception as _e:
    # Torch not available; examples will be skipped.
    pass


#############################################
# 5) Keras / TensorFlow quick reference      #
#############################################

KERAS_EXAMPLE = r"""
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(din,)),
    layers.Dropout(0.3),   # inverted dropout
    layers.Dense(num_classes)
])

model.compile(optimizer=keras.optimizers.AdamW(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
"""


if __name__ == "__main__":
    print("Mean difference (train - eval) after inverted scaling:", _check_inference_scaling())
    print("Backward shape check (x, dx):", _check_backward_shape())
