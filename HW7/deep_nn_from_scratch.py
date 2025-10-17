
"""
deep_nn_from_scratch.py
-----------------------
A minimal NumPy-only deep neural network implementation with:
  - Activation classes: Linear, ReLU, Sigmoid, Tanh, Softmax
  - Losses: MSELoss, CrossEntropyLoss (stable; softmax fused in backward-friendly form)
  - Dense (fully-connected) layers
  - NeuralNetwork class supporting forward, backward, and SGD updates

Notes
-----
1) For multi-class classification, prefer using CrossEntropyLoss directly on logits
   (WITHOUT adding a Softmax layer at the end). CrossEntropyLoss computes a stable
   log-softmax internally, and its backward uses (softmax(logits) - one_hot) / N.
   If you still want explicit Softmax (e.g., for inference), use it only at eval time.

2) All classes cache forward-pass intermediates for correct backprop.

3) Weight initialization:
   - Xavier (Glorot) for tanh/sigmoid
   - He for ReLU/Leaky ReLU style activations
   You may override by passing your own initializer.

Usage
-----
Run this file to see a tiny XOR demo at the bottom. Or import and use the APIs.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict

# -------------------------
# Utility: initializers
# -------------------------
def xavier_init(fan_in: int, fan_out: int) -> Tuple[np.ndarray, np.ndarray]:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    W = np.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)
    b = np.zeros((1, fan_out), dtype=np.float64)
    return W, b

def he_init(fan_in: int, fan_out: int) -> Tuple[np.ndarray, np.ndarray]:
    std = np.sqrt(2.0 / fan_in)
    W = np.random.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)
    b = np.zeros((1, fan_out), dtype=np.float64)
    return W, b

# -------------------------
# Base Activation
# -------------------------
class Activation:
    def forward(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def backward(self, dA: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LinearAct(Activation):
    """Identity activation: f(x) = x"""
    def __init__(self):
        self._cache = None
    def forward(self, Z: np.ndarray) -> np.ndarray:
        self._cache = Z
        return Z
    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA

class ReLU(Activation):
    def __init__(self):
        self._mask = None
    def forward(self, Z: np.ndarray) -> np.ndarray:
        self._mask = (Z > 0)
        return Z * self._mask
    def backward(self, dA: np.ndarray) -> np.ndarray:
        # d/dZ ReLU = 1(Z>0)
        return dA * self._mask

class Sigmoid(Activation):
    def __init__(self):
        self._A = None
    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = 1.0 / (1.0 + np.exp(-Z))
        self._A = A
        return A
    def backward(self, dA: np.ndarray) -> np.ndarray:
        # d/dZ sigma = sigma*(1-sigma)
        A = self._A
        return dA * A * (1.0 - A)

class Tanh(Activation):
    def __init__(self):
        self._A = None
    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = np.tanh(Z)
        self._A = A
        return A
    def backward(self, dA: np.ndarray) -> np.ndarray:
        A = self._A
        return dA * (1.0 - A**2)

class Softmax(Activation):
    """
    Numerically stable softmax.
    Note: Prefer to use CrossEntropyLoss directly on logits for training.
    This class is useful for inference or in rare cases where you need explicit softmax.
    """
    def __init__(self, axis: int = 1):
        self.axis = axis
        self._S = None  # softmax output
    def forward(self, Z: np.ndarray) -> np.ndarray:
        Z_shift = Z - np.max(Z, axis=self.axis, keepdims=True)
        expZ = np.exp(Z_shift)
        S = expZ / np.sum(expZ, axis=self.axis, keepdims=True)
        self._S = S
        return S
    def backward(self, dA: np.ndarray) -> np.ndarray:
        # Jacobian-vector product for softmax: for each sample:
        # dZ = (dA - sum(dA * S))*S
        S = self._S
        # sum over classes axis
        dot = np.sum(dA * S, axis=self.axis, keepdims=True)
        return (dA - dot) * S

# -------------------------
# Dense (fully connected) Layer
# -------------------------
class Dense:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Activation,
        initializer: Optional[Callable[[int, int], Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        # pick default initializer based on activation
        if initializer is None:
            if isinstance(activation, (ReLU,)) :
                initializer = he_init
            else:
                initializer = xavier_init
        self.W, self.b = initializer(in_features, out_features)
        # caches
        self._X = None
        self._Z = None
        # gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.W + self.b  # (N,D)@(D,M) + (1,M) -> (N,M)
        self._X = X
        self._Z = Z
        return self.activation.forward(Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        # Backprop through activation
        dZ = self.activation.backward(dA)  # (N,M)
        X = self._X  # (N,D)
        # Gradients
        self.dW = X.T @ dZ / X.shape[0]          # (D,N)@(N,M) -> (D,M), mean over batch
        self.db = np.sum(dZ, axis=0, keepdims=True) / X.shape[0]  # (1,M)
        # Propagate to previous layer
        dX = dZ @ self.W.T  # (N,M)@(M,D)->(N,D)
        return dX

    def sgd_step(self, lr: float):
        self.W -= lr * self.dW
        self.b -= lr * self.db

# -------------------------
# Loss functions
# -------------------------
class Loss:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError
    def backward(self) -> np.ndarray:
        raise NotImplementedError

class MSELoss(Loss):
    def __init__(self):
        self._y_pred = None
        self._y_true = None
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._y_pred = y_pred
        self._y_true = y_true
        return float(np.mean((y_pred - y_true) ** 2))
    def backward(self) -> np.ndarray:
        # dL/dy_pred = 2/N * (y_pred - y_true)
        N = self._y_pred.shape[0]
        return (2.0 / N) * (self._y_pred - self._y_true)

class CrossEntropyLoss(Loss):
    """
    Applies softmax + NLL in a numerically stable way on logits.
    y_true is expected as:
      - one-hot array of shape (N, C), or
      - integer class indices of shape (N,)
    """
    def __init__(self):
        self._logits = None
        self._probs = None
        self._y_true = None
    @staticmethod
    def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
        if y.ndim == 1:
            oh = np.zeros((y.shape[0], num_classes), dtype=np.float64)
            oh[np.arange(y.shape[0]), y] = 1.0
            return oh
        return y.astype(np.float64)
    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        self._logits = logits
        # stable log-softmax
        z = logits - np.max(logits, axis=1, keepdims=True)
        expz = np.exp(z)
        probs = expz / np.sum(expz, axis=1, keepdims=True)
        self._probs = probs
        # handle labels
        if y_true.ndim == 1:
            # class indices
            N, C = logits.shape
            self._y_true = self._one_hot(y_true, C)
            log_likelihood = -np.log(probs[np.arange(N), y_true] + 1e-15)
            return float(np.mean(log_likelihood))
        else:
            # one-hot
            self._y_true = y_true
            log_probs = np.log(probs + 1e-15)
            return float(-np.mean(np.sum(y_true * log_probs, axis=1)))
    def backward(self) -> np.ndarray:
        # dL/dlogits = (probs - one_hot)/N
        N = self._logits.shape[0]
        return (self._probs - self._y_true) / N

# -------------------------
# Neural Network container
# -------------------------
class NeuralNetwork:
    def __init__(self, layers: List[Dense], loss: Loss):
        self.layers = layers
        self.loss = loss
    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss.forward(y_pred, y_true)
    def backward(self):
        # Start from dL/dy_pred
        grad = self.loss.backward()
        # Walk layers in reverse
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    def sgd_step(self, lr: float = 1e-2):
        for layer in self.layers:
            layer.sgd_step(lr)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-2,
        verbose: bool = True
    ) -> List[float]:
        losses = []
        for ep in range(1, epochs + 1):
            # forward
            y_pred = self.forward(X)
            # loss
            L = self.compute_loss(y_pred, y)
            losses.append(L)
            # backward
            self.backward()
            # update
            self.sgd_step(lr)
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == 1):
                print(f"Epoch {ep:4d}/{epochs} - loss: {L:.6f}")
        return losses
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # If the network ends with logits for classification, convert to probabilities
        logits = self.forward(X)
        z = logits - np.max(logits, axis=1, keepdims=True)
        expz = np.exp(z)
        return expz / np.sum(expz, axis=1, keepdims=True)
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# -------------------------
# Example: XOR classification (2 -> 2 classes)
# -------------------------
def _xor_demo():
    np.random.seed(0)
    # XOR dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([0,1,1,0], dtype=np.int64)  # class indices: 0 or 1

    # Build a small network: 2 -> 8 -> 2 (logits), ReLU hidden
    layers = [
        Dense(2, 8, activation=ReLU()),
        Dense(8, 2, activation=LinearAct()),
    ]
    net = NeuralNetwork(layers, loss=CrossEntropyLoss())

    # Train
    net.fit(X, y, epochs=2000, lr=0.1, verbose=False)
    probs = net.predict_proba(X)
    preds = net.predict(X)
    acc = np.mean(preds == y)
    print("XOR predictions (probabilities):\n", np.round(probs, 3))
    print("XOR predicted classes:         ", preds)
    print("XOR accuracy:                  ", acc)

if __name__ == "__main__":
    _xor_demo()
