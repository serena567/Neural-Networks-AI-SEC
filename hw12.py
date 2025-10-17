"""
Mini-batch Training Implementations (assignment-ready)

This file shows how to implement a mini-batch approach in:
1) Pure NumPy (from-scratch iterator + SGD for linear/logistic regression)
2) PyTorch (Dataset + DataLoader + training loop)
3) Keras/TensorFlow (model.fit with batch_size and tf.data pipeline)

NumPy part is fully runnable with only numpy installed. PyTorch/Keras blocks
are optional and guarded by try/except for import.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, Sequence, Callable, Dict
import numpy as np

###############################
# 0) Utilities                #
###############################

def minibatch_iterator(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Yield mini-batches (X_batch, y_batch) from arrays.
    - X: shape (N, D) or arbitrary leading dimension N.
    - y: shape (N, ...) or None for unlabeled data.
    - If drop_last=True, the last small remainder batch is dropped.
    """
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, N, batch_size):
        end = start + batch_size
        if end > N and drop_last:
            break
        batch_idx = idx[start:end]
        Xb = X[batch_idx]
        yb = None if y is None else y[batch_idx]
        yield Xb, yb


############################################
# 1) NumPy: Mini-batch SGD - Linear Reg    #
############################################

def linreg_loss_grad(W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Mean-squared error loss and gradients for linear regression."""
    N = X.shape[0]
    preds = X @ W + b  # shape (N,)
    resid = preds - y
    loss = 0.5 * np.mean(resid ** 2)
    dW = (X.T @ resid) / N  # shape (D,)
    db = np.mean(resid)
    return loss, dW, db


def train_linreg_minibatch(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 1e-2,
    batch_size: int = 64,
    epochs: int = 20,
    seed: Optional[int] = 0,
) -> Dict[str, np.ndarray]:
    """Train linear regression with mini-batch SGD."""
    N, D = X.shape
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(D,))
    b = 0.0
    history = []
    for ep in range(epochs):
        for Xb, yb in minibatch_iterator(X, y, batch_size=batch_size, shuffle=True, seed=seed+ep):
            _, dW, db = linreg_loss_grad(W, b, Xb, yb)
            W -= lr * dW
            b -= lr * db
        # Monitor epoch loss (full-batch eval)
        loss, _, _ = linreg_loss_grad(W, b, X, y)
        history.append(loss)
        # Optional simple LR decay
        # lr *= 0.99
    return {"W": W, "b": np.array([b]), "loss_history": np.array(history)}


############################################
# 2) NumPy: Mini-batch SGD - Logistic Reg  #
############################################

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logreg_loss_grad(W: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """Binary logistic regression with mean cross-entropy."""
    N = X.shape[0]
    z = X @ W + b
    p = _sigmoid(z)
    # add small eps for numerical stability
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    dW = (X.T @ (p - y)) / N
    db = np.mean(p - y)
    return loss, dW, db


def train_logreg_minibatch(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 1e-2,
    batch_size: int = 64,
    epochs: int = 30,
    seed: Optional[int] = 0,
) -> Dict[str, np.ndarray]:
    N, D = X.shape
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(D,))
    b = 0.0
    history = []
    for ep in range(epochs):
        for Xb, yb in minibatch_iterator(X, y, batch_size=batch_size, shuffle=True, seed=seed+ep):
            _, dW, db = logreg_loss_grad(W, b, Xb, yb)
            W -= lr * dW
            b -= lr * db
        loss, _, _ = logreg_loss_grad(W, b, X, y)
        history.append(loss)
    return {"W": W, "b": np.array([b]), "loss_history": np.array(history)}


############################################
# 3) Demo: Compare Full vs Mini-batch      #
############################################

def _demo_compare_linear():
    rng = np.random.default_rng(42)
    N, D = 4000, 20
    W_true = rng.normal(size=(D,))
    X = rng.normal(size=(N, D))
    y = X @ W_true + 0.5 * rng.normal(size=(N,))

    # Train with mini-batch
    mb = train_linreg_minibatch(X, y, lr=5e-2, batch_size=128, epochs=25, seed=1)

    # Full-batch gradient descent for reference
    W = np.zeros(D)
    b = 0.0
    history_full = []
    for ep in range(25):
        loss, dW, db = linreg_loss_grad(W, b, X, y)
        W -= 5e-2 * dW
        b -= 5e-2 * db
        history_full.append(loss)

    return float(mb["loss_history"][-1]), float(history_full[-1])


##########################################################
# 4) PyTorch version: Dataset + DataLoader + training    #
##########################################################
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader

    class NPDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))
        def __len__(self):
            return self.X.shape[0]
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class TorchRegressor(nn.Module):
        def __init__(self, din: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(din, 64), nn.ReLU(),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def torch_train_minibatch(X: np.ndarray, y: np.ndarray, batch_size: int = 128, epochs: int = 10, lr: float = 1e-3, seed: int = 0) -> float:
        torch.manual_seed(seed)
        ds = NPDataset(X, y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        model = TorchRegressor(X.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        # Evaluate training loss for demo
        with torch.no_grad():
            pred = model(torch.from_numpy(X.astype(np.float32)))
            final_loss = loss_fn(pred, torch.from_numpy(y.astype(np.float32))).item()
        return final_loss
except Exception:
    pass


##########################################################
# 5) Keras / TensorFlow quick reference                  #
##########################################################
KERAS_MINIBATCH = r"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

X, y = ...  # numpy arrays (N, D), (N,)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(1)
])
model.compile(optimizer=keras.optimizers.AdamW(1e-3), loss='mse')

# Option A: simple batch_size
model.fit(X, y, batch_size=128, epochs=20, shuffle=True)

# Option B: tf.data pipeline (more control)
ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(5000).batch(128).prefetch(tf.data.AUTOTUNE)
model.fit(ds, epochs=20)
"""


############################################
# 6) Self-checks                           #
############################################
if __name__ == "__main__":
    # Mini-batch vs full-batch (NumPy linreg)
    mb_loss, full_loss = _demo_compare_linear()
    print("LinearReg final loss — minibatch vs full-batch:", (mb_loss, full_loss))

    # Optional Torch demo if torch installed
    try:
        rng = np.random.default_rng(0)
        N, D = 3000, 16
        W_true = rng.normal(size=(D,))
        X = rng.normal(size=(N, D))
        y = X @ W_true + 0.1 * rng.normal(size=(N,))
        t_loss = torch_train_minibatch(X, y, batch_size=128, epochs=5, lr=1e-3)
        print("PyTorch training final MSE:", t_loss)
    except Exception as e:
        print("(PyTorch not available, skipping Torch demo)")
