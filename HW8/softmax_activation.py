
"""
softmax_activation.py
---------------------
A standalone, numerically stable Softmax activation class in NumPy,
with forward() and backward() implementations and a small demo.

Usage
-----
python softmax_activation.py
"""
from __future__ import annotations
import numpy as np
from typing import Optional

class Softmax:
    """
    Numerically stable Softmax activation.

    Parameters
    ----------
    axis : int, default=1
        Axis over which to apply softmax (e.g., 1 for (N, C) logits).
    eps : float, default=1e-15
        Small constant to avoid division by zero in extreme cases.

    Notes
    -----
    - During training with cross-entropy, it's common to pass logits
      directly to a stable CrossEntropyLoss (log-softmax inside).
      Use this Softmax explicitly when you need probabilities in the
      network output (e.g., for inference), or when composing custom ops.
    """
    def __init__(self, axis: int = 1, eps: float = 1e-15) -> None:
        self.axis = int(axis)
        self.eps = float(eps)
        self._S: Optional[np.ndarray] = None  # cache probabilities

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute softmax(Z) in a numerically stable way.

        Parameters
        ----------
        Z : ndarray
            Input logits. Softmax is applied along `axis`.

        Returns
        -------
        S : ndarray
            Probabilities of same shape as Z; sums to 1 along `axis`.
        """
        Z = np.asarray(Z, dtype=np.float64)
        # shift by max to improve numerical stability
        Z_shift = Z - np.max(Z, axis=self.axis, keepdims=True)
        expZ = np.exp(Z_shift)
        denom = np.sum(expZ, axis=self.axis, keepdims=True) + self.eps
        S = expZ / denom
        self._S = S
        return S

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Jacobian-vector product for softmax. Given upstream gradient dA=dL/dS,
        return dZ=dL/dZ.

        For each slice along `axis`: J^T @ v = (v - <v, S>) * S
        where <v, S> is the dot product along the softmax axis.

        Parameters
        ----------
        dA : ndarray
            Upstream gradient with respect to the softmax output (same shape as forward).

        Returns
        -------
        dZ : ndarray
            Gradient with respect to the logits Z.
        """
        if self._S is None:
            raise RuntimeError("Softmax.backward called before forward.")
        S = self._S
        dA = np.asarray(dA, dtype=np.float64)
        # sum over classes along the chosen axis
        dot = np.sum(dA * S, axis=self.axis, keepdims=True)
        dZ = (dA - dot) * S
        return dZ

# ---------------- Demo & self-check ----------------
def _demo():
    np.random.seed(0)
    logits = np.array([[2.0, 1.0, 0.1],
                       [0.5, 0.5, -1.0],
                       [3.0, 2.0, 2.0]], dtype=np.float64)  # (N=3, C=3)
    sm = Softmax(axis=1)
    probs = sm.forward(logits)
    print("Probabilities:\n", np.round(probs, 6))
    print("Row sums:     ", np.round(probs.sum(axis=1), 6))

    # Backward demo: check softmax + NLL gradient equivalence
    # Suppose one-hot labels:
    y_idx = np.array([0, 1, 2], dtype=int)             # class indices
    y_oh = np.zeros_like(probs)
    y_oh[np.arange(len(y_idx)), y_idx] = 1.0

    # For L = -sum(y * log S) (sum over classes per sample),
    # dL/dS = -y / S ; then dL/dZ = (S - y)
    dA = - y_oh / (probs + 1e-15)                      # dL/dS
    dZ = sm.backward(dA)                                # should equal (S - y)
    check = dZ - (probs - y_oh)
    print("\nMax abs diff between dZ and (S - y):", float(np.max(np.abs(check))))

if __name__ == "__main__":
    _demo()
