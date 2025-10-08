"""
Input Normalization Algorithms (assignment-ready)

This single-file module implements common input normalization techniques with
clean NumPy code, plus streaming (online) statistics and PCA/ZCA whitening.

Included
--------
1) Standardization (z-score) with inverse_transform
2) Min–Max scaling with arbitrary feature range
3) Robust scaling (median + IQR) and (median + MAD)
4) L2 vector normalization (row-wise or column-wise)
5) PCA whitening and ZCA whitening with ε stabilization
6) Streaming (online) Normalizer using Welford for mean/variance
7) A Stateless functional API + Stateful "fit/transform" classes
8) Quick self-checks under __main__ to verify near-zero mean/unit-variance

All functions support arbitrary axes (defaults: axis=0 for "per-feature").
NaN-safe options are provided where helpful.

Only dependency: numpy
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np

Array = np.ndarray

#############################
# 1) Functional API         #
#############################

def _nanmean_std(x: Array, axis=0, ddof: int = 0, keepdims: bool = True) -> Tuple[Array, Array]:
    mean = np.nanmean(x, axis=axis, keepdims=keepdims)
    std = np.nanstd(x, axis=axis, ddof=ddof, keepdims=keepdims)
    return mean, std


def standardize(
    x: Array,
    axis: int | tuple[int, ...] = 0,
    with_mean: bool = True,
    with_std: bool = True,
    ddof: int = 0,
    eps: float = 1e-8,
    nan_safe: bool = False,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Z-score standardization along a given axis.

    Returns (x_scaled, stats) where stats contains 'mean' and 'std' for inverse.
    If with_mean=False or with_std=False, the corresponding step is skipped.
    """
    if nan_safe:
        mean, std = _nanmean_std(x, axis=axis, ddof=ddof)
    else:
        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, ddof=ddof, keepdims=True)

    x_scaled = x.astype(np.float64, copy=True)
    if with_mean:
        x_scaled = x_scaled - mean
    if with_std:
        x_scaled = x_scaled / (std + eps)
    return x_scaled, {"mean": np.squeeze(mean), "std": np.squeeze(std), "axis": axis, "with_mean": with_mean, "with_std": with_std, "eps": eps}


def inverse_standardize(x_scaled: Array, stats: Dict[str, Array]) -> Array:
    mean, std = stats["mean"], stats["std"]
    axis = stats.get("axis", 0)
    with_mean = stats.get("with_mean", True)
    with_std = stats.get("with_std", True)
    eps = stats.get("eps", 1e-8)

    mean = np.expand_dims(mean, axis=axis)
    std = np.expand_dims(std, axis=axis)

    x = x_scaled.astype(np.float64, copy=True)
    if with_std:
        x = x * (std + eps)
    if with_mean:
        x = x + mean
    return x


def minmax_scale(
    x: Array,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    axis: int | tuple[int, ...] = 0,
    clip: bool = False,
    eps: float = 1e-12,
    nan_safe: bool = False,
) -> Tuple[Array, Dict[str, Array]]:
    """Affine scale features to [a,b] per-axis, returning (x_scaled, stats)."""
    a, b = feature_range
    if nan_safe:
        minv = np.nanmin(x, axis=axis, keepdims=True)
        maxv = np.nanmax(x, axis=axis, keepdims=True)
    else:
        minv = np.min(x, axis=axis, keepdims=True)
        maxv = np.max(x, axis=axis, keepdims=True)
    denom = (maxv - minv)
    scale = (b - a) / (denom + eps)
    x_scaled = (x - minv) * scale + a
    if clip:
        x_scaled = np.clip(x_scaled, a, b)
    return x_scaled, {"min": np.squeeze(minv), "max": np.squeeze(maxv), "a": a, "b": b, "axis": axis, "eps": eps}


def inverse_minmax(x_scaled: Array, stats: Dict[str, Array]) -> Array:
    a, b = stats["a"], stats["b"]
    minv, maxv = stats["min"], stats["max"]
    eps = stats.get("eps", 1e-12)
    axis = stats.get("axis", 0)
    minv = np.expand_dims(minv, axis=axis)
    maxv = np.expand_dims(maxv, axis=axis)
    scale = (maxv - minv)
    x = (x_scaled - a) * (scale / (b - a + eps)) + minv
    return x


def robust_scale_iqr(
    x: Array,
    axis: int | tuple[int, ...] = 0,
    q_low: float = 25.0,
    q_high: float = 75.0,
    eps: float = 1e-8,
) -> Tuple[Array, Dict[str, Array]]:
    """Robust scaling using median and interquartile range (IQR)."""
    med = np.percentile(x, 50.0, axis=axis, keepdims=True)
    q1 = np.percentile(x, q_low, axis=axis, keepdims=True)
    q3 = np.percentile(x, q_high, axis=axis, keepdims=True)
    iqr = q3 - q1
    x_scaled = (x - med) / (iqr + eps)
    return x_scaled, {"median": np.squeeze(med), "iqr": np.squeeze(iqr), "axis": axis, "eps": eps}


def robust_scale_mad(
    x: Array,
    axis: int | tuple[int, ...] = 0,
    eps: float = 1e-8,
    c: float = 1.4826,
) -> Tuple[Array, Dict[str, Array]]:
    """Robust scaling using median and MAD (normalized for Gaussian via c)."""
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    x_scaled = (x - med) / (c * mad + eps)
    return x_scaled, {"median": np.squeeze(med), "mad": np.squeeze(mad), "axis": axis, "eps": eps, "c": c}


def l2_normalize(x: Array, axis: int = 1, eps: float = 1e-12) -> Array:
    """Normalize vectors along `axis` to unit L2 norm."""
    denom = np.sqrt(np.sum(x * x, axis=axis, keepdims=True))
    return x / (denom + eps)


################################
# 2) Whitening (PCA / ZCA)     #
################################

def _center(x: Array, mean: Optional[Array], axis: int) -> Tuple[Array, Array]:
    if mean is None:
        mean = np.mean(x, axis=axis, keepdims=True)
    return x - mean, np.squeeze(mean)


def pca_whiten_fit(x: Array, axis: int = 0, eps: float = 1e-5) -> Dict[str, Array]:
    """
    Fit PCA whitening on data matrix x (samples x features if axis=0).
    Returns dict containing mean, eigvecs, eigvals.
    """
    if axis != 0:
        x = np.swapaxes(x, axis, 0)
    x0, mean = _center(x, None, axis=0)
    # Covariance over features (columns)
    cov = (x0.T @ x0) / (x0.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return {"mean": mean, "eigvecs": eigvecs, "eigvals": eigvals, "eps": eps, "axis": 0}


def pca_whiten_transform(x: Array, params: Dict[str, Array], center: bool = True, axis: int = 0) -> Array:
    if axis != 0:
        x = np.swapaxes(x, axis, 0)
    mean = params["mean"] if center else 0.0
    eigvecs = params["eigvecs"]
    eigvals = params["eigvals"]
    eps = params.get("eps", 1e-5)
    x0 = x - mean
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
    xw = x0 @ W
    if axis != 0:
        xw = np.swapaxes(xw, 0, axis)
    return xw


def zca_whiten_transform(x: Array, params: Dict[str, Array], center: bool = True, axis: int = 0) -> Array:
    if axis != 0:
        x = np.swapaxes(x, axis, 0)
    mean = params["mean"] if center else 0.0
    eigvecs = params["eigvecs"]
    eigvals = params["eigvals"]
    eps = params.get("eps", 1e-5)
    x0 = x - mean
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
    xz = x0 @ W  # ZCA == PCA-whiten and rotate back by eigvecs
    if axis != 0:
        xz = np.swapaxes(xz, 0, axis)
    return xz


############################################
# 3) Stateful classes: fit/transform       #
############################################

@dataclass
class StandardScaler:
    with_mean: bool = True
    with_std: bool = True
    ddof: int = 0
    eps: float = 1e-8

    mean_: Optional[Array] = None
    std_: Optional[Array] = None
    axis: int = 0

    def fit(self, x: Array) -> "StandardScaler":
        self.mean_ = np.mean(x, axis=self.axis, keepdims=False)
        self.std_ = np.std(x, axis=self.axis, ddof=self.ddof, keepdims=False)
        return self

    def transform(self, x: Array) -> Array:
        assert self.mean_ is not None and self.std_ is not None, "Call fit first"
        mean = np.expand_dims(self.mean_, axis=self.axis)
        std = np.expand_dims(self.std_, axis=self.axis)
        y = x.astype(np.float64, copy=True)
        if self.with_mean:
            y -= mean
        if self.with_std:
            y /= (std + self.eps)
        return y

    def inverse_transform(self, y: Array) -> Array:
        assert self.mean_ is not None and self.std_ is not None, "Call fit first"
        mean = np.expand_dims(self.mean_, axis=self.axis)
        std = np.expand_dims(self.std_, axis=self.axis)
        x = y.astype(np.float64, copy=True)
        if self.with_std:
            x *= (std + self.eps)
        if self.with_mean:
            x += mean
        return x


@dataclass
class MinMaxScaler:
    feature_range: Tuple[float, float] = (0.0, 1.0)
    eps: float = 1e-12
    axis: int = 0

    min_: Optional[Array] = None
    max_: Optional[Array] = None

    def fit(self, x: Array) -> "MinMaxScaler":
        self.min_ = np.min(x, axis=self.axis, keepdims=False)
        self.max_ = np.max(x, axis=self.axis, keepdims=False)
        return self

    def transform(self, x: Array) -> Array:
        assert self.min_ is not None and self.max_ is not None, "Call fit first"
        a, b = self.feature_range
        minv = np.expand_dims(self.min_, axis=self.axis)
        maxv = np.expand_dims(self.max_, axis=self.axis)
        y = (x - minv) * ((b - a) / (maxv - minv + self.eps)) + a
        return y

    def inverse_transform(self, y: Array) -> Array:
        assert self.min_ is not None and self.max_ is not None, "Call fit first"
        a, b = self.feature_range
        minv = np.expand_dims(self.min_, axis=self.axis)
        maxv = np.expand_dims(self.max_, axis=self.axis)
        x = (y - a) * ((maxv - minv) / (b - a + self.eps)) + minv
        return x


############################################
# 4) Streaming Normalizer (Welford)        #
############################################

class OnlineStandardizer:
    """
    Streaming per-feature mean/variance with Welford's algorithm.
    Use update(batch) repeatedly, then transform(x) when needed.
    """
    def __init__(self, n_features: Optional[int] = None, eps: float = 1e-8):
        self.n = 0
        self.mean = None if n_features is None else np.zeros((n_features,), dtype=np.float64)
        self.M2 = None if n_features is None else np.zeros((n_features,), dtype=np.float64)
        self.eps = eps

    def update(self, x: Array):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.mean is None:
            self.mean = np.zeros((x.shape[1],), dtype=np.float64)
            self.M2 = np.zeros((x.shape[1],), dtype=np.float64)
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.M2 += delta * delta2

    @property
    def var(self) -> Array:
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> Array:
        return np.sqrt(self.var)

    def transform(self, x: Array) -> Array:
        assert self.mean is not None, "Call update() with data first"
        return (x - self.mean) / (self.std + self.eps)


############################################
# 5) Quick self-checks                      #
############################################

def _check_standardize():
    rng = np.random.default_rng(0)
    X = rng.normal(3.0, 5.0, size=(10000, 5))
    Xs, stats = standardize(X, axis=0)
    mu = np.mean(Xs, axis=0)
    sd = np.std(Xs, axis=0)
    return float(np.max(np.abs(mu))), float(np.max(np.abs(sd - 1.0)))


def _check_minmax():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 3))
    Y, s = minmax_scale(X, feature_range=(-1, 1))
    return float(np.min(Y)), float(np.max(Y))


def _check_whiten():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 3)) @ np.array([[3,0,0],[1,2,0],[0,1,0.5]])  # correlated
    params = pca_whiten_fit(X, axis=0, eps=1e-5)
    Xw = pca_whiten_transform(X, params)
    C = np.cov(Xw.T)
    # off-diagonal magnitude should be small; diagonal ~ 1
    off_diag = np.copy(C)
    np.fill_diagonal(off_diag, 0.0)
    return float(np.max(np.abs(off_diag))), float(np.max(np.abs(np.diag(C) - 1.0)))


if __name__ == "__main__":
    print("Standardize max |mean|, max |std-1|:", _check_standardize())
    print("MinMax range (min,max):", _check_minmax())
    print("PCA whiten offdiag max, diag-1 max:", _check_whiten())
