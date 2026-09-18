"""Mean-variance portfolio optimization estimators."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from skfin.metrics import sharpe_ratio


def compute_batch_holdings(
    pred: np.ndarray | pd.Series | pd.DataFrame,
    V: np.ndarray,
    A: np.ndarray | None = None,
    risk_target: float | None = None,
) -> np.ndarray:
    """Compute Markowitz portfolio holdings from predictions and covariance.

    Args:
        pred: Return predictions, shape (N,) or (N, K) or (K, N).
        V: Covariance matrix, shape (N, N).
        A: Constraint matrix. None for unconstrained.
        risk_target: Target portfolio volatility. None disables scaling.

    Returns:
        Holdings array, shape (K, N).
    """
    N, _ = V.shape
    if isinstance(pred, (pd.Series, pd.DataFrame)):
        pred = pred.values
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
    elif pred.shape[1] == N:
        pred = pred.T

    invV = np.linalg.inv(V)
    if A is None:
        M = invV
    else:
        U = invV.dot(A)
        if A.ndim == 1:
            M = invV - np.outer(U, U.T) / U.dot(A)
        else:
            M = invV - U.dot(np.linalg.inv(U.T.dot(A)).dot(U.T))
    h = M.dot(pred)
    if risk_target is not None:
        h = risk_target * h / np.sqrt(np.diag(h.T.dot(V.dot(h))))
    return h.T


@dataclass
class MeanVariance(BaseEstimator):
    """Mean-variance optimizer that computes portfolio holdings from predictions and covariance."""

    transform_V: Callable = field(default=lambda x: np.cov(x.T))
    A: np.ndarray | str | None = "cash-neutral"
    risk_target: float = 1.0

    def __post_init__(self):
        self.holdings_kwargs = {"risk_target": self.risk_target}

    @staticmethod
    def compute_batch_holdings(
        pred: np.ndarray,
        V: np.ndarray,
        A: np.ndarray | None,
        risk_target: float | None,
        **kwargs,
    ) -> np.ndarray:
        """Compute portfolio holdings in batch."""
        return compute_batch_holdings(pred=pred, V=V, A=A, risk_target=risk_target, **kwargs)

    def fit(self, X, y=None):
        self.V_ = self.transform_V(y)
        return self

    def predict(self, X, **kwargs):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values
        n_assets = X.shape[1] if X.ndim > 1 else X.shape[0]
        if isinstance(self.A, str) and self.A == "cash-neutral":
            A = np.ones((n_assets, 1))
        else:
            A = self.A
        kwargs = {**kwargs, **self.holdings_kwargs}
        h = self.compute_batch_holdings(pred=X, V=self.V_, A=A, **kwargs)
        return h

    def transform(self, X, **kwargs):
        """Alias for predict (enables sklearn pipeline use)."""
        return self.predict(X, **kwargs)

    def score(self, X, y):
        """Compute Sharpe ratio of the portfolio PnL."""
        return sharpe_ratio(np.sum(X * y, axis=1))


class TimingMeanVariance(BaseEstimator):
    """Single-asset timing estimator that scales positions by inverse volatility."""

    def __init__(
        self,
        transform_V: Callable | None = None,
        a_min: float | None = None,
        a_max: float | None = None,
    ):
        if transform_V is None:
            self.transform_V = lambda x: np.var(x)
        else:
            self.transform_V = transform_V
        self.a_min = a_min
        self.a_max = a_max

    def fit(self, X, y=None):
        self.V_ = self.transform_V(y)
        return self

    def predict(self, X):
        if self.a_min is None and self.a_max is None:
            h = X / self.V_
        else:
            h = np.clip(
                X / np.sqrt(self.V_), a_min=self.a_min, a_max=self.a_max
            ) / np.sqrt(self.V_)
        return h

    def transform(self, X):
        """Alias for predict (enables sklearn pipeline use)."""
        return self.predict(X)
