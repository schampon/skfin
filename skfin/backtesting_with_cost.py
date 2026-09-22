"""Walk-forward backtesting with quadratic transaction costs."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import TimeSeriesSplit

from skfin.estimators.mean_variance import MeanVariance


def average_holding_period(h: pd.DataFrame | pd.Series) -> float:
    """Compute average holding period from a holdings time series.

    Args:
        h: Holdings DataFrame (multi-asset) or Series (single-asset).

    Returns:
        Average holding period in the same time units as the index frequency.
    """
    h = h.squeeze()
    if isinstance(h, pd.Series):
        return h.abs().mean() / h.diff().abs().div(2).mean()
    return h.abs().sum(axis=1).mean() / h.diff().abs().sum(axis=1).div(2).mean()


def compute_pnl_components(
    h: pd.DataFrame,
    ret: pd.DataFrame,
    vol_liquidity_factor: pd.DataFrame | None = None,
) -> dict[str, pd.Series] | pd.Series:
    """Decompose PnL into gross, net, and impact cost components.

    Args:
        h: Holdings DataFrame with DatetimeIndex.
        ret: Asset returns (same columns as h).
        vol_liquidity_factor: Per-asset liquidity cost factor. None returns gross PnL only.

    Returns:
        Dict with keys "gross", "net = gross - impact cost", "impact cost" if
        vol_liquidity_factor is provided; otherwise a single PnL Series.
    """
    ret = ret[h.index[0]:h.index[-1]]
    pnl = h.shift(1).mul(ret).sum(axis=1)
    if vol_liquidity_factor is not None:
        vlf = vol_liquidity_factor.loc[h.index[0]:h.index[-1]]
        impact_cost = h.diff().pow(2).mul(vlf).sum(axis=1)
        return {
            "gross": pnl,
            "net = gross - impact cost": pnl.sub(impact_cost),
            "impact cost": -1 * impact_cost,
        }
    return pnl


def compute_batch_holdings_with_cost(
    pred: np.ndarray | pd.Series | pd.DataFrame,
    V: np.ndarray,
    A: np.ndarray | None = None,
    past_h: np.ndarray | None = None,
    vol_liquidity_factor: np.ndarray | None = None,
    lambda_: float | None = None,
    risk_target: float | None = None,
) -> np.ndarray:
    """Compute Markowitz holdings with quadratic transaction cost penalty.

    Args:
        pred: Return predictions, shape (N,) or (N, K) or (K, N).
        V: Covariance matrix, shape (N, N).
        A: Constraint matrix. None for unconstrained.
        past_h: Previous period holdings for cost penalty.
        vol_liquidity_factor: Per-asset cost factor (1D array of length N).
        lambda_: Risk tolerance parameter. Defaults to 1 when costs are active.
        risk_target: Target portfolio volatility (unused when costs are active).

    Returns:
        Holdings array, shape (K, N).
    """
    if lambda_ is None and vol_liquidity_factor is not None:
        lambda_ = 1
    N, _ = V.shape
    if isinstance(pred, (pd.Series, pd.DataFrame)):
        pred = pred.values
    if pred.shape == (N,):
        pred = pred[:, None]
    elif pred.shape[1] == N:
        pred = pred.T

    if vol_liquidity_factor is not None:
        invV = np.linalg.inv(V / lambda_ + 2 * np.diag(vol_liquidity_factor))
    else:
        invV = np.linalg.inv(V)

    if A is None:
        M = invV
    else:
        U = invV.dot(A)
        if A.ndim == 1:
            M = invV - np.outer(U, U.T) / U.dot(A)
        else:
            M = invV - U.dot(np.linalg.inv(U.T.dot(A)).dot(U.T))

    if vol_liquidity_factor is not None and past_h is not None:
        h = M.dot(pred + 2 * np.diag(vol_liquidity_factor).dot(past_h.T))
    else:
        h = M.dot(pred)
    return h.T


@dataclass
class MeanVarianceWithCost(MeanVariance):
    """Mean-variance optimizer with quadratic transaction cost penalty."""

    @staticmethod
    def compute_batch_holdings(
        pred: np.ndarray,
        V: np.ndarray,
        A: np.ndarray | None,
        **kwargs,
    ) -> np.ndarray:
        """Compute portfolio holdings considering transaction costs."""
        return compute_batch_holdings_with_cost(pred, V, A, **kwargs)


@dataclass
class BacktesterWithCost:
    """Walk-forward backtester that passes transaction cost parameters to the estimator."""

    estimator: BaseEstimator = None
    vol_liquidity_factor: pd.DataFrame | None = None
    max_train_size: int = 36
    test_size: int = 1
    start_date: str = "1945-01-01"
    end_date: str | None = None
    h_init: pd.Series | None = None
    return_pnl_component: bool = False

    def __post_init__(self):
        if self.estimator is None:
            self.estimator = MeanVarianceWithCost()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        ret: pd.DataFrame,
    ):
        """Run walk-forward backtest with sequential cost-aware optimization.

        Args:
            X: Feature matrix with DatetimeIndex.
            y: Target matrix for fitting.
            ret: Returns for PnL computation.

        Returns:
            self (with h_, pnl_ fitted attributes).
        """
        X = X.loc[:self.end_date]
        cv = TimeSeriesSplit(
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            n_splits=1 + len(X.loc[self.start_date:]) // self.test_size,
        )

        _h = []
        past_h = self.h_init
        for train, test in cv.split(X):
            m = clone(self.estimator)
            m.fit(X.iloc[train], y.iloc[train])
            if self.vol_liquidity_factor is None:
                vlf = None
            else:
                vlf = np.squeeze(self.vol_liquidity_factor.values[test])
            current_h = m.predict(
                X.iloc[test], past_h=past_h, vol_liquidity_factor=vlf
            )
            _h.append(current_h)
            past_h = current_h

        cols = X.columns
        idx = X.index[np.concatenate([test for _, test in cv.split(X)])]
        h_ = pd.DataFrame(np.concatenate(_h), index=idx, columns=cols)

        self.h_ = h_
        if self.return_pnl_component:
            self.pnl_ = compute_pnl_components(
                self.h_, ret, vol_liquidity_factor=self.vol_liquidity_factor
            )
        else:
            self.pnl_ = (
                h_.shift(1).mul(ret).sum(axis=1)[self.start_date:self.end_date]
            )
        return self
